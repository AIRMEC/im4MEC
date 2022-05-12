import itertools
import os
import random
import subprocess
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    WeightedRandomSampler,
)
from torch.utils.tensorboard import SummaryWriter

from model import AttentionNet


def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]


def get_feature_bag_path(data_dir, slide_id):
    return os.path.join(data_dir, f"{slide_id}_features.h5")


class FeatureBagsDataset(Dataset):
    def __init__(self, df, data_dir):
        self.slide_df = df.copy().reset_index(drop=True)
        self.data_dir = data_dir

    def __getitem__(self, idx):
        slide_id = self.slide_df["slide_id"][idx]
        label = self.slide_df["label"][idx]

        full_path = get_feature_bag_path(self.data_dir, slide_id)
        with h5py.File(full_path, "r") as hdf5_file:
            features = hdf5_file["features"][:]
            coords = hdf5_file["coords"][:]

        features = torch.from_numpy(features)
        return features, label, coords

    def __len__(self):
        return len(self.slide_df)


def evaluate_model(model, loader, n_classes, loss_fn, device):
    model.eval()

    avg_loss = 0.0

    preds = np.zeros(len(loader))
    probs = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, _ = model(data)

            loss = loss_fn(logits, label)
            avg_loss += loss.item()

            preds[batch_idx] = Y_hat.item()
            probs[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

    avg_loss /= len(loader)

    return preds, probs, labels, avg_loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, min_epochs=50, verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_epochs (int): Earliest epoch possible for stopping.
            verbose (bool): If True, prints messages for e.g. each validation loss improvement.
        """
        self.patience = patience
        self.min_epochs = min_epochs
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience and epoch > self.min_epochs:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def compute_auc(labels, probs):
    assert probs.shape[0] > 0
    assert probs.shape[1] > 1

    if probs.shape[1] == 2:
        raise Exception(
            "If you are doing binary classification, make sure to revisit the applicability of AUC macro-averaging."
        )

    return roc_auc_score(labels, probs, multi_class="ovr", average="macro")


def compute_auc_each_class(labels, probs):
    # Per-class AUC in a multi-class context.
    assert probs.shape[0] > 0
    assert (
        probs.shape[1] > 2
    ), "This function is only relevant for multi-class (non-binary) tasks."
    return [roc_auc_score(labels == i, probs[:, i]) for i in range(probs.shape[1])]


def render_confusion_matrix(cm, class_names, normalize=False):
    """Render confusion matrix as a matplotlib figure."""
    title = "Confusion matrix"
    cmap = plt.cm.Blues
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    figure = plt.figure(figsize=(8, 8))
    vmax = 1 if normalize else None
    plt.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    return figure


def run_train_eval_loop(
    train_loader,
    val_loader,
    input_feature_size,
    class_names,
    hparams,
    run_id,
    full_training,
    save_checkpoints,
):
    writer = SummaryWriter(os.path.join("./runs", run_id))
    device = torch.device("cuda")
    loss_fn = torch.nn.CrossEntropyLoss()
    n_classes = len(class_names)
    model = AttentionNet(
        model_size=hparams["model_size"],
        input_feature_size=input_feature_size,
        dropout=True,
        p_dropout_fc=hparams["p_dropout_fc"],
        p_dropout_atn=hparams["p_dropout_atn"],
        n_classes=n_classes,
    )
    model.to(device)
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_trainable_params} parameters")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hparams["initial_lr"],
        weight_decay=hparams["weight_decay"],
    )

    # Using a multi-step LR decay routine.
    milestones = [int(x) for x in hparams["milestones"].split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=hparams["gamma_lr"]
    )

    early_stop_tracker = EarlyStopping(
        patience=hparams["earlystop_patience"],
        min_epochs=hparams["earlystop_min_epochs"],
        verbose=True,
    )

    metric_history = []
    for epoch in range(hparams["max_epochs"]):
        model.train()
        epoch_start_time = time.time()
        train_loss = 0.0
        preds = np.zeros(len(train_loader))
        probs = np.zeros((len(train_loader), n_classes))
        labels = np.zeros(len(train_loader))

        batch_start_time = time.time()
        for batch_idx, (data, label) in enumerate(train_loader):
            data_load_duration = time.time() - batch_start_time

            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, _ = model(data)

            preds[batch_idx] = Y_hat.item()
            probs[batch_idx] = Y_prob.cpu().detach().numpy()
            labels[batch_idx] = label.item()

            loss = loss_fn(logits, label)
            train_loss += loss.item()

            # backward pass
            loss.backward()

            # step
            optimizer.step()
            optimizer.zero_grad()

            batch_duration = time.time() - batch_start_time
            batch_start_time = time.time()

            print(
                f"epoch {epoch}, batch {batch_idx}, batch took: {batch_duration:.2f}s, data loading: {data_load_duration:.2f}s, loss: {loss.item():.4f}, label: {label.item()}"
            )
            writer.add_scalar("data_load_duration", data_load_duration, epoch)
            writer.add_scalar("batch_duration", batch_duration, epoch)

        epoch_duration = time.time() - epoch_start_time
        print(f"Finished training on epoch {epoch} in {epoch_duration:.2f}s")

        train_loss /= len(train_loader)
        train_avg_auc = compute_auc(labels, probs)

        writer.add_scalar("epoch_duration", epoch_duration, epoch)
        writer.add_scalar("LR", get_lr(optimizer), epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("AUC/train", train_avg_auc, epoch)

        if n_classes > 2:
            train_single_aucs = compute_auc_each_class(labels, probs)
            for class_index in range(n_classes):
                writer.add_scalar(
                    f"AUC/train-{class_names[class_index]}",
                    train_single_aucs[class_index],
                    epoch,
                )

        for class_index in range(n_classes):
            writer.add_pr_curve(
                f"PRcurve/train-{class_names[class_index]}",
                labels == class_index,
                probs[:, class_index],
                epoch,
            )

        if not full_training:
            print("Evaluating model on validation set...")
            preds, probs, labels, val_loss = evaluate_model(
                model, val_loader, n_classes, loss_fn, device
            )
            val_avg_auc = compute_auc(labels, probs)

            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("AUC/validation", val_avg_auc, epoch)

            for class_index in range(n_classes):
                writer.add_pr_curve(
                    f"PRcurve/validation-{class_names[class_index]}",
                    labels == class_index,
                    probs[:, class_index],
                    epoch,
                )

            metric_dict = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_inst_loss": 0,
                "val_composite_loss": 0,
                "val_auc": val_avg_auc,
                "trainable_params": n_trainable_params,
            }

            if n_classes > 2:
                val_single_aucs = compute_auc_each_class(labels, probs)
                for class_index in range(n_classes):
                    writer.add_scalar(
                        f"AUC/validation-{class_names[class_index]}",
                        val_single_aucs[class_index],
                        epoch,
                    )
                for idx, each_auc_class in enumerate(val_single_aucs):
                    metric_dict[f"val_auc_{class_names[idx]}"] = each_auc_class

                cm = confusion_matrix(
                    [class_names[l] for l in labels.astype(int)],
                    [class_names[p] for p in preds.astype(int)],
                    labels=class_names,
                )
                writer.add_figure(
                    "Confusion matrix",
                    render_confusion_matrix(cm, class_names, normalize=False),
                    epoch,
                )
                writer.add_figure(
                    "Normalized confusion matrix",
                    render_confusion_matrix(cm, class_names, normalize=True),
                    epoch,
                )

            metric_history.append(metric_dict)
            early_stop_tracker(epoch, val_loss)

        if save_checkpoints:
            torch.save(
                model.state_dict(),
                os.path.join(writer.log_dir, f"{epoch}_checkpoint.pt"),
            )

        # Update LR decay.
        scheduler.step()

        if early_stop_tracker.early_stop:
            print(
                f"Early stop criterion reached. Broke off training loop after epoch {epoch}."
            )
            break

    if not full_training:
        # Log the hyperparameters of this experiment and the performance metrics of the best epoch.
        best = sorted(metric_history, key=lambda x: x["val_loss"])[0]
        writer.add_hparams(hparams, best)

    writer.close()


def define_data_sampling(train_split, val_split, method, workers):
    # Reproducibility of DataLoader
    g = torch.Generator()
    g.manual_seed(0)

    # Set up training data sampler.
    if method == "random":
        print("random sampling setting")
        train_loader = DataLoader(
            dataset=train_split,
            batch_size=1,  # model expects one bag of features at the time.
            shuffle=False,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    elif method == "balanced":
        print("balanced sampling setting")
        train_labels = train_split.slide_df["label"]

        # Compute sample weights to alleviate class imbalance with weighted sampling.
        sample_weights = compute_sample_weight("balanced", train_labels)

        train_loader = DataLoader(
            dataset=train_split,
            batch_size=1,  # model expects one bag of features at the time.
            # Use the weighted sampler using the precomputed sample weights.
            # Note that replacement is true by default, so
            # some slides of rare classes will be sampled multiple times per epoch.
            shuffle=False,
            sampler=WeightedRandomSampler(sample_weights, len(sample_weights)),
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        raise Exception(f"Sampling method '{method}' not implemented.")

    # val_split would be an empty list if not validation is asked in training.
    if len(val_split) == 0:
        val_loader = val_split
    else:
        val_loader = DataLoader(
            dataset=val_split,
            batch_size=1,  # model expects one bag of features at the time.
            sampler=SequentialSampler(val_split),
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    return train_loader, val_loader


def get_class_names(df):
    n_classes = len(df["label"].unique())
    class_names = [None] * n_classes
    for i in df["label"].unique():
        class_names[i] = df[df["label"] == i]["class"].unique()[0]
    assert len(class_names) == n_classes
    return class_names


def main(args):

    # Set random seed for some degree of reproducibility. See PyTorch docs on this topic for caveats.
    # https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    set_seed()

    if not torch.cuda.is_available():
        raise Exception(
            "No CUDA device available. Training without one is not feasible."
        )

    df = pd.read_csv(args.manifest)
    class_names = get_class_names(df)
    fold_index = str(args.fold)

    if args.full_training is not None:
        print(
            f"Training on full dataset (training + validation) with hparam set {args.full_training}"
        )
        if args.fold is not None:
            raise Exception(
                "Both --full_training and --fold have been provided. These arguments are mutually exclusive."
            )
        training_set = df
        val_split = [None]
        base_run_id = f"full_dataset"
    else:
        print(f"=> Fold {fold_index}")
        base_run_id = f"fold_{fold_index}"
        try:
            training_set = df[df[f"fold-{fold_index}"] == "training"]
            validation_set = df[df[f"fold-{fold_index}"] == "validation"]
        except:
            raise Exception(
                f"Column fold-{fold_index} does not exist in {args.manifest}"
            )

        val_split = FeatureBagsDataset(validation_set, args.data_dir)

    train_split = FeatureBagsDataset(training_set, args.data_dir)

    git_sha = (
        subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
    )
    train_run_id = f"{git_sha}_{time.strftime('%Y%m%d-%H%M')}"

    print(f"=> Git SHA {train_run_id}")
    print(f"=> Training on {len(train_split)} samples")
    print(f"=> Validating on {len(val_split)} samples")

    base_hparams = dict(
        sampling_method="random",
        max_epochs=100,
        earlystop_patience=20,
        earlystop_min_epochs=20,
        # Optimizer settings
        initial_lr=1e-3,
        milestones="2, 5, 15, 30",
        gamma_lr=0.1,
        weight_decay=1e-5,
        # Model architecture parameters. See model class for details.
        model_size="small",
        p_dropout_fc=0.5,
        p_dropout_atn=0.25,
    )

    hparam_sets = [
        base_hparams,
        {
            **base_hparams,
            "initial_lr": 1e-4,
            "milestones": "5, 15, 30",
        },
        {
            **base_hparams,
            "initial_lr": 1e-5,
            "milestones": "10, 30",
        },
        {
            **base_hparams,
            "weight_decay": 1e-3,
        },
    ]

    hparams_to_use = hparam_sets
    if args.full_training is not None:
        hparams_to_use = [hparam_sets[args.full_training]]

    for i, hps in enumerate(hparams_to_use):
        run_id = f"{base_run_id}_{hps['model_size']}_{hps['sampling_method']}_hp{i}_{train_run_id}"
        print(f"Running train-eval loop {i} for {run_id}")
        print(hps)

        train_loader, val_loader = define_data_sampling(
            train_split,
            val_split,
            method=hps["sampling_method"],
            workers=args.workers,
        )

        run_train_eval_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            input_feature_size=args.input_feature_size,
            class_names=class_names,
            hparams=hps,
            run_id=run_id,
            full_training=args.full_training is not None,
            save_checkpoints=args.full_training is not None,
        )

    print("Finished training.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--manifest",
        type=str,
        help="CSV file listing all slides, their labels, and which split (train/test/val) they belong to.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="Index of the fold in cross-validation",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where all *_features.h5 files are stored",
    )
    parser.add_argument(
        "--input_feature_size",
        help="The size of the input features from the feature bags.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loaders.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--full_training",
        type=int,
        help="Provide an index of the hyperparameter set you want to use to train the final model on the combined training and validation sets.",
    )
    args = parser.parse_args()

    main(args)

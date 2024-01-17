# import packages
import os
import random
import albumentations as albu
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchsummary import summary
import numpy as np
import time
import sys

# import code
sys.path.append("..")  # Add parent directory to Python path
import config  # noqa: E402
from utils.utils_run_data_sampling import data_sampling_function
from utils._preprocessing import preprocess_input
from utils.utils_dataset import Dataset
import utils.utils_metrics as utils_metrics
from utils.utils_augmentations import get_training_augmentation
from utils.utils_plotter import visualize
import utils.utils_train_epoch as train

# import models
from models import STnet
from sklearn.model_selection import train_test_split


def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_tensor(x, **kwargs):
    """Convert image into torch.Tensor"""
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def loocv_sampler(df, city_balanced=False):
    """Sample dataset for LOOCV"""
    # get number of samples per class
    count_classes = df.groupby("class").size().tolist()

    if city_balanced:
        count_per_class = 200
    else:
        count_per_class = int(count_classes[-1])
    # group by class
    group_classes = df.groupby(["class"])
    # get random samples per class
    classes_ix = np.hstack(
        [
            np.random.choice(v, count_per_class, replace=True)
            for v in group_classes.groups.values()
        ]
    )
    df_x_ = df.loc[classes_ix]
    return df_x_


def Trainer(
    epochs=10,
    lr=1e-5,
    augmentations="weak",
    LSP=25,
    mode="balanced_big",
    traintestval_split="random",
    LOOCV_city=None,
    shots_per_class=None,
    seed=config.SEED,
    load_weights=None,
    modelpath_save=None,
    warmup=True,
):
    # set seed
    if seed is None:
        set_seed(config.SEED)
    else:
        set_seed(seed)

    # select method to split dataset into train, test and validation
    df = data_sampling_function(LSP=LSP, mode="unbalanced")

    if traintestval_split == "LOOCV_pretrain":
        # check if LOOCV is selected
        if len(config.LOOCV_CITY) == 0:
            print("select LOOCV city")
            print(1 / 0)

        # get dataframe without LOOCV city
        df_LOOCV = df.loc[(df["city"] != LOOCV_city)]
        df_city = df.loc[(df["city"] == LOOCV_city)]

        if mode == "balanced_city":
            df_train = loocv_sampler(df_LOOCV)
        elif mode == "balanced_class":
            df_train = loocv_sampler(df_LOOCV, city_balanced=True)
        else:
            df_train = df_LOOCV

        df_val = loocv_sampler(df_city)

        train_images_dir_list = df_train["path"].tolist()
        train_class_values_list = df_train["class"].tolist()

        val_images_dir_list = df_val["path"].tolist()
        val_class_values_list = df_val["class"].tolist()
    elif traintestval_split == "2fold":
        # Split the DataFrame into train and test sets
        df_city = df.loc[(df["city"] == LOOCV_city)]
        print(df_city.head())
        df_train, df_val = train_test_split(
            df_city, test_size=0.5, stratify=df_city["class"], random_state=42
        )

        train_images_dir_list = df_train["path"].tolist()
        train_class_values_list = df_train["class"].tolist()

        val_images_dir_list = df_val["path"].tolist()
        val_class_values_list = df_val["class"].tolist()
    else:
        train_images_dir_list = []
        train_class_values_list = []

        val_images_dir_list = []
        val_class_values_list = []
        print('choose traintestval split from "4fold" or "LOOCV_pretrain"')
        print(1 / 0)

    # get train dataset
    train_dataset = Dataset(
        train_images_dir_list,
        train_class_values_list,
        augmentation=get_training_augmentation(augmentations),
        preprocessing=get_preprocessing(preprocess_input),
    )

    if config.VERBOSE >= 3:
        randomlist = random.sample(range(0, len(train_images_dir_list)), 5)
        for i in randomlist:
            image, label, _ = train_dataset[i]
            visualize(image=image, label=label)

    # get validation dataset
    valid_dataset = Dataset(
        val_images_dir_list,
        val_class_values_list,
        augmentation=get_training_augmentation(augmentations),
        preprocessing=get_preprocessing(preprocess_input),
    )

    # load pytorch dataset
    if shots_per_class is not None and config.BATCH_SIZE > shots_per_class:
        newBatchSize = int(shots_per_class * config.CLASSES)
        train_loader = DataLoader(
            train_dataset,
            batch_size=newBatchSize,
            shuffle=True,
            num_workers=10,
            drop_last=False,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=newBatchSize, shuffle=False, num_workers=10
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=10,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=10
        )

    # Create model
    device = torch.device("cuda")
    if config.MODEL == "STnet":
        model = STnet.STnet(input_channel=3, num_classes=config.CLASSES)
    else:
        print("select model")
        print(1 / 0)

    if config.VERBOSE >= 2:
        print(summary(model, (3, config.IMAGESIZE, config.IMAGESIZE), device="cpu"))

    if config.WEIGHTED_LOSS:
        # get class weights for weighted loss
        _, counts = np.unique(train_class_values_list, return_counts=True)
        # get normalized weights
        normed_weights = [
            np.sum(counts) / (count_class * len(counts)) for count_class in counts
        ]
        # scale weights to sum to 1
        normed_weights_scaled = [f / sum(normed_weights) for f in normed_weights]
        # convert to tensor
        weights_tensor = torch.FloatTensor(normed_weights_scaled)
    else:
        weights_tensor = torch.FloatTensor([1.0, 1.0, 1.0])

    loss = CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
    loss.__name__ = "loss"

    if config.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(
            [
                dict(params=model.parameters(), lr=lr),
            ]
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                dict(params=model.parameters(), lr=lr, weight_decay=0.0001),
            ]
        )

    if load_weights is None:
        print("Start training from scratch")
    else:
        model.to(device)
        checkpoint = torch.load(load_weights)
        print("loading model {}".format(load_weights))
        model.load_state_dict(checkpoint["model_state_dict"])
        if "fine" not in modelpath_save:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    metrics = [
        utils_metrics.Accuracy(threshold=0.5),
        utils_metrics.Fscore(threshold=0.5),
        utils_metrics.Precision(threshold=0.5),
        utils_metrics.Recall(threshold=0.5),
    ]

    train_epoch = train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device="cuda",
        verbose=True,
    )

    valid_epoch = train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device="cuda",
        verbose=True,
    )

    max_score = 0
    early_stop_i = 0
    time_start = time.perf_counter()
    for i in range(0, epochs):
        if warmup and i < config.WARM_UP_EPOCHS:
            optimizer.param_groups[0]["lr"] = config.WARM_UP_LEARNING_RATE
            print("Current learning rate {}".format(optimizer.param_groups[0]["lr"]))
            print("Epoch: {} of {}".format(i, epochs))
            _ = train_epoch.run(train_loader)
            continue

        optimizer.param_groups[0]["lr"] = lr
        print("Current learning rate {}".format(optimizer.param_groups[0]["lr"]))
        print("Epoch: {} of {}".format(i, epochs))
        _ = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # save model
        if max_score < valid_logs["fscore"] and i > config.WARM_UP_EPOCHS + 3:
            max_score = valid_logs["fscore"]
            time_saved_last_epoch = time.perf_counter()

            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                modelpath_save,
            )
            print("Model saved to {} in epoch {}".format(modelpath_save, i))

            # reset early stoping iterator
            early_stop_i = 0
        early_stop_i += 1
        if early_stop_i > config.EARLY_STOPPING:
            break

    print("Processing time: {}".format((time_saved_last_epoch - time_start)))


if __name__ == "__main__":
    Trainer()

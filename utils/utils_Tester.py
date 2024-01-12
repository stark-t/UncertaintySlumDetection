# import packages
import os
import random
import albumentations as albu
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchsummary import summary
import numpy as np
import pandas as pd
import tqdm
import time


# import code
import config as config
from run_data_sampling import data_sampling_function
from _preprocessing import preprocess_input
from utils_dataset import Dataset
import utils_functional as F
from utils_augmentations import get_training_augmentation, get_validation_augmentation
from utils_plotter import visualize
from models import LSPXSnet_torch
from models import STnet
import utils_train_epoch as train


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_tensor(x, **kwargs):
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


def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith("Dropout"):
            each_module.train()


def get_var_metrics(y_pred_mc, label_array):
    accuracies = []
    for i in range(len(y_pred_mc)):
        accuracy = F.accuracy(
            torch.from_numpy(y_pred_mc[i]).to(torch.float32).squeeze(),
            torch.from_numpy(label_array).to(torch.float32),
            threshold=0.5,
            # ignore_channels=[0, 1],  # None
        )
        accuracies.append(accuracy)
    return np.var(accuracies)


def Tester(
    mci=1,
    LSP=50,
    mode="unbalanced",
    LOOCV_city=None,
    shots_per_class=None,
    seed=None,
    modelpath_base="1",
):
    """

    :param model:
    :param mci:
    :return:
    """

    if seed is None:
        set_seed(config.seed)
    else:
        set_seed(seed)

    # get sampled dataset
    df = data_sampling_function(LSP=LSP, mode=mode, seed=seed)

    # check if LOOCV is selected
    if len(config.LOOCV) == 0:
        print("select LOOCV city")
        error = 1 / 0

    # get dataframe for LOOCV city
    df_loocv = df.loc[(df["city"] == LOOCV_city)]

    # print stats if necessary
    if config.verbose >= 2:
        print_df = df_loocv.groupby(["class"])["ID"].count()
        print(print_df)

    test_images_dir_list = df_loocv["path"].tolist()
    test_class_values_list = df_loocv["class"].tolist()

    # get validation dataset
    test_dataset = Dataset(
        test_images_dir_list,
        test_class_values_list,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    # # Create model
    device = torch.device("cuda")
    if config.model == "LSPXSnet":
        model = LSPXSnet_torch.LSPXSnet(input_channel=3, num_classes=config.CLASSES)
    elif config.model == "STnet":
        model = STnet.STnet(input_channel=3, num_classes=config.CLASSES)
    elif config.model == "LSPXSnet_MC":
        # model = LSPXSnet_MC_torch.LSPXSnet_MC(input_channel=3, num_classes=config.CLASSES)
        model = "missing"
    elif config.model == "Xception":
        # model = xception_torch.Xception(input_channel=3, num_classes=config.CLASSES)
        model = "missing"
    elif config.model == "ResNet50":
        model = "missing"
        # model = resnet.resnet50(input_shape=[config.imagesize, config.imagesize, 3], n_classes=config.n_classes)
    else:
        model = 1

    loss = CrossEntropyLoss(label_smoothing=0.1)
    loss.__name__ = "loss"

    model.to(device)
    if not "base" in modelpath_base:
        model_info = (
            str(config.model)
            + "_"
            + LOOCV_city
            + "_"
            + str(config.imagesize)
            + "px"
            + "_"
            + str(shots_per_class)
            + "shots"
        )
    else:
        model_info = str(config.model) + "_" + str(config.imagesize) + "px"

    if config.mulitstage:
        modelpath = os.path.join(modelpath_base, (model_info + "_MS.pth"))
        checkpoint = torch.load(modelpath)
        print("loading model {}".format(modelpath))
    else:
        modelpath = os.path.join(modelpath_base, (model_info + ".pth"))
        checkpoint = torch.load(modelpath)
        print("loading model {}".format(modelpath))

    model.load_state_dict(checkpoint["model_state_dict"])

    y_pr_tensor_list = []
    y_tensor_list = []
    metrics_variance = []

    tic = time.time()

    model.eval()
    enable_dropout(model)
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(test_images_dir_list))):
            # get image tiles
            image_tile, label_array, _ = test_dataset[i]
            label_onehot_list = list(label_array)
            label = label_onehot_list.index(max(label_onehot_list))

            x_tensor = torch.from_numpy(image_tile).to("cuda").unsqueeze(0)
            x_tensor = x_tensor.to(torch.float32)
            y_pred_mc = []
            # y_prob_slums_mc_list = []

            for mc_i in range(mci):
                pr_tensor_mc_i = model(x_tensor)
                pred = pr_tensor_mc_i.cpu().numpy()
                # y_pred_mc_i_tensor = torch.argmax(pr_tensor_mc_i)
                # y_pred_mc_i = y_pred_mc_i_tensor.cpu().numpy()
                y_pred_mc.append(pred)

            variance_metrics = get_var_metrics(y_pred_mc, label_array)

            y_pred_arr = np.vstack(y_pred_mc)
            y_pred_mc_mean = np.mean(y_pred_arr, axis=0)
            y_pred_mc_var = np.var(y_pred_arr, axis=0)

            y_pr_tensor = torch.from_numpy(y_pred_mc_mean).to(torch.float32)
            y_tensor = torch.from_numpy(label_array).to(torch.float32)

            metrics_variance.append(variance_metrics)
            y_pr_tensor_list.append(y_pr_tensor)
            y_tensor_list.append(y_tensor)

    toc = time.time()
    tictoc = toc - tic

    metrics_fscore = F.f_score(
        torch.stack(y_pr_tensor_list, dim=0),
        torch.stack(y_tensor_list, dim=0),
        eps=1e-9,
        beta=1,
        threshold=0.5,
        ignore_channels=[0, 1],  # None
    )

    metrics_precision = F.precision(
        torch.stack(y_pr_tensor_list, dim=0),
        torch.stack(y_tensor_list, dim=0),
        eps=1e-9,
        threshold=0.5,
        ignore_channels=[0, 1],  # None
    )

    metrics_recall = F.recall(
        torch.stack(y_pr_tensor_list, dim=0),
        torch.stack(y_tensor_list, dim=0),
        eps=1e-9,
        threshold=0.5,
        ignore_channels=[0, 1],  # None
    )

    metrics_accuracy = F.accuracy(
        torch.stack(y_pr_tensor_list, dim=0),
        torch.stack(y_tensor_list, dim=0),
        threshold=0.5,
        ignore_channels=[0, 1],  # None
    )

    metrics_var = np.mean(metrics_variance)

    metrics = [
        metrics_fscore.cpu().numpy(),
        metrics_precision.cpu().numpy(),
        metrics_recall.cpu().numpy(),
        metrics_var,
        tictoc,
    ]

    model_info = (
        str(config.model)
        + "_"
        + LOOCV_city
        + "_"
        + str(config.imagesize)
        + "px"
        + "_"
        + str(shots_per_class)
        + "shots"
    )

    acc_file_path = os.path.join(config.base_dir, "results", "metrics")
    if not os.path.exists(acc_file_path):
        os.makedirs(acc_file_path)

    acc_file_path = os.path.join(
        acc_file_path, (str(model_info) + "_" + str(mci) + "mci.txt")
    )
    with open(acc_file_path, "a") as f:
        f.write(
            "\nF1:{:6.4f} P:{:6.4f} R:{:6.4f} OA:{:6.4f} VAR:{:6.4f} Time:{:6.4f}".format(
                metrics_fscore.cpu().numpy(),
                metrics_precision.cpu().numpy(),
                metrics_recall.cpu().numpy(),
                metrics_accuracy.cpu().numpy(),
                metrics_var,
                tictoc,
            )
        )
    print(
        "F1:{:6.4f} P:{:6.4f} R:{:6.4f} OA:{:6.4f} VAR:{:6.4f} Time:{:6.4f}\n".format(
            metrics_fscore.cpu().numpy(),
            metrics_precision.cpu().numpy(),
            metrics_recall.cpu().numpy(),
            metrics_accuracy.cpu().numpy(),
            metrics_var,
            tictoc,
        )
    )
    return metrics


if __name__ == "__main__":
    Tester()

import os
import numpy as np
import tifffile as tif
import tqdm
import cv2
import torch
import albumentations as albu
from tqdm import tqdm
from shutil import copyfile
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from math import e
import utils.utils_functional as F
import time

import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.append("..")  # Add parent directory to Python path
import config  # noqa: E402
from utils._preprocessing import preprocess_input  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from utils.utils_dataset import Dataset  # noqa: E402
from utils.utils_augmentations import get_validation_augmentation  # noqa: E402
from utils.utils_run_data_sampling import data_sampling_function  # noqa: E402

# import models
from models import STnet


# define helper functions
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),  # , mask=to_tensor),
    ]
    return albu.Compose(_transform)  # , additional_targets={"ndsm": "image"})


def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith("Dropout"):
            each_module.train()


def entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def run_inference(city="1", overlap=1, shots=1, multistage=False, modelpath="None"):
    image = tif.imread("data/mumbai_3m.tif")
    image = image[:, :, 0:3]

    # get sampled dataset
    df = data_sampling_function(LSP=config.LSP, mode="unbalanced")
    df_loocv = df.loc[(df["city"] == city)]
    files_test = df_loocv["path"].tolist()
    class_values_list = df_loocv["class"].tolist()

    # load test dataset
    test_dataset = Dataset(
        files_test,
        class_values_list,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    # create empty prediction image
    prediction_image = np.zeros(
        [image.shape[0], image.shape[1], config.OVERLAP], dtype=np.float32
    )
    entropy_image = np.zeros(
        [image.shape[0], image.shape[1], overlap], dtype=np.float32
    )
    ytrues = []
    ypreds = []
    entropy_list = []
    tictoc_list = []

    # Create model
    model_name = modelpath.split(os.sep)[-1]
    model_name = model_name.split(".")[0]

    device = torch.device("cuda")

    model = STnet.STnet(input_channel=3, num_classes=config.CLASSES)

    loss = CrossEntropyLoss(label_smoothing=0.1)
    loss.__name__ = "loss"
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=0.0),
        ]
    )

    model.to(device)
    checkpoint = torch.load(modelpath)
    print("loading model {}".format(modelpath))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("[INFO] classify {}".format(city))
    model.eval()
    enable_dropout(model)

    print("create classification result with {}-overlaps".format(overlap))
    with torch.no_grad():
        for i in tqdm(range(len(files_test))):
            tic = time.time()
            # get image tiles
            image_tile, label, file_name = test_dataset[i]

            x_tensor = torch.from_numpy(image_tile).to("cuda").unsqueeze(0)
            x_tensor = x_tensor.to(torch.float32)
            y_pred_mc = []

            for mc_i in range(config.MC_ITERATION):
                pr_tensor_mc_i = model(x_tensor)
                pred = pr_tensor_mc_i.cpu().detach().numpy()
                y_pred_mc.append(pred)

            y_pred_arr = np.vstack(y_pred_mc)
            y_pred_mc_mean = np.mean(y_pred_arr, axis=0)

            clipped_arr = np.clip(y_pred_mc_mean, a_min=1e-5, a_max=np.inf)
            norm_arr = [float(i) / sum(clipped_arr) for i in clipped_arr]
            if norm_arr.index(np.max(norm_arr)) == 2:
                norm_arr_max_prob = norm_arr[2]
            else:
                norm_arr_max_prob = 0.0

            entropy_arr = entropy(clipped_arr)
            entropy_list.append(entropy_arr)

            # get ytrue and ypred
            ytrues.append(torch.tensor(label).float())
            ypred = norm_arr.index(np.max(norm_arr))
            y_pred_onehote = np.eye(config.CLASSES)[ypred]
            ypreds.append(torch.tensor(y_pred_onehote).float())

            # get x0 and x1 coordinates
            image_tile_name = file_name.split(os.sep)[-1]
            x0 = image_tile_name.split("x0_")[-1]
            x0 = x0.split("_")[0]
            x0 = int(x0)

            x1 = image_tile_name.split("x1_")[-1]
            x1 = x1.split("_")[0]
            x1 = x1.split(".")[0]
            x1 = int(x1)

            # write predictions to image array
            # TODO! add overlap
            prediction_image[
                x0 : x0 + config.IMAGESIZE, x1 : x1 + config.IMAGESIZE, :
            ] = norm_arr_max_prob  # norm_arr#[2]
            entropy_image[
                x0 : x0 + config.IMAGESIZE, x1 : x1 + config.IMAGESIZE, 0
            ] = entropy_arr

            toc = time.time()
            tictoc = toc - tic
            tictoc_list.append(tictoc)

    # accuracy metrics
    ytrue = [np.argmax(f.cpu().detach().numpy()) for f in ytrues]
    ypred = [np.argmax(f.cpu().detach().numpy()) for f in ypreds]

    ytrue_binary = [1 if f == 2 else 0 for f in ytrue]
    ypred_binary = [1 if f == 2 else 0 for f in ypred]

    metrics_fscore = f1_score(ytrue_binary, ypred_binary, average="binary", pos_label=1)
    metrics_precision = precision_score(
        ytrue_binary, ypred_binary, average="binary", pos_label=1
    )
    metrics_recall = recall_score(
        ytrue_binary, ypred_binary, average="binary", pos_label=1
    )
    metrics_accuracy = accuracy_score(ytrue_binary, ypred_binary)
    metrics_entropy = np.mean(entropy_list)
    metrics_timestep = np.mean(tictoc_list)
    print(metrics_fscore)
    print(metrics_entropy)

    net_name = model_name.split("_")[0]

    acc_file_path = modelpath.split("/models")[0]
    acc_file_path = os.path.join(
        acc_file_path,
        "results",
        ("metrics_" + net_name + "_" + config.OPTIMIZER + "_88px"),
    )

    # get majority value
    prediction_image_majority = np.mean(prediction_image, axis=-1)
    entropy_image_majority = np.mean(entropy_image, axis=-1)

    if config.SAVE_PREDICTIONS:
        # save image with geocoded information
        save_dir_base = os.path.join(config.BASE_DIR, "results", city, model_name)
        if not os.path.exists(save_dir_base):
            os.makedirs(save_dir_base)

        cv2.imwrite(
            os.path.join(
                save_dir_base,
                (model_name + str(int(metrics_fscore * 100)) + "_probability.tif"),
            ),
            prediction_image_majority,
        )
        tfw_source_file = image_path[0].replace(".tif", ".tfw")
        tfw_destination_file = os.path.join(
            save_dir_base,
            (model_name + str(int(metrics_fscore * 100)) + "_probability.tfw"),
        )
        copyfile(tfw_source_file, tfw_destination_file)

        cv2.imwrite(
            os.path.join(
                save_dir_base,
                (model_name + str(int(metrics_fscore * 100)) + "_entropy.tif"),
            ),
            entropy_image_majority,
        )
        tfw_source_file = image_path[0].replace(".tif", ".tfw")
        tfw_destination_file = os.path.join(
            save_dir_base,
            (model_name + str(int(metrics_fscore * 100)) + "_entropy.tfw"),
        )
        copyfile(tfw_source_file, tfw_destination_file)


if __name__ == "__main__":
    # for multistage in [False, True]:
    for city in config.LOOCV_CITY:
        for shots in config.SHOTS_PER_CLASS:
            run_inference(
                city=city,
                overlap=1,
                shots=shots,
                multistage=False,
                modelpath="models/finetune_models/STnet/SS/STnet_mumbai_88px_100shots_42_SS_adam_WCEL_.pth",
            )

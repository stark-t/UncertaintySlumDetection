import os
import cv2
import numpy as np
import pandas as pd
import tifffile as tif
import sys

sys.path.append("..")  # Add parent directory to Python path
import config
from tqdm import tqdm


def resize_image(image, threshold):
    """

    :param image:
    :param threshold:
    :return:
    """
    # get distribution of classes
    unique_imgs = np.unique(image, return_counts=True)
    px_counts = unique_imgs[1]
    classes = unique_imgs[0]

    # check number of classes
    if len(classes) > 1:
        distribution = px_counts[1] / (px_counts[0] + px_counts[1])
        if distribution > threshold:
            resized_image = 1.0
        else:
            resized_image = 0.0
    elif len(classes) == 1 and 1 in image:
        resized_image = 1.0
        distribution = 1.0
    elif len(classes) == 1 and 0 in image:
        resized_image = 0.0
        distribution = 0.0
    else:
        resized_image = 0.0
        distribution = 0.0

    return np.array(resized_image, dtype="float32"), distribution


def get_df(datalist):
    """
    create dataframe for image-, referece-, and urban-data
    :param datalist: list of all files
    :return: dataframe
    """
    image_data_list = [f for f in datalist if "3m.tif" in f and "urban" not in f]
    reference_data_list = [f for f in datalist if "reference" in f]
    urban_data_list = [f for f in datalist if "urban" in f]
    datarecords = []
    for id, image_data in enumerate(image_data_list):
        city_name = image_data.split(os.sep)[-1]
        city_name = city_name.split("_3m")[0]
        urban = [f for f in urban_data_list if city_name.casefold() in f.casefold()][0]
        ref = [f for f in reference_data_list if city_name in f][0]

        record = {
            "image_data_path": image_data,
            "ref_data_path": ref,
            "urban_data_path": urban,
        }
        datarecords.append(record)
    data = pd.DataFrame(datarecords)
    return data


def image_padding(img):
    """

    :param img:
    :return:
    """
    if len(img.shape) == 3:
        img_pad = np.zeros((config.IMAGESIZE, config.IMAGESIZE, 3), dtype=np.uint8)
        img_pad[: img.shape[0], : img.shape[1], : img.shape[2]] = img
    else:
        img_pad = np.zeros((config.IMAGESIZE, config.IMAGESIZE), dtype=np.uint8)
        img_pad[: img.shape[0], : img.shape[1]] = img
    return img_pad


def split_raster(datalist):
    """

    :param datalist:
    :return:
    """

    records = []

    data = get_df(datalist)
    if config.VERBOSE >= 3:
        print(data)
    for data_i, row in data.iterrows():
        imagepath = row["image_data_path"]
        refpath = row["ref_data_path"]
        urbanpath = row["urban_data_path"]
        # city name
        city_name = imagepath.split(os.sep)[-1]
        city_name = city_name.split("_3m")[0]

        dataset_dir = config.TRAINVALTEST_DATASET_DIR
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        print("[INFO] Processing dataset {}".format(city_name))

        # read image file and scale to 8bit
        image = tif.imread(imagepath)
        image = image[:, :, 0:3]
        image = image.astype(np.uint8)

        # get image stats
        for i in range(image.shape[-1]):
            image_mean_ch = np.mean(image[:, :, i])
            image_std_ch = np.std(image[:, :, i])
            #
            # image_mean_ch_.append(image_mean_ch)
            # image_std_ch_.append(image_mean_ch)

            record = {
                "city": city_name,
                "channel": i,
                "mean": image_mean_ch,
                "std": image_std_ch,
            }

            records.append(record)

        reference = tif.imread(refpath)
        reference = reference.astype(np.uint16)

        # read urban file
        urbanurban = tif.imread(urbanpath)

        # get iterator id
        i = -1

        for ax0 in tqdm(
            range(0, int(image.shape[0]), int(config.IMAGESIZE / config.OVERLAP))
        ):
            for ax1 in range(
                0, int(image.shape[1]), int(config.IMAGESIZE / config.OVERLAP)
            ):
                # new iterator id
                i += 1
                # get image content
                img = image[
                    ax0 : ax0 + config.IMAGESIZE, ax1 : ax1 + config.IMAGESIZE, 0:3
                ]
                img_urban = urbanurban[
                    ax0 : ax0 + config.IMAGESIZE, ax1 : ax1 + config.IMAGESIZE
                ]
                img_slum = reference[
                    ax0 : ax0 + config.IMAGESIZE, ax1 : ax1 + config.IMAGESIZE
                ]

                # handle uneven images
                if img.shape[:-1] != (config.IMAGESIZE, config.IMAGESIZE):
                    if config.UNEVEN_IMAGE_TILES == "padding":
                        # zero padding uneven image-tile
                        img = image_padding(img)
                        img_urban = image_padding(img_urban)
                        img_slum = image_padding(img_slum)

                    elif config.uneven_image_tiles == "resize":
                        # resize uneven image-tile
                        img = cv2.resize(
                            img,
                            (config.IMAGESIZE, config.IMAGESIZE),
                            interpolation=cv2.INTER_AREA,
                        )
                        img_urban = cv2.resize(
                            img_urban,
                            (config.IMAGESIZE, config.IMAGESIZE),
                            interpolation=cv2.INTER_AREA,
                        )
                        img_slum = cv2.resize(
                            img_slum,
                            (config.IMAGESIZE, config.IMAGESIZE),
                            interpolation=cv2.INTER_AREA,
                        )

                    elif config.UNEVEN_IMAGE_TILES == "skip":
                        # skip uneven image-tile
                        continue
                    else:
                        # Error
                        print(
                            'Error in handling uneven images: Select one of ["padding", "resize", "skip"]'
                        )
                        print(1 / 0)

                # get distribution of classes
                # fuse urban and slum array
                img_slum[img_slum > 0] = 2
                ref_array = img_urban + img_slum
                ref_array[ref_array > 2] = 2
                for c in range(config.CLASSES):
                    ref_array[0, c] = c
                _, label_dist = np.unique(ref_array, return_counts=True)
                label_distribution = label_dist / (config.IMAGESIZE**2)

                # write image tile to dataset
                label_dist_str = [
                    (str(int(f * 100)) + "_") for f in list(label_distribution)
                ]
                label_dist_str = "".join(label_dist_str)
                image_name = (
                    city_name
                    + "_ID_"
                    + str(int(i))
                    + "_dst_"
                    + label_dist_str
                    + "x0_"
                    + str(ax0)
                    + "_x1_"
                    + str(ax1)
                    + ".tif"
                )

                image_path = os.path.join(dataset_dir, image_name)
                cv2.imwrite(image_path, img)

    image_stats_df = pd.DataFrame(records)
    image_stats_df.to_csv(config.TRAINVALTEST_DATASET_DIR + "_image_stats.csv")

    print("finished")

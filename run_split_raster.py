import glob
import os
import config as config
from utils.utils_split_raster import split_raster


def split_raster_function(data_path="PATH"):
    """

    :return:
    """

    datalist = glob.glob(data_path + os.sep + "*.tif")

    split_raster(datalist=datalist)


if __name__ == "__main__":
    split_raster_function(
        data_path="/mnt/ushelf_star_th/projects/2023_TDGUP_Dissertation/2022_P2/UncertaintySlumDetection/data"
    )

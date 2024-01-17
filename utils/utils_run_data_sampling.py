# import packages
import glob
import os
import pandas as pd
import numpy as np

# import code
import config as config


def create_distribution_dataframe(image_tile_list):
    """
    :param image_tile_list:
    :return:
    """

    file_names_list = [f.split(os.sep)[-1] for f in image_tile_list]
    file_name_parts = [f.split("_") for f in file_names_list]
    records = []

    for i, _ in enumerate(file_name_parts):
        record = {
            "ID": str(file_name_parts[i][2]),
            "city": str(file_name_parts[i][0]),
            "path": str(image_tile_list[i]),
            "dist_bg": str(file_name_parts[i][4]),
            "dist_urban": str(file_name_parts[i][5]),
            "dist_slum": str(file_name_parts[i][6]),
        }
        records.append(record)

    # create dataframe
    distribution_df = pd.DataFrame(records)
    distribution_df["ID"] = distribution_df["ID"].astype("int")
    distribution_df["dist_bg"] = distribution_df["dist_bg"].astype("float32")
    distribution_df["dist_urban"] = distribution_df["dist_urban"].astype("float32")
    distribution_df["dist_slum"] = distribution_df["dist_slum"].astype("float32")

    if __name__ == "main" or config.VERBOSE >= 2:
        print("original dataset")
        print(distribution_df.describe())
        print(distribution_df.head())

    return distribution_df


def data_sampling_function(LSP=50, mode="unbalanced", seed=0):
    """

    :param LSP:
    :param sampling: select from random, LOOCV
    :param mode: select from unbalanced/original, balanced_big, balanced_few
    :return:
    """

    image_tile_list = glob.glob(config.TRAINVALTEST_DATASET_DIR + "/*.tif")
    data_sampling_df = create_distribution_dataframe(image_tile_list)
    data_sampling_df.drop("ID", axis=1, inplace=True)
    data_sampling_df.reset_index(inplace=True)
    data_sampling_df = data_sampling_df.rename(columns={"index": "ID"})

    # get class dataframe depending on their LSP
    df_slums = data_sampling_df.loc[(data_sampling_df["dist_slum"] >= LSP)]
    df_slums.insert(0, "class", 2)

    df_urban = data_sampling_df.loc[
        (data_sampling_df["dist_urban"] >= LSP) & (data_sampling_df["dist_slum"] < 1)
    ]
    df_urban.insert(0, "class", 1)

    df_other = data_sampling_df.loc[
        (data_sampling_df["dist_bg"] >= LSP) & (data_sampling_df["dist_slum"] < 1)
    ]
    df_other.insert(0, "class", 0)

    # combine all dfs together
    df_classes = pd.concat([df_slums, df_urban, df_other])
    col_names = df_classes.columns
    df_classes = df_classes.drop_duplicates(col_names[1:])

    # get class dataframe depending on class
    df_slums = df_classes.loc[(df_classes["class"] == 2)]
    df_urban = df_classes.loc[(df_classes["class"] == 1)]
    df_other = df_classes.loc[(df_classes["class"] == 0)]

    # # get image count per city per class

    # select dataframe parameters
    # balanced/unbalanced and label pixel percentage
    if mode == "unbalanced":
        # unbalanced
        # df_sampled = df_classes
        df_sampled = df_classes.sample(frac=1, random_state=seed)
        if __name__ == "main" or config.VERBOSE >= 2:
            print('Number of slum tiles per city in original "unbalanced" dataset mode')
            print_df = df_classes.groupby(["city", "class"])["ID"].count()
            print(print_df)
    # balanced_city, balanced_class, balanced_few]')
    elif mode == "balanced_city" or mode == "balanced_class" or mode == "balanced_few":
        # balanced per class
        city_count_slums = df_slums.groupby(["city"])["ID"].count()
        city_count_urban = df_urban.groupby(["city"])["ID"].count()
        city_count_other = df_other.groupby(["city"])["ID"].count()

        if __name__ == "main" or config.VERBOSE >= 2:
            print('Number of slum tiles per city in "balanced_big" dataset mode')
            print(city_count_slums)
            print('Number of urban tiles per city in "balanced_big" dataset mode')
            print(city_count_urban)
            print('Number of background tiles per city in "balanced_big" dataset mode')
            print(city_count_other)

        # get number of images per class
        if mode == "balanced_city":
            # get min count for slums and median for urban
            min_slum_count = 1000  # int(city_count_slums.median())
            min_urban_count = 1000  # int(city_count_urban.median())
            min_other_count = 200  # int(city_count_other.median())
        elif mode == "balanced_class":
            # get shots per class
            min_slum_count = 200  # int(city_count_slums.median())
            min_urban_count = 200  # int(city_count_slums.median())
            min_other_count = 200  # int(city_count_slums.median())

        elif mode == "balanced_few":
            # get shots per class
            min_slum_count = int(config.SHOTS_PER_CLASS)
            min_urban_count = int(config.SHOTS_PER_CLASS)
            min_other_count = int(config.SHOTS_PER_CLASS)

        group_slum = df_slums.groupby(["city"])
        slum_ix = np.hstack(
            [
                np.random.choice(v, min_slum_count, replace=True)
                for v in group_slum.groups.values()
            ]
        )
        df_slums_sampled = df_slums.loc[slum_ix]

        group_urban = df_urban.groupby(["city"])
        urban_ix = np.hstack(
            [
                np.random.choice(v, min_urban_count, replace=True)
                for v in group_urban.groups.values()
            ]
        )
        df_urban_sampled = df_urban.loc[urban_ix]

        group_other = df_other.groupby(["city"])
        other_ix = np.hstack(
            [
                np.random.choice(v, min_other_count, replace=True)
                for v in group_other.groups.values()
            ]
        )
        df_other_sampled = df_other.loc[other_ix]

        df_sampled = pd.concat([df_slums_sampled, df_urban_sampled, df_other_sampled])
        df_sampled = df_sampled.sample(frac=1, random_state=seed)

        if __name__ == "main" or config.VERBOSE >= 1:
            print('Number of image tiles per city in "balanced_big" dataset mode')
            print_df = df_sampled.groupby(["city", "class"])["ID"].count()
            print(print_df)

    else:
        print(
            "Select datasampling, choose from [unbalanced, balanced_city, balanced_class, balanced_few]"
        )
        print(1 / 0)

    return df_sampled


def dataset_info(data_sampling_df, LSP=25, mode="unbalanced"):
    """

    :return:
    """
    df = data_sampling_df.groupby(["city", "class"])["ID"].count()

    path = os.path.join(
        config.BASE_DIR, "datasets", ("LSP" + str(LSP) + "_" + mode + ".csv")
    )
    df.to_csv(path)

    return 1


if __name__ == "__main__":
    LSP = 10  # 50
    mode = "unbalanced"

    data_sampling_df = data_sampling_function(LSP=LSP, mode=mode)
    dataset_info(data_sampling_df, LSP=LSP, mode=mode)

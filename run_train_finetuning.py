# import packages
import os

# import code
import config as config
from utils.utils_Trainer import Trainer


def run_train_finetuning():
    """
    Function to run the finetuning phase of the training process.

    :return: None
    """

    for LOOCV_city in config.LOOCV_CITY:
        for shots in config.SHOTS_PER_CLASS:
            if not shots == 0:
                stage = "SS"
                modelpath_base = os.path.join(
                    config.BASE_MODELPATH, config.MODEL, stage
                )

                if not os.path.exists(modelpath_base):
                    os.makedirs(modelpath_base)

                if config.WEIGHTED_LOSS:
                    WCEL = "_WCEL_"
                else:
                    WCEL = "_"

                modelpath_base = os.path.join(
                    modelpath_base,
                    (
                        config.MODEL
                        + "_"
                        + LOOCV_city
                        + WCEL
                        + config.OPTIMIZER
                        + "_"
                        + str(config.IMAGESIZE)
                        + "px_"
                        + stage
                    ),
                )
                modelpath_save = os.path.join(
                    config.FINE_MODELPATH, config.MODEL, stage
                )

                if not os.path.exists(modelpath_save):
                    os.makedirs(modelpath_save)

                modelpath_save = os.path.join(
                    modelpath_save,
                    (
                        config.MODEL
                        + "_"
                        + LOOCV_city
                        + "_"
                        + str(config.IMAGESIZE)
                        + "px_"
                        + str(shots)
                        + "shots_"
                        + str(config.SEED)
                        + "_"
                        + stage
                        + "_"
                        + config.OPTIMIZER
                        + WCEL
                        + ".pth"
                    ),
                )
                modelpath_base = modelpath_base + ".pth"
                # for seed in range(300, 901, 300):
                print("Start fine tuning")
                Trainer(
                    epochs=50,
                    lr=config.INITIAL_LEARNING_RATE,
                    augmentations="weak",
                    LSP=config.LSP,
                    mode="LOOCV",
                    traintestval_split="2fold",
                    LOOCV_city=LOOCV_city,
                    shots_per_class=shots,
                    seed=config.SEED,
                    load_weights=modelpath_base,
                    modelpath_save=modelpath_save,
                    warmup=True,
                )

    print("Done")


if __name__ == "__main__":
    run_train_finetuning()

# import packages
import os

# import code
import config as config
from utils.utils_Trainer import Trainer


def run_train_pretraining():
    """
    Function to run the pretraining phase of the training process.

    :return: None
    """

    # train pretrain
    epochs_stage_1 = 50

    if config.LOOCV_CITY is None:
        LOOCV = ["random4fold"]
    else:
        LOOCV = config.LOOCV_CITY

    for LOOCV_city in LOOCV:
        print("Start training")
        stage = "SS"

        modelpath_save = os.path.join(config.BASE_MODELPATH, config.MODEL, stage)

        if not os.path.exists(modelpath_save):
            os.makedirs(modelpath_save)

        if config.WEIGHTED_LOSS:
            WCEL = "WCEL_"
        else:
            WCEL = ""

        modelpath_save = os.path.join(
            modelpath_save,
            (
                config.MODEL
                + "_"
                + LOOCV_city
                + "_"
                + WCEL
                + config.OPTIMIZER
                + "_"
                + str(config.IMAGESIZE)
                + "px_"
                + stage
                + ".pth"
            ),
        )
        Trainer(
            epochs=epochs_stage_1,
            lr=config.INITIAL_LEARNING_RATE,
            augmentations="weak",
            LSP=config.LSP,
            mode="unbalanced",
            traintestval_split="LOOCV_pretrain",
            LOOCV_city=LOOCV_city,
            shots_per_class=None,
            load_weights=None,
            modelpath_save=modelpath_save,
            warmup=True,
        )

    print("Done")


if __name__ == "__main__":
    run_train_pretraining()

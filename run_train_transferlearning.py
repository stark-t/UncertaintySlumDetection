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

    # loop over all cities
    for LOOCV_city in config.LOOCV_CITY:
        # loop over all shots
        for shots in config.SHOTS_PER_CLASS:
            # if not inference starte transfer learning
            if not shots == 0:
                # Set state to Single Stage training
                stage = "SS"
                # Set modelpath base directory
                modelpath_base = os.path.join(
                    config.BASE_MODELPATH, config.MODEL, stage
                )
                # Create modelpath if it does not exist
                if not os.path.exists(modelpath_base):
                    os.makedirs(modelpath_base)

                if config.WEIGHTED_LOSS:
                    WCEL = "_WCEL_"
                else:
                    WCEL = "_"
                # Set modelpath for pretrained model
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
                # Set modelbasepath for finetuned model
                modelpath_save = os.path.join(
                    config.FINE_MODELPATH, config.MODEL, stage
                )
                # Create modelpath if it does not exist
                if not os.path.exists(modelpath_save):
                    os.makedirs(modelpath_save)
                # Set modelpath for finetuned model
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
                # Start transfer learning
                print("Start transfer-learning")
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

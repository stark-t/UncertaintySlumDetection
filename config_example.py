## CONFIGURATION FILE
# VARIABLES AND HYPERPARAMETERS
BATCH_SIZE = 16
CLASSES = 3
EARLY_STOPPING = 50
IMAGESIZE = 88
INITIAL_LEARNING_RATE = 1e-4
LIMIT_AOI = True
LOOCV_CITY = [
    "capetown",
    "caracas",
    "lagos",
    "medellin",
    "mumbai",
    "nairobi",
    "rio",
    "saopaulo",
]
LSP = 25
MC_ITERATION = 1
MODEL = "STnet"
OPTIMIZER = "adam"
OVERLAP = 1
SAVE_PREDICTIONS = True
SEED = 42
SHOTS_PER_CLASS = [100]
UNEVEN_IMAGE_TILES = "padding"
VAL_SHOTS_PER_CLASS = 100
VERBOSE = 2
WEIGHTED_LOSS = True
WARM_UP_EPOCHS = 3
WARM_UP_LEARNING_RATE = 1e-8

# DIRECTORIES
BASE_DIR = "/path/.../"
BASE_MODELPATH = BASE_DIR + "/path/.../models/base_models"
FINE_MODELPATH = BASE_DIR + "/path/.../models/finetune_models"
IMAGE_STATS_PATH = (
    "/path/.../data/datasets/.../" + str(IMAGESIZE) + "px_image_stats.csv"
)
TRAINVALTEST_DATASET_DIR = "/path/.../data/datasets/.../" + str(IMAGESIZE) + "px"

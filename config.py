## CONFIGURATION FILE
# VARIABLES AND HYPERPARAMETERS
BATCH_SIZE = 16
CLASSES = 3
EARLY_STOPPING = 50
IMAGESIZE = 88
INITIAL_LEARNING_RATE = 1e-4
LIMIT_AOI = True
LIMIT_AOI = True
# LOOCV_CITY = [
#     "capetown",
#     "caracas",
#     "lagos",
#     "medellin",
#     "mumbai",
#     "nairobi",
#     "rio",
#     "saopaulo",
# ]
LOOCV_CITY = [
    "mumbai",
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
VERBOSE = 3
WEIGHTED_LOSS = True
WARM_UP_EPOCHS = 3
WARM_UP_LEARNING_RATE = 1e-8

# DIRECTORIES
BASE_DIR = "/mnt/ushelf_star_th/projects/2023_TDGUP_Dissertation/2022_P2/UncertaintySlumDetection/"
BASE_MODELPATH = BASE_DIR + "models/base_models"
FINE_MODELPATH = BASE_DIR + "models/finetune_models"
IMAGE_STATS_PATH = (
    "/mnt/ushelf_star_th/projects/2023_TDGUP_Dissertation/2022_P2/UncertaintySlumDetection/data/datasets/"
    + str(IMAGESIZE)
    + "px_image_stats.csv"
)
TRAINVALTEST_DATASET_DIR = (
    "/mnt/ushelf_star_th/projects/2023_TDGUP_Dissertation/2022_P2/UncertaintySlumDetection/data/datasets/"
    + str(IMAGESIZE)
    + "px"
)

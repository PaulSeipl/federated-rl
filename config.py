SAVE_MODELS = True
SAVE_PLOTS = True
SAVE_VIDEOS = True

TRAINING_EPISODES = 10
WORKING_INTERVALS = 3
TEST_EPISODES = 10
MAX_STEPS = 100
ROOMS_DIR = "9_9_4"
CUSTOM_PATH = f"{ROOMS_DIR}/tr{TRAINING_EPISODES}_x{WORKING_INTERVALS}_mSt{MAX_STEPS}"
PLOT_PATH = f"./plots/{CUSTOM_PATH}"
MOVIE_PATH = f"./movies/{CUSTOM_PATH}"

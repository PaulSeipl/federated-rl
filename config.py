SAVE_MODELS = True
SAVE_PLOTS = True
SAVE_VIDEOS = True

TRAINING_EPISODES = 2500
WORKING_INTERVALS = 4
TEST_EPISODES = 100
MAX_STEPS = 100
ROOMS_DIR = "9_9_4_test"
CUSTOM_PATH = (
    f"{ROOMS_DIR}/tr{TRAINING_EPISODES}_x{WORKING_INTERVALS}_mSt{MAX_STEPS}_wa"
)
PLOT_PATH = f"./plots/{CUSTOM_PATH}"
MOVIE_PATH = f"./movies/{CUSTOM_PATH}"

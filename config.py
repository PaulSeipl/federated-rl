SAVE_MODELS = True
SAVE_PLOTS = True
SAVE_VIDEOS = False

TRAINING_EPISODES = 200
WORKING_INTERVALS = 50
TEST_EPISODES = 50
MAX_STEPS = 100
MAX_DONE_COUNTER = 5
AGGREATION_TYPES = {1: "average", 2: "bad_worker", 3: "good_worker"}
AGGREATION_TYPE = AGGREATION_TYPES[1]
ROOMS_DIR = "9_9_4"
TEST_STOCHASTIC = True
TEST_DIRS = {1: "easy", 2: "normal", 3: "difficult"}
TEST_DIR = TEST_DIRS[2]
AGENT_PER_PERMUTATION = False
ROTATE_MAP = not AGENT_PER_PERMUTATION
CUSTOM_PATH = f"{ROOMS_DIR}/tr{TRAINING_EPISODES}_x{WORKING_INTERVALS}_mSt{MAX_STEPS}_{AGGREATION_TYPE}_test{'Stochastic' if TEST_STOCHASTIC else 'Determenistic'}_agentPer{'Permutation' if AGENT_PER_PERMUTATION else 'Room'}"
PLOT_PATH = f"./plots/{CUSTOM_PATH}"
MOVIE_PATH = f"./movies/{CUSTOM_PATH}"

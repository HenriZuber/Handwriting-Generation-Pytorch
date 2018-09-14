RANDOM_SEED = 100


STROKES_PATH = "data/strokes.npy"
TEXTS_PATH = "data/sentences.txt"
IMAGES_TRAIN_UNCON_PATH = "images/train_uncon/"
IMAGES_TRAIN_CON_PATH = "images/train_con/"
IMAGES_INF_PATH = "images/inf/"
UNCON_GEN_MODEL_PATH = "models/uncon_gen.pt"
CON_GEN_MODEL_PATH = "models/con_gen.pt"

BATCH_SIZE = 32
N_LAYER = 2
HIDDEN_LAYER_SIZE = 256
GAUSSIAN_MIX_NUM = 20
WINDOWS_NUM = 10

EPOCH = 200
BATCH_PER_EPOCH = 100
LR = 0.001
GRAD_CLIP = 5

STROKE_LENGTH_COEFF = 2
NEW_STROKE_LENGTH = 700
SUBSTROKE_LENGTH = 150

PROBABILITY_BIAS = 1

VERBOSE_EVERY = 1

TEXT_TO_HANDWRITE = 'nice one'

CONDITIONAL = True
INFERENCE = True

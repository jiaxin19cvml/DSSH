import os
import os.path as op
"""
input and output path
"""
## general
ROOT_DIR = "/media/jiaxinchen/Big Data/Project/3D/3DRetrieval"
DATASET = "SHREC13"
NUMB_CLASS = 90
RESUME = False
EMD_DIM = 16
## settings for shape
PROJMETHOD = "C4RAND"
NAME_SUBNET_SHAPE = "hashNet"
FLAG_RANDOM_SAMPLING = False
NUMB_CHANNELS = 4
## settings for sketch
NAME_SUBNET_SKETCH = "hashNet"
## general settings
EXP_ROOT = op.join(ROOT_DIR, "trainedModels", DATASET, "SVP_FEAT_noview", str(EMD_DIM)+"bits_resnet50")
LOSS_FUNCTION = "SVP" #{"crossentropy", "triplet", "tripCrossEntropy"}
"""
parameters for training
"""
FLAG_SUBSTRACT_MEAN=True
FLAG_SHUFFLE=True

LEARNING_RATE = 1e-4
MAX_NUM_ITER = 5000
CHECKPOINT_FREQUENCY = 200
VALIDATION_FREQUENCY = 200
DECAY_STAT_ITERATION = 1000

BATCH_P = 32
BATCH_K = 4
BATCH_SIZE = BATCH_P*BATCH_K

"""
Data augmentations for sketch
"""
FLAG_AUG_CROP = False
FLAG_AUG_FLIP = False
FLAG_AUG_CROP_TEST = False
CROP_AUGMENT_TEST = "five"
LOADING_THREADS = 8

"""
settings for test
"""
BATCH_SIZE_TEST_SKETCH = 256
BATCH_SIZE_TEST_SHAPE = 128
CHECKPOINT = None

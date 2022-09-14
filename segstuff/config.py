import torch
import os


WD = '/home/bmk/personal_code/kaggle/card_MRI'
DATA_DIR = os.path.join(WD,'data')
TRAIN_IMAGES = os.path.join(DATA_DIR, 'imagesTr')
TRAIN_LABELS = os.path.join(DATA_DIR, 'labelsTr')
TEST_IMAGES = os.path.join(DATA_DIR, 'imagesTs')
TENSOR_DIR = os.path.join(DATA_DIR, 'tensorsTr')
TEST_SPLIT = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

NUM_CHANNELS = 1
NUM_CLASSES = 1


INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 64

INPUT_IMAGE_WIDTH = 320
INPUT_IMAGE_HEIGHT = 320

THRESHOLD = 0.5

BASE_OUTPUT = "output"

MODEL_PATH = os.path.join(BASE_OUTPUT, "mri_seg_unet.pt")
TEST_PATH = os.path.join(BASE_OUTPUT, "test_paths.txt")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot.png")

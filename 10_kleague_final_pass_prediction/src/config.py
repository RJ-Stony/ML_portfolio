import torch

TRAIN_PATH = "../Data/train.csv"
TEST_PATH = "../Data/test.csv"
SUBMISSION_PATH = "../Data/sample_submission.csv"
SAVE_MODEL_PATH = "../Data/model.pt"

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
HIDDEN_DIM = 96

# uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")
import argparse
import torch

parser = argparse.ArgumentParser(description="Arguments for running the scripts")

# * Data.
# Config.
parser.add_argument(
    "--data",
    type=str,
    default="one",
    choices=["one", "all"],
    help="Choose to train on one subject or all",
)
parser.add_argument("--joints", type=int, default=32)
parser.add_argument("--input_n", type=int, default=25)
parser.add_argument("--output_n", type=int, default=25)
parser.add_argument("--miss_rate", type=int, default=20)
# Files.
parser.add_argument("--output_dir", type=str, default="out/")
parser.add_argument("--model_s", type=str, default="out/short/")
parser.add_argument("--model_l", type=str, default="out/long/")
parser.add_argument("--data_dir", type=str, default="/media/hdd/datasets_common/")
# TODO Logging.
parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none")
# parser.add_argument("--log_interval", type=int, default=10)
# parser.add_argument(
#     "--resume", action="store_true", help="Resume training from checkpoint."
# )

# * Config.
parser.add_argument(
    "--mode",
    type=str,
    default="train",
    choices=["train", "test"],
    help="Choose to train or test from the model.",
)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--weight-decay", type=float, default=1e-6)
parser.add_argument("--skip_rate", type=float, default=1)
# Model
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--conditional", action="store_true")
parser.add_argument("--time-embedding", type=int, default=128)
parser.add_argument("--feature_embedding", type=int, default=16)
# Diffusion
parser.add_argument("--diffusion-layers", type=int, default=12)
parser.add_argument("--diffusion-channels", type=int, default=64)
parser.add_argument("--diffusion-heads", type=int, default=8)
parser.add_argument(
    "--diffusion-embedding",
    type=int,
    default=128,
    help="Diffusion embedding's dimension.",
)
parser.add_argument("--diffusion-beta-start", type=float, default=1e-4)
parser.add_argument("--diffusion-beta-end", type=float, default=0.5)
parser.add_argument("--diffusion-timesteps", type=int, default=50)
parser.add_argument(
    "--variance-scheduler",
    type=str,
    choices=["linear", "cosine", "quadratic"],
    default="cosine",
)


args = parser.parse_args()

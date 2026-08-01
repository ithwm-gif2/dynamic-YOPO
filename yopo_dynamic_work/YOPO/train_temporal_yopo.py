import argparse
import os
import random

import numpy as np
import torch

from policy.temporal_yopo_trainer_v2 import TemporalYopoTrainer


def configure_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train continuous-frame YOPO with time-indexed occupancy loss."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument(
        "--max-samples-per-episode",
        type=int,
        default=0,
        help="Limit each episode for smoke tests; zero uses every valid frame.",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=0,
        help="Limit weighted samples per epoch; zero uses full training-set size.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    configure_random_seed(args.seed)
    trainer = TemporalYopoTrainer(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        run_name=args.run_name,
        resume=args.resume,
        max_samples_per_episode=args.max_samples_per_episode,
        samples_per_epoch=args.samples_per_epoch,
        early_stopping_patience=args.early_stopping_patience,
    )
    trainer.train()

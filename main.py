"""
main.py
Aspect-Based Sentiment Analysis (ABSA) Project Launcher
This module provides a command-line interface for launching supervised and unsupervised ABSA tasks
in English and Turkish. It supports training and prediction modes for both approaches.
Usage:
    python main.py --mode {supervised,unsupervised} [options]
Arguments:
    --mode           : Selects the ABSA mode ('supervised' or 'unsupervised').
    --lang           : Language code ('en' for English, 'tr' for Turkish). Default is 'en'.
    --dataset        : Path to the training dataset (XML for supervised, CSV for unsupervised).
    --output_dir     : Directory to save outputs. Default is 'outputs/supervised/absa'.
    --predict        : If set, runs prediction instead of training.
    --text           : Text(s) to predict sentiment for (used with --predict).
    --model_dir      : Path to the trained model directory (for prediction).
    --epochs         : Number of training epochs. Default is 1.
    --batch_size     : Training batch size. Default is 4.
    --lr             : Learning rate. Default is 5e-5.
    --weight_decay   : Weight decay for optimizer. Default is 0.01.
    --val_ratio      : Validation split ratio. Default is 0.2.
    --max_length     : Maximum sequence length for tokenization. Default is 128.
    --seed           : Random seed for reproducibility. Default is 42.
    --aspect_count   : Number of aspects to extract (unsupervised mode only). Default is 5.
Functions:
    main()           : Parses arguments and launches the appropriate ABSA workflow.
Imports:
    supervised_absa_multilingual (src.sup)   : Supervised ABSA implementation.
    unsupervised_absa_multilingual (src.unsup): Unsupervised ABSA implementation.

Author: Cem Rifki Aydin
Date: 03.09.2025
    
"""
import argparse

from src import supervised_absa_multilingual as sup
from src import unsupervised_absa_multilingual as unsup

def main():
    parser = argparse.ArgumentParser(description="ABSA Project Launcher")
    parser.add_argument("--mode", type=str, choices=["supervised", "unsupervised"], required=True)
    parser.add_argument("--lang", type=str, choices=["en", "tr"], default="en")
    parser.add_argument("--dataset", type=str, help="Path to training XML for supervised or csv for unsupervised")
    parser.add_argument("--output_dir", type=str, default="outputs/supervised/absa")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--text", type=str, nargs="*")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aspect_count", type=int, default=5, help="Count of aspects to be extracted for unsupervised")
    

    args = parser.parse_args()

    if args.mode == "supervised":

        if args.predict:
            sup.predict(args)
        else:
            sup.train(args)
    elif args.mode == "unsupervised":

        if args.predict:
            unsup.predict(args)
        else:
            unsup.train(args)

if __name__ == "__main__":
    main()


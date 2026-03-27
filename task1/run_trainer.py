"""
Simple script to generate data and run the trainer.
"""

import os
from generate_data import generate_synthetic_data
from trainer import Trainer


def main():
    # Generate synthetic data if it doesn't exist
    if not os.path.exists('data/protein_fitness.csv'):
        print("Generating synthetic data...")
        generate_synthetic_data()
        print()

    # Train model
    print("Starting training...")
    trainer = Trainer()
    trainer.load_data()
    trainer.train(epochs=20)  # Fewer epochs for quick demo
    trainer.evaluate()


if __name__ == '__main__':
    main()

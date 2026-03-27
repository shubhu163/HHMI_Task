"""
Generate synthetic protein fitness data for testing the trainer.
"""

import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_samples=1000, seq_length=50, output_dir='data'):
    """Generate synthetic protein sequences with fitness scores"""

    # Amino acids (20 standard)
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')

    sequences = []
    fitness_scores = []

    np.random.seed(42)  # For reproducibility of data generation

    for i in range(n_samples):
        # Generate random sequence
        seq = ''.join(np.random.choice(amino_acids, size=seq_length))

        # Simple synthetic fitness function
        # Fitness depends on composition and some position-specific effects
        fitness = 0.0

        # Compositional bias (more hydrophobic = higher fitness)
        hydrophobic = set('AILMFWV')
        hydrophobic_count = sum(1 for aa in seq if aa in hydrophobic)
        fitness += hydrophobic_count * 0.1

        # Position-specific effects (having certain AAs at certain positions)
        if seq[0] == 'M':  # Methionine at start
            fitness += 2.0
        if seq[10] == 'C':  # Cysteine at position 10
            fitness += 1.5
        if seq[25] in 'DE':  # Charged residue at position 25
            fitness += 1.0

        # Epistatic interaction (positions 15 and 30)
        if seq[15] == 'K' and seq[30] == 'E':
            fitness += 3.0

        # Add some noise
        fitness += np.random.normal(0, 1.0)

        sequences.append(seq)
        fitness_scores.append(fitness)

    # Create DataFrame
    df = pd.DataFrame({
        'sequence': sequences,
        'fitness': fitness_scores
    })

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'protein_fitness.csv')
    df.to_csv(output_path, index=False)

    print(f"Generated {n_samples} synthetic protein sequences")
    print(f"Saved to {output_path}")
    print(f"\nFitness statistics:")
    print(f"  Mean: {df['fitness'].mean():.2f}")
    print(f"  Std: {df['fitness'].std():.2f}")
    print(f"  Min: {df['fitness'].min():.2f}")
    print(f"  Max: {df['fitness'].max():.2f}")

    return df


if __name__ == '__main__':
    generate_synthetic_data()

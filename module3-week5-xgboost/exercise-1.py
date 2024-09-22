import numpy as np
import pandas as pd

# Initialize the dataset
data = pd.DataFrame({
    'X': [23, 24, 26, 27],
    'Y': [50, 70, 80, 85]
})

# Parameters
lr = 0.3
depth = 1
lambda_ = 0

# Step 1: Initialize f0 with the mean of Y
f0 = data['Y'].mean()
print(f"Initial prediction (f0): {f0}")

# Step 2: Calculate residuals (Y - f0)
data['Residuals'] = data['Y'] - f0
print("\nStep 2: Residuals")
print(data[['X', 'Residuals']])

# Similarity Score Calculation


def similarity_score(residuals, lambda_):
    sum_of_residuals = residuals.sum()
    number_of_residuals = len(residuals)
    score = (sum_of_residuals**2) / (number_of_residuals + lambda_)
    return score


# Root Similarity Score
root_similarity_score = similarity_score(data['Residuals'], lambda_)
print(f"\nRoot Similarity Score: {root_similarity_score}")

# Step 3: Try splitting at different thresholds
split_points = [23.5, 25, 26.5]


def calculate_gain(data, split_value, lambda_):
    # Split data into left and right branches
    left = data[data['X'] < split_value]
    right = data[data['X'] >= split_value]

    # Calculate similarity scores for left and right
    left_similarity_score = similarity_score(left['Residuals'], lambda_)
    right_similarity_score = similarity_score(right['Residuals'], lambda_)

    # Calculate Gain
    gain = left_similarity_score + right_similarity_score - root_similarity_score
    return gain, left_similarity_score, right_similarity_score


# Calculate gain for each split
gains = []
for split in split_points:
    gain, left_score, right_score = calculate_gain(data, split, lambda_)
    gains.append((split, gain, left_score, right_score))
    print(f"Split at {split}: Gain = {gain}")

# Step 4: Choose the split with the highest gain
best_split = max(gains, key=lambda x: x[1])
best_split_value, best_gain, left_score, right_score = best_split
print(f"\nBest split: X < {best_split_value}, Gain = {best_gain}")

# Step 5: Calculate the Output for left and right nodes


def calculate_output(residuals):
    return residuals.sum() / len(residuals)


left_data = data[data['X'] < best_split_value]
right_data = data[data['X'] >= best_split_value]

left_output = calculate_output(left_data['Residuals'])
right_output = calculate_output(right_data['Residuals'])

print(f"Left Output: {left_output}")
print(f"Right Output: {right_output}")

# Step 6: Make final predictions for X=25 (which falls into the right node based on the split)
new_prediction = f0 + lr * right_output
print(f"\nFinal prediction for X=25: {new_prediction}")

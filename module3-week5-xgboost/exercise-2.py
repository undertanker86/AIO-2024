import numpy as np
import pandas as pd
import math as Math
# Initialize the dataset
data = pd.DataFrame({
    'X': [23, 24, 26, 27],
    'Y': [False, False, True, True]
})

# Convert Y to binary (0 and 1)
data['Y'] = data['Y'].astype(int)

# Parameters
lr = 0.3  # learning rate
depth = 1
lambda_ = 0  # regularization parameter

# Step 1: Initialize with probability 0.5 (f0)
data['PreviousProbability'] = 0.5
f0 = 0.5
print(f"Initial probability (f0): {f0}")

# Step 2: Calculate residuals
# Residuals = Y - PreviousProbability
data['Residuals'] = data['Y'] - data['PreviousProbability']
print("\nStep 2: Residuals")
print(data[['X', 'Residuals']])

# Step 3: Similarity Score Calculation


def similarity_score(residuals, probabilities, lambda_):
    sum_of_residuals = residuals.sum()
    weighted_residuals = np.sum(probabilities * (1 - probabilities))
    score = (((sum_of_residuals**2)) / (weighted_residuals + lambda_))

    return score


# Root Similarity Score
root_similarity_score = similarity_score(
    data['Residuals'], data['PreviousProbability'], lambda_)
print(f"\nRoot Similarity Score: {root_similarity_score}")

# Step 4: Try splitting at different thresholds
split_points = [23.5, 25, 26.5]


def calculate_gain(data, split_value, lambda_):
    # Split data into left and right branches
    left = data[data['X'] < split_value]
    right = data[data['X'] >= split_value]

    # Calculate similarity scores for left and right
    left_similarity_score = similarity_score(
        left['Residuals'], left['PreviousProbability'], lambda_)
    right_similarity_score = similarity_score(
        right['Residuals'], right['PreviousProbability'], lambda_)

    # Calculate Gain
    gain = left_similarity_score + right_similarity_score - root_similarity_score
    return gain, left_similarity_score, right_similarity_score


# Calculate gain for each split
gains = []
for split in split_points:
    gain, left_score, right_score = calculate_gain(data, split, lambda_)
    gains.append((split, gain, left_score, right_score))
    print(f"Split at {split}: Gain = {gain}")

# Step 5: Choose the split with the highest gain
best_split = max(gains, key=lambda x: x[1])
best_split_value, best_gain, left_score, right_score = best_split
print(f"\nBest split: X < {best_split_value}, Gain = {best_gain}")

# Calculate the Output for left and right nodes


def calculate_output(residuals, probabilities):
    weighted_residuals = np.sum(probabilities * (1 - probabilities))
    return residuals.sum() / weighted_residuals


left_data = data[data['X'] < best_split_value]
right_data = data[data['X'] >= best_split_value]

left_output = calculate_output(
    left_data['Residuals'], left_data['PreviousProbability'])
right_output = calculate_output(
    right_data['Residuals'], right_data['PreviousProbability'])

print(f"Left Output: {left_output}")
print(f"Right Output: {right_output}")

# Step 6: Make the final prediction for X=25 (which falls into the right node based on the split)
# LogPrediction formula


def log_prediction(prev_prob, lr, output):
    log_pred = np.log(prev_prob / (1 - prev_prob)) + lr * output
    return 1 / (1 + np.exp(-log_pred))


# Since X=25 falls into the right node, use right_output for prediction
new_prediction = log_prediction(f0, lr, right_output)
print(f"\nFinal predicted probability for X=25: {new_prediction}")

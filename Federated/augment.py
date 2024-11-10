# augment.py
import numpy as np
from scipy.interpolate import CubicSpline
import cupy as cp
import random

# No Augmentation
def no_augmentation(x):
    return x

# Jittering
def jitter(x, sigma=0.05):
    return x + np.random.normal(loc=0, scale=sigma, size=x.shape)

# Scaling
def scale(x, sigma=0.5):
    x_scaled = np.copy(x)
    for i in range(x_scaled.shape[0]):
        factor = np.random.normal(loc=1.0, scale=sigma)
        x_scaled[i] *= factor
    return x_scaled

# Flipping
def flip(x):
    if x.ndim == 3 and x.shape[2] == 1:
        return np.flip(x, axis=1)
    else:
        return np.flip(x, axis=0)

# Permutation
def permute(x, num_segments=10):
    split = np.array_split(x, num_segments, axis=0)
    np.random.shuffle(split)
    return np.concatenate(split, axis=0)

# Magnitude Warping
def magnitude_warp(x, sigma=0.001, num_knots=4):
    time_steps = np.linspace(0, 1, num=x.shape[0])
    knot_positions = np.linspace(0, 1, num=num_knots)
    knot_values = np.random.normal(loc=1.0, scale=sigma, size=num_knots)
    spline = CubicSpline(knot_positions, knot_values)
    warp_values = spline(time_steps)
    return x * warp_values

# Rotate
def rotate(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    random_matrix = np.random.normal(size=(x.shape[1], x.shape[1]))
    q, _ = np.linalg.qr(random_matrix)
    rotated_x = np.dot(x, q)
    if rotated_x.shape[0] == 1:
        rotated_x = rotated_x.flatten()
    return rotated_x

# Rotate with GPU
def rotate_gpu(x):
    input_type = type(x)
    x_gpu = cp.asarray(x) if isinstance(x, np.ndarray) else x
    if x_gpu.ndim == 1:
        x_gpu = x_gpu.reshape(1, -1)
    random_matrix = cp.random.normal(size=(x_gpu.shape[1], x_gpu.shape[1]))
    q, _ = cp.linalg.qr(random_matrix)
    rotated_x = cp.dot(x_gpu, q)
    if rotated_x.shape[0] == 1:
        rotated_x = rotated_x.flatten()
    return cp.asnumpy(rotated_x) if input_type is np.ndarray else rotated_x

# Time Warping
def time_warp(x, sigma=0.2, num_knots=4):
    if x.ndim == 1:
        original_steps = np.arange(len(x))
        new_steps = np.linspace(0, len(x) - 1, num=len(x))
    elif x.ndim == 2 and x.shape[1] == 1:
        original_steps = np.arange(x.shape[0])
        new_steps = np.linspace(0, x.shape[0] - 1, num=x.shape[0])
    else:
        raise ValueError("Unsupported data shape for time_warp")
    knot_positions = np.linspace(0, len(new_steps) - 1, num=num_knots)
    knot_values = np.random.normal(loc=0.0, scale=sigma, size=num_knots) + knot_positions
    spline = CubicSpline(knot_positions, knot_values)
    warped_steps = spline(new_steps)
    warped_steps[warped_steps < 0] = 0
    warped_steps[warped_steps > len(new_steps) - 1] = len(new_steps) - 1
    return np.array([x[int(warped_step)] if x.ndim == 1 else x[int(warped_step), 0] for warped_step in warped_steps])

# Window Slicing
def window_slice(x, window_size=50):
    if window_size >= len(x):
        raise ValueError("Window size must be smaller than the length of the data")
    start = np.random.randint(0, len(x) - window_size)
    windowed_data = x[start:start + window_size]
    padding_length = len(x) - window_size
    padded_data = np.pad(windowed_data, (0, padding_length), mode='constant')
    return padded_data

# Window Warping
def window_warp(x, window_size=50, scale=2):
    if len(x) <= window_size:
        raise ValueError("The window size must be less than the size of the data.")
    start = random.randint(0, len(x) - window_size)
    windowed_data = x[start:start + window_size]
    warped_window = np.repeat(windowed_data, scale)
    remove_length = len(warped_window) - window_size
    if remove_length > 0:
        warped_window = warped_window[:-remove_length]  
    return np.concatenate([x[:start], warped_window, x[start + window_size:]])



def leva_embedding_augmentation(embeddings, labels, graph):
    augmented_embeddings = []
    augmented_labels = []

    for i in range(len(embeddings)):
        # Add the original embedding
        augmented_embeddings.append(embeddings[i])
        augmented_labels.append(labels[i])

        # Use graph-based relationships for augmentation
        related_indices = graph.get(i, [])
        for index in related_indices:
            relational_embedding = embeddings[i] + embeddings[index]
            augmented_embeddings.append(relational_embedding / 2)  # Averaging relational embedding
            augmented_labels.append(labels[i])

    return np.array(augmented_embeddings), np.array(augmented_labels)

def random_walk_embedding_augmentation(embedding, label):
    """
    Augments the embedding based on the multi-hot encoded label, 
    but skips augmentation if the label has no active classes.

    Args:
        embedding (numpy.ndarray): The embedding to augment.
        label (numpy.ndarray): The multi-hot encoded label (vector of length 5).
    
    Returns:
        augmented_embedding (numpy.ndarray): Augmented or original embedding.
        augmented_label (numpy.ndarray): The same label.
    """
    # Check if the label has any active classes
    active_classes = np.where(label == 1)[0]
    
    if len(active_classes) == 0:
        # Skip augmentation if the label is empty (no active class)
        print("Skipping augmentation for empty label.")
        return embedding, label

    # Apply augmentation to the embedding if there are active classes
    augmented_embedding = embedding * np.random.uniform(0.9, 1.1, size=embedding.shape)

    return augmented_embedding, label




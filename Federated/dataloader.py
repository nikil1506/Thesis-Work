# data_loader.py

import pickle
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def load_train_data(train_path):
    with open(train_path, 'rb') as f:
        train_x = pickle.load(f)
    return train_x

def load_test_data(test_path):
    with open(test_path, 'rb') as f:
        test_x = pickle.load(f)
    return test_x

def load_train_labels(train_labels_path):
    return np.load(train_labels_path, allow_pickle=True)

def load_test_labels(test_labels_path):
    return np.load(test_labels_path, allow_pickle=True)

def prepare_data(train_x, test_x, train_y, test_y):
    mlb = MultiLabelBinarizer()
    train_y_tensor = mlb.fit_transform(train_y)
    test_y_tensor = mlb.transform(test_y)  
    
    train_data = np.array(train_x['lead_1'])
    test_data = np.array(test_x['lead_1'])
    
    return train_data, train_y_tensor, test_data, test_y_tensor, mlb

def prepare_test_data(test_x, test_y):
    mlb = MultiLabelBinarizer()
    test_y_tensor = mlb.fit_transform(test_y)
    test_data = np.array(test_x['lead_1'])
    return test_data, test_y_tensor, mlb

def prepare_train_data(train_data, train_labels):
    # Add your preprocessing steps here, such as normalization, reshaping, etc.
    # Example:
    train_data = np.array(train_data)  # Convert to numpy array if it's not already
    train_labels = np.array(train_labels)  # Convert labels to numpy
    return train_data, train_labels, None  # Example return structure


def print_data_shapes(train_x, train_y, test_x, test_y):
    print(train_x['lead_2'].shape, train_y.shape, test_x['lead_2'].shape, test_y.shape)
    for i in range(20):
        print(test_y[i])
    print("Space")
    for i in range(20):
        print(train_y[i])

def print_class_distribution(test_y_tensor, mlb):
    samples_per_class = test_y_tensor.sum(axis=0)
    for class_name, count in zip(mlb.classes_, samples_per_class):
        print(f"Class {class_name}: {count} samples")

# Example usage:
# train_x = load_train_data(train_data_path)
# test_x = load_test_data(test_data_path)
# train_y = load_train_labels(train_labels_path)
# test_y = load_test_labels(test_labels_path)
# train_data, train_y_tensor, test_data, test_y_tensor, mlb = prepare_data(train_x, test_x, train_y, test_y)
# print_data_shapes(train_x, train_y, test_x, test_y)
# print_class_distribution(test_y_tensor, mlb)

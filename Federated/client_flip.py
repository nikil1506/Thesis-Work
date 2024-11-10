# client_flip.py

import numpy as np
import tensorflow as tf
import flwr as fl
import os
import datetime
from dataloader import load_train_data, load_test_data, load_train_labels, load_test_labels, prepare_data, print_data_shapes, print_class_distribution
from augment import flip  # Import other augmentation functions as needed
from model import build_model
from utilities import plot_graphs

# Define paths
train_data_path = r"D:\Research Work\Thesis\PTBXL\500Hz\500Hz\train_leads500.pkl"
test_data_path = r"D:\Research Work\Thesis\PTBXL\500Hz\500Hz\test_leads500.pkl"
train_labels_path = r"D:\Research Work\Thesis\PTBXL\500Hz\500Hz\y_train_labels500.npy"
test_labels_path = r"D:\Research Work\Thesis\PTBXL\500Hz\500Hz\y_test_labels500.npy"

# Load and prepare data
train_data, train_y_tensor, test_data, test_y_tensor, mlb = prepare_data(
    load_train_data(train_data_path),
    load_test_data(test_data_path),
    load_train_labels(train_labels_path),
    load_test_labels(test_labels_path)
)

# Print data shapes and class distribution
print_data_shapes(load_train_data(train_data_path), load_train_labels(train_labels_path), load_test_data(test_data_path), load_test_labels(test_labels_path))
print_class_distribution(test_y_tensor, mlb)

batchsize = 128
num_classes = train_y_tensor.shape[1]
class_counts = [0] * num_classes

# Augmentation Data Generator
def augment_data_generator(x, y, batch_size, augment_function):
    global class_counts
    while True:
        augmented_x = np.array([augment_function(sample) for sample in x])
        doubled_x = np.concatenate((x, augmented_x), axis=0)
        doubled_y = np.concatenate((y, y), axis=0)
        class_counts = np.sum(doubled_y, axis=0)

        indices = np.arange(len(doubled_x))
        np.random.shuffle(indices)

        for i in range(0, len(doubled_x), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_x = doubled_x[batch_indices]
            batch_y = doubled_y[batch_indices]
            yield batch_x, batch_y

# Model Checkpoint Callback
ckpt_folder  = os.path.join(os.getcwd(), 'Ckpt_Single_Lead')
ckpt_path = os.path.join(ckpt_folder, 'CustomCNN_Lead0_{epoch}')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    save_weights_only=True,
    monitor='val_AUC',
    mode='max',
    save_best_only=True,
    verbose=1,
)

# Build and compile model
model = build_model()
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.legacy.Adam(decay=.01, learning_rate=0.001, beta_1=.009, beta_2=.8, epsilon=1e-08),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Recall(name='Recall'),
        tf.keras.metrics.Precision(name='Precision'),
        tf.keras.metrics.AUC(name='AUC')
    ]
)

selected_augmentation = flip  # Replace with any other function as needed

# Define the federated client
class MyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        train_generator = augment_data_generator(train_data, train_y_tensor, batchsize, selected_augmentation)
        history = model.fit(train_generator, epochs=1, steps_per_epoch=2 * (len(train_data) // batchsize), validation_data=(test_data, test_y_tensor))
        
        # Plot training history
        plot_graphs(history)
        
        return model.get_weights(), len(train_data), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        results = model.evaluate(test_data, test_y_tensor)
        loss = results[0]
        metrics = {name: value for name, value in zip(model.metrics_names, results)}
        return loss, len(test_data), metrics

# Start the federated client
fl.client.start_numpy_client(server_address="localhost:9000", client=MyClient())

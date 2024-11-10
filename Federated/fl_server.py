import flwr as fl
import tensorflow as tf
import os
from dataloader import load_test_data, load_test_labels, prepare_test_data
from model import build_model
from typing import List, Tuple, Dict

# Define paths for evaluation
test_data_path = r"D:\Research Work\Thesis\PTBXL\500Hz\500Hz\test_leads500.pkl"
test_labels_path = r"D:\Research Work\Thesis\PTBXL\500Hz\500Hz\y_test_labels500.npy"

# Load and prepare test data
test_data, test_y_tensor, mlb = prepare_test_data(
    load_test_data(test_data_path),
    load_test_labels(test_labels_path)
)


import tensorflow as tf

# List physical devices (GPUs)
gpus = tf.config.list_physical_devices('GPU')

# If GPUs are available, configure memory
if gpus:
    try:
        # Set memory limit or fraction per client
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # Set 4GB memory per client
    except RuntimeError as e:
        print(f"Error setting GPU memory configuration: {e}")




# Build model for evaluation
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

# Define the evaluation function
def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round, parameters_ndarrays, config):
        # Set model weights from the current round parameters
        model.set_weights(parameters_ndarrays)

        # Evaluate the model on the test set
        results = model.evaluate(test_data, test_y_tensor, verbose=0)

        # Save the model after evaluation
        model_save_path = f"./saved_models/round_{server_round}_model.h5"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        print(f"Saved global model for round {server_round} to {model_save_path}")

        # Unpack results (loss and metrics)
        loss = results[0]
        metrics = {name: value for name, value in zip(model.metrics_names, results)}
        return loss, metrics

    return evaluate

# Aggregation function for fit metrics
def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    aggregated_metrics = {}
    for metric_name in metrics[0][1].keys():
        aggregated_metrics[metric_name] = sum([client_metrics[1][metric_name] for client_metrics in metrics]) / len(metrics)
    return aggregated_metrics

# Aggregation function for evaluate metrics
def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    aggregated_metrics = {}
    for metric_name in metrics[0][1].keys():
        aggregated_metrics[metric_name] = sum([client_metrics[1][metric_name] for client_metrics in metrics]) / len(metrics)
    return aggregated_metrics

# Create strategy with server-side evaluation
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,  # Fraction of clients used during training
    fraction_evaluate=0.1,  # Fraction of clients used during evaluation
    min_fit_clients=2,  # Minimum number of clients used during training
    min_evaluate_clients=2,  # Minimum number of clients used during evaluation
    min_available_clients=2,  # Minimum number of clients available for the round
    evaluate_fn=get_evaluate_fn(model),  # Server-side evaluation function
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,  # Aggregation function for fit metrics
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn  # Aggregation function for evaluate metrics
)

# Start the server
fl.server.start_server(
    server_address="localhost:9000",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)

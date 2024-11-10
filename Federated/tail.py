import numpy as np
import tensorflow as tf
import flwr as fl
import os
from dataloader import load_train_data, load_test_data, load_train_labels, load_test_labels, prepare_data
from model import build_model, build_embedding_model
from utilities import plot_graphs
from variance import calculate_time_step_angular_variance  
# Set up GPU memory growth (if necessary)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Define paths
train_data_path = r"C:\Research Work\Thesis\PTBXL\500Hz\500Hz\500Hz\train_leads500.pkl"
test_data_path = r"C:\Research Work\Thesis\PTBXL\500Hz\500Hz\500Hz\test_leads500.pkl"
train_labels_path = r"C:\Research Work\Thesis\PTBXL\500Hz\500Hz\500Hz\y_train_labels500.npy"
test_labels_path = r"C:\Research Work\Thesis\PTBXL\500Hz\500Hz\500Hz\y_test_labels500.npy"

# Load and prepare data
train_data, train_y_tensor, test_data, test_y_tensor, mlb = prepare_data(
    load_train_data(train_data_path),
    load_test_data(test_data_path),
    load_train_labels(train_labels_path),
    load_test_labels(test_labels_path)
)

# Print data shapes (Optional for debugging)
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Build the full model
model = build_model()

# Split the model at 'conv5'
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('conv5').output)

# Build the embedding model (Part 2) that takes 'conv5' embeddings as input
embedding_model = build_embedding_model(input_shape=intermediate_layer_model.output_shape[1:])

# Function to extract embeddings in batches
def extract_embeddings_in_batches(model, data, batch_size=32, layer_name='conv5'):
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    num_samples = data.shape[0]
    embeddings = []

    # Process data in batches
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_data = data[start:end]
        print(f"Processing batch {start} to {end}, batch size: {batch_data.shape[0]}")
        batch_embeddings = intermediate_layer_model.predict(batch_data)
        embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings into one array
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Final embeddings shape: {embeddings.shape}")
    return embeddings

# New augmentation function using class variances
def augment_with_class_variances(embeddings, labels, variances):
    print(f"Augmenting embeddings with class variances...")
    augmented_embeddings = []
    augmented_labels = []
    
    for embedding, label in zip(embeddings, labels):
        if np.sum(label) == 0:  # Skip if no active label
            continue
        
        class_index = np.argmax(label)  # Get the class index from the label
        class_variance = variances[class_index]  # Get the corresponding class variance list
        
        # Apply the variance to each time step in the embedding
        augmented_embedding = embedding * (1 + np.array(class_variance)[:, np.newaxis])
        
        augmented_embeddings.append(augmented_embedding)
        augmented_labels.append(label)
    
    augmented_embeddings = np.array(augmented_embeddings)
    augmented_labels = np.array(augmented_labels)
    
    print(f"Final augmented embeddings shape: {augmented_embeddings.shape}")
    return augmented_embeddings, augmented_labels


# Extract intermediate embeddings for test data
def extract_embeddings_for_validation(model, test_data, batch_size=32, layer_name='conv5'):
    print("Extracting embeddings for validation data...")
    return extract_embeddings_in_batches(model, test_data, batch_size=batch_size, layer_name=layer_name)

# Model Training Function with Augmented Embeddings after 'conv5'
def train_with_embedding_switch(model, embedding_model, train_data, train_y_tensor, test_data, test_y_tensor, variances, batch_size=32):
    # Step 1: Train up to the 'conv5' layer using original data
    print("Training the model up to 'conv5'...")
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.legacy.Adam(decay=.01, learning_rate=0.001, beta_1=.009, beta_2=.8, epsilon=1e-08),
        metrics=['accuracy']
    )

    model.fit(train_data, train_y_tensor, epochs=100, batch_size=batch_size, validation_data=(test_data, test_y_tensor))  

    # Step 2: Extract embeddings from 'conv5' layer
    print("Extracting embeddings from 'conv5' layer...")
    embeddings = extract_embeddings_in_batches(model, train_data, layer_name='conv5')

    # Extract validation embeddings
    validation_embeddings = extract_embeddings_for_validation(model, test_data, layer_name='conv5')

    # Step 3: Augment embeddings using class variances
    print("Augmenting embeddings with class variances...")
    augmented_embeddings, augmented_labels = augment_with_class_variances(embeddings, train_y_tensor, variances)

    # Step 4: Train the embedding model
    print("Training the embedding model with augmented embeddings...")
    embedding_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.legacy.Adam(),
        metrics=['accuracy', 'Recall', 'Precision', 'AUC']
    )
    history = embedding_model.fit(
        augmented_embeddings,
        augmented_labels,
        epochs=100,  # Adjust as needed
        batch_size=128,
        validation_data=(validation_embeddings, test_y_tensor)
    )
    
    return history

# Extract intermediate embeddings from the 'conv5' layer
embeddings = extract_embeddings_in_batches(model, train_data, batch_size=32, layer_name='conv5')


class_variances = calculate_time_step_angular_variance(embeddings, train_y_tensor, mlb)



# Define the federated client
class MyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)

        # Train with embedding switch using class variances
        history = train_with_embedding_switch(model, embedding_model, train_data, train_y_tensor, test_data, test_y_tensor, class_variances)

        return model.get_weights(), len(train_data), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        results = model.evaluate(test_data, test_y_tensor)
        loss = results[0]
        metrics = {name: value for name, value in zip(model.metrics_names, results)}
        return loss, len(test_data), metrics

# Start the federated client
fl.client.start_numpy_client(server_address="localhost:9000", client=MyClient())

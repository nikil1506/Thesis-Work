import numpy as np
import tensorflow as tf
from dataloader import load_train_data, load_test_data, load_train_labels, load_test_labels, prepare_data
from model import build_model  # Assuming you have a model-building function in a separate script

# Set up GPU memory growth (if necessary)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Define paths
train_data_path = r"D:\Research Work\Thesis\PTBXL\500Hz\500Hz\train_leads500.pkl"
test_data_path = r"D:\Research Work\Thesis\PTBXL\500Hz\500Hz\test_leads500.pkl"
train_labels_path = r"D:\Research Work\Thesis\PTBXL\500Hz\500Hz\y_train_labels500.npy"
test_labels_path = r"D:\Research Work\Thesis\PTBXL\500Hz\500Hz\y_test_labels500.npy"

# Load and prepare data using dataloader.py
train_data, train_y_tensor, test_data, test_y_tensor, mlb = prepare_data(
    load_train_data(train_data_path),
    load_test_data(test_data_path),
    load_train_labels(train_labels_path),
    load_test_labels(test_labels_path)
)

# Build the full model
model = build_model()

# Extract embeddings from 'conv5' layer
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

# Function to calculate and print angular variance
def calculate_time_step_angular_variance(embeddings, labels, mlb):
    class_centers = {}
    class_variances = {}
    num_classes = labels.shape[1]
    num_time_steps = embeddings.shape[1]

    # Calculate the center for each class at each time step
    for class_index in range(num_classes):
        # Get all the embeddings that belong to the current class
        class_embeddings = embeddings[labels[:, class_index] == 1]
        if len(class_embeddings) > 0:
            time_step_variances = []
            
            for time_step in range(num_time_steps):
                # Skip time steps where there are no embeddings
                if np.all(labels[:, class_index] == 0):
                    time_step_variances.append(np.nan)
                    continue
                
                # Calculate the class center at the current time step
                class_center = np.mean(class_embeddings[:, time_step, :], axis=0)
                norm_class_center = np.linalg.norm(class_center)
                
                # Ensure no zero-norm vectors to avoid invalid divisions
                if norm_class_center == 0:
                    time_step_variances.append(np.nan)
                    continue

                # Calculate the angles for non-zero norm embeddings
                norms_class_embeddings = np.linalg.norm(class_embeddings[:, time_step, :], axis=1)
                valid_embeddings = norms_class_embeddings > 0  # Only keep valid embeddings
                valid_class_embeddings = class_embeddings[valid_embeddings][:, time_step, :]
                valid_norms = norms_class_embeddings[valid_embeddings]

                # Calculate angular variance
                if len(valid_class_embeddings) > 0:
                    angles = np.arccos(np.clip(np.dot(valid_class_embeddings, class_center) /
                                               (valid_norms * norm_class_center), -1.0, 1.0))
                    angular_variance = np.var(angles)
                    time_step_variances.append(angular_variance)
                else:
                    time_step_variances.append(np.nan)
            
            # Store variance results
            class_variances[class_index] = time_step_variances
        else:
            class_variances[class_index] = None

    # Print for debugging
    print("Angular Variance per Class, Across Time Steps:")
    for class_index, variances in class_variances.items():
        class_name = mlb.classes_[class_index]
        print(f"Class {class_name}: Variances across time steps: {variances}")

    return class_variances


# Extract embeddings for training data
train_embeddings = extract_embeddings_in_batches(model, train_data, layer_name='conv5')

angular_variances = calculate_time_step_angular_variance(train_embeddings, train_y_tensor, mlb)





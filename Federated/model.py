# model.py
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D

def build_model():
    model = Sequential()

    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(5000, 1), padding='same', name='conv1'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))

    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv2'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same', name='conv3'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(256, kernel_size=3, activation='relu', padding='same', name='conv4'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))

    model.add(Conv1D(512, kernel_size=3, activation='relu', padding='same', name='conv5'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(1024, kernel_size=3, activation='relu', padding='same', name='conv6'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.6))

    model.add(Conv1D(2048, kernel_size=3, activation='relu', padding='same', name='conv7'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.7))

    model.add(Dense(units=5, activation='sigmoid', name='output'))

    return model

# Example usage:
# model = build_model()
# model.summary()

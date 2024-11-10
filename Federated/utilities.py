# utilities.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_graphs(history):
    plt.plot(history.epoch, history.history["loss"], 'b', label='Training loss')
    plt.plot(history.epoch, history.history["val_loss"], 'g', label='Validation loss')
    plt.legend()
    plt.title("Loss")
    plt.show()

    plt.plot(history.epoch, history.history["accuracy"], 'b', label='Training accuracy')
    plt.plot(history.epoch, history.history["val_accuracy"], 'g', label='Validation accuracy')
    plt.legend()
    plt.title("Accuracy")
    plt.show()
    
    plt.plot(history.epoch, history.history["AUC"], 'b', label='Training AUC')
    plt.plot(history.epoch, history.history["val_AUC"], 'g', label='Validation AUC')
    plt.legend()
    plt.title("AUC")
    plt.show()

def evaluate_model(model, data_test, labels_test):
    score = model.evaluate(data_test, labels_test, verbose=1)
    print("\nAccuracy on ORIGINAL test data: %0.2f" % score[1])
    print("\nLoss on ORIGINAL test data: %0.2f" % score[0])
    return score

def model_predict(model, data_test, labels_test):
    y_pred_test = model.predict(data_test)
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(labels_test, axis=1)
    return max_y_test, max_y_pred_test 

from sklearn import metrics

def show_confusion_matrix(validations, predictions, labels):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns_hm = sns.heatmap(matrix, cmap="YlGnBu", linecolor='white', linewidths=1, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    figure = sns_hm.get_figure()  
    plt.show()
    return figure

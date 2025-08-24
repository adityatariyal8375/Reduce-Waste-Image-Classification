import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import os
import tensorflow as tf

class ModelEvaluator:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

    def evaluate(self, X_test, y_test):
        print("\n📊 Evaluating model...")
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names, cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    def save_model(self, path="artifacts/model/waste_classifier.h5"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"💾 Model saved to {path}")

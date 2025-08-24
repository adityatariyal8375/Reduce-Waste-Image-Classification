import sys
import json
import os
import yaml
from sklearn.model_selection import train_test_split

# Add the src folder to sys.path so Python can find the package
sys.path.append(os.path.join(os.getcwd(), "src"))

from imageClassifier.pipeline.data_ingestion import DataIngestion
from imageClassifier.pipeline.data_preprocessing import DataPreprocessing
from imageClassifier.pipeline.model_training import ModelTraining
from imageClassifier.pipeline.model_evaluation import ModelEvaluator

import matplotlib.pyplot as plt

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Accuracy plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Step 1: Data Ingestion
    ingestion = DataIngestion()
    ingestion.prepare_data()

    # Step 2: Load config
    with open(os.path.join("config", "config.yaml")) as f:
        config = yaml.safe_load(f)['data_ingestion']

    # Get class names from train dir
    train_dir = config['train_dir']
    class_names = sorted(os.listdir(train_dir))

    # Step 3: Data Preprocessing
    # Add image_size into the config dictionary before passing it in
    config['image_size'] = 64
    preprocessing = DataPreprocessing(config)

    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    # Step 4: Train/Validation Split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Step 5: Model Training
    num_classes = len(class_names)
    image_size = 64
    trainer = ModelTraining(image_size, num_classes)
    model = trainer.build_model()

    print("Starting training...")
    history = trainer.train(model, X_train_final, y_train_final, X_val, y_val, epochs=4)
    print("Training complete.")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/class_names.json", "w") as f:
        json.dump(class_names, f)

    print("Class names saved successfully!")

    plot_training_history(history)

    # Save the trained model
    model_save_path = "artifacts/model/my_model.h5"
    model.save(model_save_path)
    print(f"Model saved at: {model_save_path}")



    # Step 6: Model Evaluation
    evaluator = ModelEvaluator(model, class_names)
    evaluator.evaluate(X_test, y_test)

    # Step 7: Save the model
    evaluator.save_model()

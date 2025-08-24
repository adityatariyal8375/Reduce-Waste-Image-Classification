import os
import shutil
import yaml

class DataIngestion:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['data_ingestion']

    def prepare_data(self):
        # Load paths from config
        source_train_dir = self.config['source_train_dir']
        source_test_dir = self.config['source_test_dir']
        train_dir = self.config['train_dir']
        test_dir = self.config['test_dir']

        # Remove existing artifact directories if they exist
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        # Create parent artifact directories if needed
        os.makedirs(os.path.dirname(train_dir), exist_ok=True)
        os.makedirs(os.path.dirname(test_dir), exist_ok=True)

        # Copy training data
        shutil.copytree(source_train_dir, train_dir)
        print(f"✅ Copied training data from {source_train_dir} to {train_dir}")

        # Copy testing data
        shutil.copytree(source_test_dir, test_dir)
        print(f"✅ Copied testing data from {source_test_dir} to {test_dir}")

        print("✅ Data ingestion complete.")

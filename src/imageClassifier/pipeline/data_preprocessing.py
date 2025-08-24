import os
import cv2
import numpy as np

class DataPreprocessing:
    def __init__(self, config):
        self.train_dir = config['train_dir']
        self.test_dir = config['test_dir']
        self.image_size = config.get('image_size', 64)  
        self.class_names = []


    def augment_image(self, img):
        # Random horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        # Random rotation
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

        # Random brightness
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[..., 2] *= random.uniform(0.7, 1.3)
        hsv[..., 2][hsv[..., 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img

    def load_images(self, directory):
        images = []
        labels = []
        class_names = os.listdir(directory)
        class_names.sort()  # to have consistent label order
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

        for class_name in class_names:
            class_path = os.path.join(directory, class_name)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (self.image_size, self.image_size))
                img = img / 255.0  # normalize to [0,1]
                images.append(img)
                labels.append(class_to_idx[class_name])
        return np.array(images), np.array(labels)

    def preprocess(self):
        print("Loading training data...")
        X_train, y_train = self.load_images(self.train_dir)
        print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")

        print("Loading testing data...")
        X_test, y_test = self.load_images(self.test_dir)
        print(f"Testing data: {X_test.shape}, Labels: {y_test.shape}")

        return X_train, y_train, X_test, y_test

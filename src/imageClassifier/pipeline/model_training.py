import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout


class ModelTraining:
    def __init__(self, image_size, num_classes):
        self.image_size = image_size
        self.num_classes = num_classes

    def build_model(self):
        base_model = MobileNetV2(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # freeze base layers

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)  
        predictions = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, model, X_train, y_train, X_val, y_val, epochs=5, batch_size=16):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )

        model.save("artifacts/data_ingestion/model/my_model.h5")
        return history

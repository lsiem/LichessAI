import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score

def preprocess(images):
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images

class ImageClassifier:
    def __init__(self, train_dir, test_dir, val_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.label_encoder = LabelEncoder()
        self.train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            preprocessing_function=preprocess
        )
        self.val_datagen = ImageDataGenerator(preprocessing_function=preprocess)
        self.test_datagen = ImageDataGenerator(preprocessing_function=preprocess)
        try:
            self.train_generator = self.train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical')
            self.val_generator = self.val_datagen.flow_from_directory(
                self.val_dir, 
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical')
            self.test_generator = self.test_datagen.flow_from_directory(
                self.test_dir, 
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical')
        except Exception as e:
            print(f"Error occurred while loading images: {e}")
            return
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), kernel_regularizer=l2(0.01)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(len(self.train_generator.class_indices), activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', Precision(), Recall(), F1Score(num_classes=len(self.train_generator.class_indices))])
        return model

    def train_model(self):
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
        learning_rate_scheduler = LearningRateScheduler(scheduler)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = ModelCheckpoint('chess_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        self.model.fit(self.train_generator, epochs=50, validation_data=self.val_generator, callbacks=[early_stopping, model_checkpoint, tensorboard, learning_rate_scheduler, reduce_lr], batch_size=32)
        self.model.save('chess_model.keras')

try:
    claclassifier = ImageClassifier('data/train', 'data/test', 'data/val')
    claclassifier.train_model()
except Exception as e:
    print(f"Error occurred while training the model: {e}")



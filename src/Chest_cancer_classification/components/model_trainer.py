from Chest_cancer_classification.config.configuration import TrainingConfig
import os
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import shutil
# UPDATED: Import both ReduceLROnPlateau and EarlyStopping
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.validation_dir,
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=30,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_dir,
            shuffle=True,
            **dataflow_kwargs
        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.train_generator.classes),
            y=self.train_generator.classes
        )
        class_weights = dict(enumerate(class_weights))

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_recall', 
            mode='max',
            factor=0.3, 
            patience=5, 
            min_lr=0.000001
        )
        
        # UPDATED: Define the EarlyStopping callback
        early_stopper = EarlyStopping(
            monitor='val_recall', # Monitor the same metric
            mode='max',
            patience=10,          # Stop if no improvement after 10 epochs
            restore_best_weights=True # Restore the best model weights
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            class_weight=class_weights,
            # UPDATED: Add the early_stopper to the callbacks list
            callbacks=[lr_scheduler, early_stopper]
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        model_deployment_path = os.path.join("model", "model.h5")
        shutil.copy(self.config.trained_model_path, model_deployment_path)
        print(f"Model saved for deployment at: {model_deployment_path}")
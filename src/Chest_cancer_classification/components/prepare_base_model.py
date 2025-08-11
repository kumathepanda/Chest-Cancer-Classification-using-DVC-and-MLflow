import tensorflow as tf
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config):
        self.config = config

    def get_base_model(self):
        """Loads DenseNet121 as the base model."""
        self.model = tf.keras.applications.DenseNet121(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """Unfreezes the top layers of the model for fine-tuning."""
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        # This logic unfreezes the last 'freeze_till' layers
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Add our custom head
        flatten_in = tf.keras.layers.Flatten()(model.output)
        dropout = tf.keras.layers.Dropout(0.5)(flatten_in)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(dropout)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.Recall(name="recall")]
        )
        full_model.summary()
        return full_model

    def update_base_model(self):
        """Prepares and saves the model for fine-tuning."""
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            # We will now fine-tune the model
            freeze_all=False,
            # Unfreeze the top 10 layers for fine-tuning
            freeze_till=10,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
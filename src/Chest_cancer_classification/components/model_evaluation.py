import mlflow
import mlflow.keras
from urllib.parse import urlparse
from Chest_cancer_classification.entity.config_entity import EvaluationConfig
from Chest_cancer_classification.utils.common import save_json,create_directories,read_yaml
from sklearn.metrics import f1_score,recall_score,precision_score,roc_auc_score
from pathlib import Path
import tensorflow as tf
import numpy as np

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _test_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        
        self.test_generator = test_datagenerator.flow_from_directory(
            directory=self.config.test_dir,
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._test_generator()
        
        self.score = self.model.evaluate(self.test_generator, verbose=1)
        
        y_true = self.test_generator.classes
        y_pred_probs = self.model.predict(self.test_generator, verbose=1)
        threshold = 0.48
        y_pred = (y_pred_probs[:,1]>=threshold).astype(int)
        
        auc = roc_auc_score(y_true,y_pred_probs,multi_class="ovr",average="weighted")
        precision = precision_score(y_true, y_pred, average="weighted",zero_division=1)
        recall = recall_score(y_true, y_pred, average="weighted",zero_division=1)
        f1 = f1_score(y_true, y_pred, average="weighted",zero_division=1)

        self.metrics = {
            "loss": self.score[0],
            "accuracy": self.score[1],
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc
        }

        self.save_score()

    def save_score(self):
        scores = self.metrics
        save_json(path=Path("scores.json"), data=scores)
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                self.metrics
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="DenseNet121")
            else:
                mlflow.keras.log_model(self.model, "model")
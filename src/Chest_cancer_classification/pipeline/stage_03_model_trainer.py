from Chest_cancer_classification.config.configuration import ConfigurationManager
from Chest_cancer_classification.components.model_trainer import Training
from Chest_cancer_classification import logger

STAGE_NAME = "Model Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_training_config()
        model_trainer = Training(config=model_training_config)
        model_trainer.get_base_model()
        model_trainer.train_valid_generator()
        model_trainer.train()
        
if __name__ == "__main__":
    try:
        logger.info(f"<<<<< stage {STAGE_NAME} is initiated <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed >>>>>\n")
    except Exception as e:
        logger.exception(e)
        raise e
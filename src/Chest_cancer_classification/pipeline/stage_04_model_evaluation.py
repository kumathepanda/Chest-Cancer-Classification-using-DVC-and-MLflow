from Chest_cancer_classification.config.configuration import ConfigurationManager
from Chest_cancer_classification.components.model_evaluation import Evaluation
from Chest_cancer_classification import logger


STAGE_NAME = "Model Evaluation"

class ModelEvalPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        #evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f"<<<<< stage {STAGE_NAME} is initiated <<<<<")
        obj = ModelEvalPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed >>>>>\n")
    except Exception as e:
        logger.exception(e)
        raise e
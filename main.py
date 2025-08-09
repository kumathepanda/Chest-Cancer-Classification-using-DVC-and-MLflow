from Chest_cancer_classification import logger
from Chest_cancer_classification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Chest_cancer_classification.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from Chest_cancer_classification.pipeline.stage_03_model_trainer import ModelTrainingPipeline
STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"<<<<< stage {STAGE_NAME} has initiated >>>>>")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} has been completed >>>>>\n")
except Exception as e:
    logger.exception(e)
    raise e
        

STAGE_NAME = "Preapre Base Model"


try:
    logger.info(f"<<<<< stage {STAGE_NAME} is initiated <<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed >>>>>\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training"

try:
    logger.info(f"<<<<< stage {STAGE_NAME} is initiated <<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed >>>>>\n")
except Exception as e:
        logger.exception(e)
        raise e
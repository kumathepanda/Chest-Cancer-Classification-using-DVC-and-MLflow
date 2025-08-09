from Chest_cancer_classification import logger
from Chest_cancer_classification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"<<<<< stage {STAGE_NAME} has initiated >>>>>")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} has been completed >>>>>\n")
except Exception as e:
    logger.exception(e)
    raise e
        
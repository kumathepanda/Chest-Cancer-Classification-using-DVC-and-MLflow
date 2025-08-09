import os 
import zipfile
import gdown
from Chest_cancer_classification import logger
from Chest_cancer_classification.utils.common import get_size 
from Chest_cancer_classification.entity.config_entity import (DataIngestionConfig)

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config
        

    def download_file(self)->str:
        '''
        fetch data from url
        '''
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion",exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")
            
            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)
            
            logger.info(f"Download data from {dataset_url} into file {zip_download_dir}")
            
        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        """
        zip_filepath:str
        Extracts the zip file into the data directory
        function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_file:
            zip_file.extractall(unzip_path)
            
            
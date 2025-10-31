import os
import sys
from src.exception import CustomException
from src.loggers import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfiq:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join("artifacts",'test.csv')
    raw_data_path:str = os.path.join("artifacts",'data.csv')

class DataIngestion:
    def __init__(self):
         self.ingestion_confiq=DataIngestionConfiq()

    def initiate_ingestion(self):
        logging.info("Data Ingestion Initiated")
        try:
            df=pd.read_csv('notebook/data/StudentsPerformance.csv')
            os.makedirs(os.path.dirname(self.ingestion_confiq.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_confiq.raw_data_path,index=False,header=True)

            logging.info("Train Test Split initiated")

            train_set,test_set=train_test_split(df,test_size=0.33,random_state=42)
            train_set.to_csv(self.ingestion_confiq.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_confiq.test_data_path,index=False,header=True)

            return(
                self.ingestion_confiq.train_data_path,
                self.ingestion_confiq.test_data_path

            )

        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_ingestion()

    data_transformer=DataTransformation()

    test_arr,train_arr,_=data_transformer.initiate_data_trans(train_data,test_data)
    


import sys
import os
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.loggers import logging
from src.utils import save_object

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','processor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_trans_config=DataTransformationConfig()

    def get_data_trans_obj(self):
        try:
            num_feature=['writing score','reading score']
            cat_feature=["gender","race/ethnicity","parental level of education","lunch","test preparation course"]

            num_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='median')),
                    ("Scalar",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ('One_hot_encoder',OneHotEncoder()),
                    ('Scalar',StandardScaler(with_mean=False))
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ("Num_Pipeline",num_pipeline,num_feature),
                    ("Cat_piprline",cat_pipeline,cat_feature)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_trans(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            preprocessor_obj=self.get_data_trans_obj()

            input_train_df=train_df.drop('math score',axis=1)
            target_train_df=train_df['math score']

            input_test_df=test_df.drop('math score',axis=1)
            target_test_df=test_df['math score']

            input_train_arr=preprocessor_obj.fit_transform(input_train_df)
            input_test_arr=preprocessor_obj.transform(input_test_df)

            train_arr=np.c_[
                input_train_arr,np.array(target_train_df)
            ]
            test_arr=np.c_[
                input_test_arr,np.array(target_test_df)
            ]
            save_object(
                file_path=self.data_trans_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info("Processor file is saved")

            return (train_arr,test_arr,self.data_trans_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)
    

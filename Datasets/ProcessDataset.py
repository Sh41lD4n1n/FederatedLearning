import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale

class Dataset:
    def __init__(self,dataset_name):
        self.d = None
        self.X = None
        self.y = None
        
        self._load_dataset(dataset_name)
        self._preprocess_dataset()


    def _load_dataset(self,dataset_name):
        if dataset_name=='Titanic':
            self.d = pd.read_csv('./Datasets/Titanic/train.csv')
        
        if dataset_name=='House_Rent':
            self.d = pd.read_csv('./house_rent/House_Rent_Dataset.csv')

    def _preprocess_dataset(self):
        self.X = self.d.copy()

        to_drop = ['Ticket','Cabin','Name','PassengerId']
        self.X.drop(columns = to_drop,inplace=True)

        self.X = self.X.dropna()

        # For classification
        self.y = self.X['Survived'].to_numpy()
        self.X.drop(columns = ['Survived'],inplace=True)

        numerical_features = ['Age','SibSp','Parch','Fare']
        scaled_data = minmax_scale(self.X[numerical_features])

        to_encode = ['Sex','Embarked']
        encoder = OneHotEncoder()
        encoder.fit(self.X[to_encode])
        data_encoded = encoder.transform(self.X[to_encode]).toarray()

        self.X.drop(columns = to_encode,inplace=True)
        self.X.drop(columns = numerical_features,inplace=True)
        self.X = self.X.to_numpy()
        self.X = np.concatenate([self.X,data_encoded,scaled_data],axis=1)
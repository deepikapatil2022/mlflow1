import os
import pandas as pd
import numpy as np
import mlflow.sklearn
import mlflow
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split
import argparse

def get_data():
    try:
        df=pd.read_csv("winequality-red.csv",sep=";")
        return df
    except Exception as e:
        raise e
    
def evaluate_model(y_true,y_pred):
    '''mae=mean_absolute_error(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_true,y_pred)
    return mae,mse,rmse,r2'''
    accuracy=accuracy_score(y_true,y_pred)

    return accuracy



def main(n_estimators,max_depth):
  
    try:
        df=get_data()
        train,test=train_test_split(df)
        x_train=train.drop(["quality"],axis=1)
        x_test=test.drop(["quality"],axis=1)

        y_train=train[["quality"]]
        y_test=test[["quality"]]

        #model training

        """lr=ElasticNet()
        lr.fit(x_train,y_train)
        pred=lr.predict(x_test)"""

        rf=RandomForestClassifier()
        rf.fit(x_train,y_train)
        pred=rf.predict(x_test)

        #evaluation of model

        #mae,mse,rmse,r2=evaluate_model(y_test,pred)
        #print(f"mean absolute error={mae}  mean sqaured error={mse},  root mean square error={rmse}, r2 sqaure={r2}")
        accuracy=evaluate_model(y_test,pred)
        print(f'accuracy is: {accuracy}')
        
    except Exception as e:
        raise e
    
if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--n_estimators", "-n", default=50, type=int)
    args.add_argument("--max_depth", "-m", default=5, type=int)
    parse_args=args.parse_args()

    try:
        main(n_estimators=parse_args.n_estimators,max_depth=parse_args.max_depth)  
    except Exception as e:
        raise e

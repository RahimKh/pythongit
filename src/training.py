import os 
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
import numpy as np
import xgboost as xgb
import joblib


TRAINING_DATA="./input/train.csv"
TEST_DATA='./input/test.csv'


if __name__ == "__main__":
	df=pd.read_csv(TRAINING_DATA)
	test=pd.read_csv(TEST_DATA)
	X=df.drop(columns=['target'])
	Y=df.target
	X_train,X_val,y_train,y_val=model_selection.train_test_split(X,Y,test_size=0.2,shuffle=True,random_state=42)
	MODEL="XGBRegressor"
	clf=xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.45, learning_rate = 0.9,
                max_depth = 20, alpha = 12, reg_lambda = 3 , n_estimators = 200)
	clf.fit(X_train,y_train)
	preds=clf.predict(X_val)
	print(np.sqrt(metrics.mean_squared_error(preds,y_val)))

	joblib.dump(clf,f'models/{MODEL}.pkl')

import os
import pandas as pd 
import xgboost as xgb
import joblib


def predict(pathfile,model_type,model_path):
	df=pd.read_csv(pathfile)
	test=pd.read_excel("Dataset.xlsx",sheet_name="Output")
	clf = joblib.load(os.path.join(model_path, f"{model_type}.pkl"))
	preds=clf.predict(df)
	test['target']=preds

	return test

if __name__=="__main__":
	submission=predict(pathfile="./input/test.csv",model_type="XGBRegressor",model_path="./Models")
	submission.to_csv(f'Models/xgb_submission.csv',index=False)
import pandas as pd 
import numpy as np 


if __name__=="__main__":
	DATA="Dataset.xlsx"
	df=pd.read_excel(DATA,sheet_name="Dataset")
	test=pd.read_excel(DATA,sheet_name="Output")
	site=pd.read_excel(DATA,sheet_name="Information")

	combine=[df,test]

	for dataset in combine:
		dataset.X=dataset.X-site.X0.values
		dataset.Y=dataset.Y-site.Y0.values
		dataset['distance']=np.sqrt((dataset.X)**2+dataset.Y**2)
		dataset['distance2']=dataset.distance**2
		cosa=dataset.X/dataset.distance
		dataset['alpha']=np.arccos(cosa)
		dataset.X=np.absolute(dataset.X)
		dataset.Y=np.absolute(dataset.Y)
	
	df.to_csv("./input/train.csv",index=False)
	test.to_csv('./input/test.csv',index=False)


from sklearn import ensemble

MODELS={
	"randomforest": ensemble.RandomForestRegressor(n_estimators=200,n_jobs=-1,verbose=2),
	"extratrees": ensemble.ExtraTreesRegressor(n_estimators=200,n_jobs=-1,verbose=2),
}
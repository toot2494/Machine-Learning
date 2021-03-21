from sklearn.datasets import load_boston
from sklearn.model_selection import RepeatedKFold
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
boston = load_boston()
data_boston= pd.DataFrame(boston['data'])
data_boston.columns=boston['feature_names']
data_boston['target']=boston['target']
coef=[]
intercept=[]
mse=[]
r2=[]
kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
for train_index, test_index in kf.split(data_boston.iloc[:,:-1]):
    train_X, train_y = data_boston.iloc[train_index,:-1], data_boston.iloc[train_index,-1]
    test_X, test_y = data_boston.iloc[test_index,:-1], data_boston.iloc[test_index,-1]
    regr = linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)
    regr.fit(train_X, train_y)
    predict_y = regr.predict(test_X)
    print('Coefficients: \n', regr.coef_)
    print('Mean squared error: %.2f'
          % mean_squared_error(test_y, predict_y))
    print('Coefficient of determination: %.2f'
          % r2_score(test_y, predict_y))
    coef.append(regr.coef_)
    intercept.append(regr.intercept_)
    mse.append(mean_squared_error(test_y, predict_y))
    r2.append(r2_score(test_y, predict_y))
    # plt.scatter(test_X, test_y, color='black')
    # plt.plot(test_X, predict_y, color='blue', linewidth=3)
    #
    # plt.xticks(())
    # plt.yticks(())
    #
    # plt.show()
result=pd.DataFrame(coef)
result.columns=boston['feature_names']
result['intercept_']=pd.DataFrame(intercept)
result['mse']=pd.DataFrame(mse)
result['r2']=pd.DataFrame(r2)
result.loc['mean']=result.mean()
result.loc['min']=result.min()
result.loc['max']=result.max()
result.loc['median']=result.median()
result.loc['std']=result.std()
result.loc['var']=result.var()
result.loc['quantile_25%']=result.quantile(0.25)
result.loc['quantile_75%']=result.quantile(0.75)
result.loc['skew']=result.skew()
result.loc['kurt']=result.kurt()
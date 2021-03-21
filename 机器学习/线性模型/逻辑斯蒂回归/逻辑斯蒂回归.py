# print(__doc__)
#
# # Code source: Gaël Varoquaux
# # Modified for documentation by Jaques Grobler
# # License: BSD 3 clause
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
#
# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
# Y = iris.target
#
# # Create an instance of Logistic Regression Classifier and fit the data.
# logreg = LogisticRegression(penalty='l1',intercept_scaling=1,solver='liblinear')
# logreg.fit(X, Y)
#
# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]x[y_min, y_max].
# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# h = .02  # step size in the mesh
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1, figsize=(4, 3))
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
#
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
#
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
#
# plt.show()


from sklearn import datasets
from sklearn.model_selection import RepeatedKFold
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
iris = datasets.load_iris()
data_iris= pd.DataFrame(iris['data'])
data_iris.columns=iris['feature_names']
data_iris['target']=iris['target']
# data_iris=data_iris.loc[0:99]
# for i in range(150,200):
#     data_iris.loc[i]=[i/22,i/14,i/15,i/19,3]
# for i in range(201, 300):
#     data_iris.loc[151]=[i/2,i/3,i/10,i/20,4]
#这里的data,把label放到了最后一列，之前列全是特征：data_iris.iloc[:,:-1]为特征即X；data_iris.iloc[:,-1]为标签即Y

#下面的一些参数说明：n_splits为“k折交叉验证”中的k，也就是将数据集划分成k个集合，每次仅取一个作为测试集，然后重复k次使得每个集合都会做一次测试集；n_repeats表示"n次k折交叉验证"中的n次，也就是把“k折交叉验证”重复n次；random_state为随机数种子，设置成相同的数能够使得每次划分都相同，便于后续调参
coef=[]
intercept=[]
mse=[]
r2=[]
kf = RepeatedKFold(n_splits=2, n_repeats=1, random_state=0)
for train_index, test_index in kf.split(data_iris.iloc[:,:-1]):
    train_X, train_y = data_iris.iloc[train_index,:-1], data_iris.iloc[train_index,-1]
    test_X, test_y = data_iris.iloc[test_index,:-1], data_iris.iloc[test_index,-1]
    logreg = LogisticRegression(penalty='l2',intercept_scaling=1,solver='sag',multi_class='ovr',max_iter=9999,random_state=0,verbose=1)
    logreg.fit(train_X, train_y)
    predict_y = logreg.predict(test_X)
    predict_y2 = logreg.predict_log_proba(test_X)
    predict_y3 = logreg.predict_proba(test_X)

    # print('Coefficients: \n', logreg.coef_)
    # print('Mean squared error: %.2f'
    #       % mean_squared_error(test_y, predict_y))
    # print('Coefficient of determination: %.2f'
    #       % r2_score(test_y, predict_y))
    coef.append(logreg.coef_)
    intercept.append(logreg.intercept_)
    mse.append(mean_squared_error(test_y, predict_y))
    r2.append(r2_score(test_y, predict_y))
result=pd.DataFrame(coef)
result.columns=iris['feature_names']
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
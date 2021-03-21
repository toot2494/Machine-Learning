from sklearn import datasets
from sklearn.model_selection import RepeatedKFold
import pandas as pd
import numpy as np
iris = datasets.load_iris()
data_iris= pd.DataFrame(iris['data'])
data_iris.columns=iris['feature_names']
data_iris['target']=iris['target']
#这里的data,把label放到了最后一列，之前列全是特征：data_iris.iloc[:,:-1]为特征即X；data_iris.iloc[:,-1]为标签即Y
seed=np.random.RandomState(2)
train_index=seed.randint(0, len(data_iris)-1,len(data_iris))#必须先执行一遍seed再执行该条语句才能保证种子不变
test_index=list(set(data_iris.index).difference(set(train_index)))
train_X, train_y = data_iris.iloc[train_index,:-1], data_iris.iloc[train_index,-1]
test_X, test_y = data_iris.iloc[test_index, :-1], data_iris.iloc[test_index, -1]
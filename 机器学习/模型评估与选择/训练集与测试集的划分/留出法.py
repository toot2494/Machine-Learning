from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
iris = datasets.load_iris()
data_iris= pd.DataFrame(iris['data'])
data_iris.columns=iris['feature_names']
data_iris['target']=iris['target']
#这里的data,把label放到了最后一列，之前列全是特征：data_iris.iloc[:,:-1]为特征即X；data_iris.iloc[:,-1]为标签即Y
#参数说明：test_size为划分比例，这里为训练集70%，测试集30%；random_state为随机数种子，设置成相同的数能够使得每次划分都相同，便于后续调参
X_train, X_test, y_train, y_test = train_test_split(data_iris.iloc[:,:-1], data_iris.iloc[:,-1], test_size=0.7, random_state=0)

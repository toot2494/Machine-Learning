from sklearn import datasets
from sklearn.model_selection import RepeatedKFold
import pandas as pd
iris = datasets.load_iris()
data_iris= pd.DataFrame(iris['data'])
data_iris.columns=iris['feature_names']
data_iris['target']=iris['target']
#这里的data,把label放到了最后一列，之前列全是特征：data_iris.iloc[:,:-1]为特征即X；data_iris.iloc[:,-1]为标签即Y

#下面的一些参数说明：n_splits为“k折交叉验证”中的k，也就是将数据集划分成k个集合，每次仅取一个作为测试集，然后重复k次使得每个集合都会做一次测试集；n_repeats表示"n次k折交叉验证"中的n次，也就是把“k折交叉验证”重复n次；random_state为随机数种子，设置成相同的数能够使得每次划分都相同，便于后续调参
kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
for train_index, test_index in kf.split(data_iris.iloc[:,:-1]):
    train_X, train_y = data_iris.iloc[train_index,:-1], data_iris.iloc[train_index,-1]
    test_X, test_y = data_iris.iloc[test_index,:-1], data_iris.iloc[test_index,-1]

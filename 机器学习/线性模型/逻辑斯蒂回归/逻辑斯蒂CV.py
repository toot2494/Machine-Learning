from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
iris=pd.read_excel('C:\\Users\\Tao\\Desktop\\机器学习\\机器学习\\input\\dataset_Iris.xlsx',sheet_name=None)
train_X= iris['dataset_for_train'].iloc[:,:-1]
train_y= iris['dataset_for_train'].iloc[:,-1]
predict_X=iris['dataset_for_predict'].copy()
logreg = LogisticRegressionCV(cv=10,penalty='elasticnet', solver='saga',multi_class='multinomial', max_iter=9999,
                            random_state=0, verbose=0,l1_ratios=[0.9,0.2,0.5,0.7,0.1,1])
logreg.fit(train_X,train_y)
iris['dataset_for_predict']['target']=logreg.predict(predict_X)
# logreg.predict_log_proba(predict_X)
# logreg.predict_proba(predict_X)
train_dataset_performance=pd.DataFrame()
train_dataset_performance['precision']=metrics.precision_recall_fscore_support(train_y,logreg.predict(train_X))[0]
train_dataset_performance['recall']=metrics.precision_recall_fscore_support(train_y,logreg.predict(train_X))[1]
train_dataset_performance['f1-score']=metrics.precision_recall_fscore_support(train_y,logreg.predict(train_X))[2]
train_dataset_performance['support']=metrics.precision_recall_fscore_support(train_y,logreg.predict(train_X))[3]
train_dataset_performance.loc['marco avg']=train_dataset_performance.iloc[:,:-1].mean()
train_dataset_performance.loc['marco avg','precision']=metrics.precision_score(train_y,logreg.predict(train_X),average='macro')
train_dataset_performance.loc['marco avg','recall']=metrics.recall_score(train_y,logreg.predict(train_X),average='macro')
train_dataset_performance.loc['marco avg','f1-score']=metrics.f1_score(train_y,logreg.predict(train_X),average='macro')
train_dataset_performance.loc['marco avg','support']=len(train_y)
train_dataset_performance.loc['weighted avg','precision']=metrics.precision_score(train_y,logreg.predict(train_X),average='weighted')
train_dataset_performance.loc['weighted avg','recall']=metrics.recall_score(train_y,logreg.predict(train_X),average='weighted')
train_dataset_performance.loc['weighted avg','f1-score']=metrics.f1_score(train_y,logreg.predict(train_X),average='weighted')
train_dataset_performance.loc['weighted avg','support']=len(train_y)
train_dataset_performance.loc['accuracy','accuracy']=metrics.accuracy_score(train_y,logreg.predict(train_X))
with pd.ExcelWriter('C:\\Users\\Tao\\Desktop\\机器学习\\机器学习\\output\\Iris_LogisticRegressionCV_result.xlsx') as writer:
    iris['dataset_for_predict'].to_excel(writer,sheet_name='predict_result',index=False)
    train_dataset_performance.to_excel(writer,sheet_name='train_dataset_performance')
writer.save()
writer.close()
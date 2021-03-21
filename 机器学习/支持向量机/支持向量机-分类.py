from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
data_iris=pd.read_excel('C:\\Users\\Tao\\Desktop\\机器学习\\机器学习\\input\\dataset_Iris.xlsx',sheet_name=None)

#设置调参的选择，因为是用的是网格搜索
parameter_C=[0.1,1,10] #all
parameter_kernel=['rbf','poly','sigmoid','linear'] #sigmoid不是很懂
parameter_degree=[1,2,3,4,5,6,7,8,9,10] #only poly
parameter_gamma=['scale','auto']  # 'rbf', 'poly', 'sigmoid'
parameter_coef=[0,1] #'poly', 'sigmoid'

grid_search_result={}
kf = RepeatedKFold(n_splits=3, n_repeats=1, random_state=0)
for train_index, test_index in kf.split(data_iris['dataset_for_train'].iloc[:,:-1]):
    train_X, train_y = data_iris['dataset_for_train'].iloc[train_index,:-1], data_iris['dataset_for_train'].iloc[train_index,-1]
    test_X, test_y = data_iris['dataset_for_train'].iloc[test_index,:-1], data_iris['dataset_for_train'].iloc[test_index,-1]
    for C_chosen in parameter_C:
        for kernel_chosen in parameter_kernel:
            if kernel_chosen=='rbf':
                for gamma_chosen in parameter_gamma:
                    svc=SVC(C=C_chosen,kernel=kernel_chosen,gamma=gamma_chosen,shrinking=False,cache_size=4096,class_weight='balanced',verbose=0,decision_function_shape='ovo')
                    svc.fit(train_X,train_y)
                    temp_str=str(C_chosen)+','+kernel_chosen+','+gamma_chosen
                    grid_search_result.setdefault(temp_str,[]).append(metrics.accuracy_score(test_y,svc.predict(test_X)))
                    svc=None
            if kernel_chosen=='poly':
                for gamma_chosen in parameter_gamma:
                    for coef_chosen in parameter_coef:
                        for degree_chosen in parameter_degree:
                            svc = SVC(C=C_chosen, kernel=kernel_chosen, gamma=gamma_chosen,coef0=coef_chosen,degree=degree_chosen,shrinking=False,
                                      cache_size=4096, class_weight='balanced', verbose=0,
                                      decision_function_shape='ovo')
                            svc.fit(train_X, train_y)
                            temp_str = str(C_chosen) + ',' + kernel_chosen + ',' + gamma_chosen+','+str(coef_chosen)+','+str(degree_chosen)
                            grid_search_result.setdefault(temp_str, []).append(
                                metrics.accuracy_score(test_y, svc.predict(test_X)))
                            svc = None
            if kernel_chosen=='sigmoid':
                for gamma_chosen in parameter_gamma:
                    for coef_chosen in parameter_coef:
                        svc = SVC(C=C_chosen, kernel=kernel_chosen, gamma=gamma_chosen,coef0=coef_chosen,shrinking=False,
                                  cache_size=4096, class_weight='balanced', verbose=0,
                                  decision_function_shape='ovo')
                        svc.fit(train_X, train_y)
                        temp_str = str(C_chosen) + ',' + kernel_chosen + ',' + gamma_chosen+','+str(coef_chosen)
                        grid_search_result.setdefault(temp_str, []).append(
                            metrics.accuracy_score(test_y, svc.predict(test_X)))
                        svc = None
            if kernel_chosen=='linear':
                svc = SVC(C=C_chosen, kernel=kernel_chosen, shrinking=False,
                          cache_size=4096, class_weight='balanced', verbose=0,
                          decision_function_shape='ovo')
                svc.fit(train_X, train_y)
                temp_str = str(C_chosen) + ',' + kernel_chosen
                grid_search_result.setdefault(temp_str, []).append(
                    metrics.accuracy_score(test_y, svc.predict(test_X)))
                svc = None
grid_search_result_excel=pd.DataFrame.from_dict(grid_search_result,orient='index')
grid_search_result_excel['mean']=grid_search_result_excel.mean(axis=1)
grid_search_result_excel.to_excel('C:\\Users\\Tao\\Desktop\\机器学习\\机器学习\\output\\grid_search_result_for_SCV.xlsx')
parameter_chosen=grid_search_result_excel[grid_search_result_excel['mean']==grid_search_result_excel['mean'].max()].index.values

train_X= data_iris['dataset_for_train'].iloc[:,:-1]
train_y= data_iris['dataset_for_train'].iloc[:,-1]
predict_X=data_iris['dataset_for_predict'].copy()
for parameter in parameter_chosen:
    temp_parameter=parameter.split(',')
    if 'rbf' in temp_parameter:
        svc = SVC(C=float(temp_parameter[0]), kernel=temp_parameter[1], gamma=temp_parameter[2], shrinking=False, cache_size=4096,
                  class_weight='balanced', verbose=0, decision_function_shape='ovo')
    if 'poly' in temp_parameter:
        svc = SVC(C=float(temp_parameter[0]), kernel=temp_parameter[1], gamma=temp_parameter[2], coef0=float(temp_parameter[3]), degree=int(temp_parameter[4]),
                  shrinking=False,
                  cache_size=4096, class_weight='balanced', verbose=0,
                  decision_function_shape='ovo')
    if 'sigmoid' in temp_parameter:
        svc = SVC(C=float(temp_parameter[0]), kernel=temp_parameter[1], gamma=temp_parameter[2], coef0=float(temp_parameter[3]),
                  shrinking=False,
                  cache_size=4096, class_weight='balanced', verbose=0,
                  decision_function_shape='ovo')
    if 'linear' in temp_parameter:
        svc = SVC(C=float(temp_parameter[0]), kernel=temp_parameter[1],
                  shrinking=False,
                  cache_size=4096, class_weight='balanced', verbose=0,
                  decision_function_shape='ovo')
    svc.fit(train_X,train_y)
    data_iris['dataset_for_predict']['target'] = svc.predict(predict_X)
    train_dataset_performance = pd.DataFrame()
    train_X_predict_y=svc.predict(train_X)
    train_dataset_performance['precision'] = metrics.precision_recall_fscore_support(train_y, train_X_predict_y)[
        0]
    train_dataset_performance['recall'] = metrics.precision_recall_fscore_support(train_y, train_X_predict_y)[1]
    train_dataset_performance['f1-score'] = metrics.precision_recall_fscore_support(train_y, train_X_predict_y)[2]
    train_dataset_performance['support'] = metrics.precision_recall_fscore_support(train_y, train_X_predict_y)[3]
    train_dataset_performance.loc['marco avg'] = train_dataset_performance.iloc[:, :-1].mean()
    train_dataset_performance.loc['marco avg', 'precision'] = metrics.precision_score(train_y, train_X_predict_y,
                                                                                      average='macro')
    train_dataset_performance.loc['marco avg', 'recall'] = metrics.recall_score(train_y, train_X_predict_y,
                                                                                average='macro')
    train_dataset_performance.loc['marco avg', 'f1-score'] = metrics.f1_score(train_y, train_X_predict_y,
                                                                              average='macro')
    train_dataset_performance.loc['marco avg', 'support'] = len(train_y)
    train_dataset_performance.loc['weighted avg', 'precision'] = metrics.precision_score(train_y,
                                                                                         train_X_predict_y,
                                                                                         average='weighted')
    train_dataset_performance.loc['weighted avg', 'recall'] = metrics.recall_score(train_y, train_X_predict_y,
                                                                                   average='weighted')
    train_dataset_performance.loc['weighted avg', 'f1-score'] = metrics.f1_score(train_y, train_X_predict_y,
                                                                                 average='weighted')
    train_dataset_performance.loc['weighted avg', 'support'] = len(train_y)
    train_dataset_performance.loc['accuracy', 'accuracy'] = metrics.accuracy_score(train_y, train_X_predict_y)
    writer_name='C:\\Users\\Tao\\Desktop\\机器学习\\机器学习\\output\\Iris_SVC_result'+'('+parameter+')'+'.xlsx'
    with pd.ExcelWriter(writer_name) as writer:
        data_iris['dataset_for_predict'].to_excel(writer, sheet_name='predict_result', index=False)
        train_dataset_performance.to_excel(writer, sheet_name='train_dataset_performance')
    writer.save()
    writer.close()
grid_search_result_excel.to_excel('C:\\Users\\Tao\\Desktop\\机器学习\\机器学习\\output\\grid_search_result_for_SCV.xlsx')
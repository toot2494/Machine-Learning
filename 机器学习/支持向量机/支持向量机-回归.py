from sklearn.svm import SVR
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
dataset=pd.read_excel('C:\\Users\\Tao\\Desktop\\机器学习\\机器学习\\input\\dataset_Boston.xlsx',sheet_name=None)

#设置调参的选择，因为是用的是网格搜索
parameter_C=[0.1,1,10] #大于0的浮点数
parameter_kernel=['rbf','poly','sigmoid'] #sigmoid不是很懂，'linear'很慢
parameter_degree=[2,5,8] #only poly，大于零的整数
parameter_gamma=['scale']  # 'rbf', 'poly', 'sigmoid', 'poly'的auto跑不出来
parameter_coef=[0,1] #'poly', 'sigmoid'时才设置
parameter_epsilon=[0.1] #大于0的浮点数

grid_search_result={} #mae error, 是svr优化的error
grid_search_result2={} #mape error,非svr优化的error，但是实物中会看重这个误差
kf = RepeatedKFold(n_splits=3, n_repeats=1, random_state=0)
for train_index, test_index in kf.split(dataset['dataset_for_train'].iloc[:,:-1]):
    train_X, train_y = dataset['dataset_for_train'].iloc[train_index,:-1], dataset['dataset_for_train'].iloc[train_index,-1]
    test_X, test_y = dataset['dataset_for_train'].iloc[test_index,:-1], dataset['dataset_for_train'].iloc[test_index,-1]
    for epsilon_chosen in parameter_epsilon:
        for C_chosen in parameter_C:
            for kernel_chosen in parameter_kernel:
                if kernel_chosen=='rbf':
                    for gamma_chosen in parameter_gamma:
                        svr=SVR(C=C_chosen,kernel=kernel_chosen,gamma=gamma_chosen,shrinking=False,cache_size=4096,verbose=0,epsilon=epsilon_chosen)
                        svr.fit(train_X,train_y)
                        temp_str=str(C_chosen)+','+kernel_chosen+','+gamma_chosen+','+str(epsilon_chosen)
                        grid_search_result.setdefault(temp_str,[]).append(metrics.mean_absolute_error(test_y,svr.predict(test_X)))
                        grid_search_result2.setdefault(temp_str, []).append(
                            mean_absolute_percentage_error(test_y, svr.predict(test_X)))
                        svr=None
                        print(temp_str)
                if kernel_chosen=='poly':
                    for gamma_chosen in parameter_gamma:
                        for coef_chosen in parameter_coef:
                            for degree_chosen in parameter_degree:
                                svr = SVR(C=C_chosen, kernel=kernel_chosen, gamma=gamma_chosen,coef0=coef_chosen,degree=degree_chosen,shrinking=False,
                                          cache_size=4096, verbose=0,epsilon=epsilon_chosen)
                                svr.fit(train_X, train_y)
                                temp_str = str(C_chosen) + ',' + kernel_chosen + ',' + gamma_chosen+','+str(coef_chosen)+','+str(degree_chosen)+','+str(epsilon_chosen)
                                grid_search_result.setdefault(temp_str, []).append(
                                    metrics.mean_absolute_error(test_y, svr.predict(test_X)))
                                grid_search_result2.setdefault(temp_str, []).append(
                                    metrics.mean_absolute_percentage_error(test_y, svr.predict(test_X)))
                                svr = None
                                print(temp_str)
                if kernel_chosen=='sigmoid':
                    for gamma_chosen in parameter_gamma:
                        for coef_chosen in parameter_coef:
                            svr = SVR(C=C_chosen, kernel=kernel_chosen, gamma=gamma_chosen,coef0=coef_chosen,shrinking=False,
                                      cache_size=4096,  verbose=0,epsilon=epsilon_chosen)
                            svr.fit(train_X, train_y)
                            temp_str = str(C_chosen) + ',' + kernel_chosen + ',' + gamma_chosen+','+str(coef_chosen)+','+str(epsilon_chosen)
                            grid_search_result.setdefault(temp_str, []).append(
                                metrics.mean_absolute_error(test_y, svr.predict(test_X)))
                            grid_search_result2.setdefault(temp_str, []).append(
                                metrics.mean_absolute_percentage_error(test_y, svr.predict(test_X)))
                            svr = None
                            print(temp_str)
                if kernel_chosen=='linear':
                    svr = SVR(C=C_chosen, kernel=kernel_chosen, shrinking=False,
                              cache_size=4096,  verbose=0,epsilon=epsilon_chosen)
                    svr.fit(train_X, train_y)
                    temp_str = str(C_chosen) + ',' + kernel_chosen+','+str(epsilon_chosen)
                    grid_search_result.setdefault(temp_str, []).append(
                        metrics.mean_absolute_error(test_y, svr.predict(test_X)))
                    grid_search_result2.setdefault(temp_str, []).append(
                        metrics.mean_absolute_percentage_error(test_y, svr.predict(test_X)))
                    svr = None
                    print(temp_str)
grid_search_result_excel=pd.DataFrame.from_dict(grid_search_result,orient='index')
grid_search_result_excel['mean']=grid_search_result_excel.mean(axis=1)
grid_search_result_excel.to_excel('C:\\Users\\Tao\\Desktop\\机器学习\\机器学习\\output\\grid_search_result_for_SVR.xlsx')
parameter_chosen=grid_search_result_excel[grid_search_result_excel['mean']==grid_search_result_excel['mean'].min()].index.values

train_X= dataset['dataset_for_train'].iloc[:,:-1]
train_y= dataset['dataset_for_train'].iloc[:,-1]
predict_X=dataset['dataset_for_predict'].copy()
for parameter in parameter_chosen:
    temp_parameter=parameter.split(',')
    if 'rbf' in temp_parameter:
        svr = SVR(C=float(temp_parameter[0]), kernel=temp_parameter[1], gamma=temp_parameter[2], shrinking=False, cache_size=4096,verbose=0,epsilon=float(temp_parameter[3]))
    if 'poly' in temp_parameter:
        svr = SVR(C=float(temp_parameter[0]), kernel=temp_parameter[1], gamma=temp_parameter[2], coef0=float(temp_parameter[3]), degree=int(temp_parameter[4]),
                  shrinking=False,
                  cache_size=4096,verbose=0,epsilon=float(temp_parameter[5]))
    if 'sigmoid' in temp_parameter:
        svr = SVR(C=float(temp_parameter[0]), kernel=temp_parameter[1], gamma=temp_parameter[2], coef0=float(temp_parameter[3]),
                  shrinking=False,
                  cache_size=4096, verbose=0,epsilon=float(temp_parameter[3]))
    if 'linear' in temp_parameter:
        svr = SVR(C=float(temp_parameter[0]), kernel=temp_parameter[1],
                  shrinking=False,
                  cache_size=4096, verbose=0,epsilon=float(temp_parameter[2]))
    svr.fit(train_X,train_y)
    dataset['dataset_for_predict']['target'] = svr.predict(predict_X)
    train_dataset_performance = pd.DataFrame()
    train_X_predict_y=svr.predict(train_X)
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
    train_dataset_performance.loc['accuracy', 'accuracy'] = metrics.mean_absolute_error(train_y, train_X_predict_y)
    writer_name='C:\\Users\\Tao\\Desktop\\机器学习\\机器学习\\output\\Iris_SVC_result'+'('+parameter+')'+'.xlsx'
    with pd.ExcelWriter(writer_name) as writer:
        dataset['dataset_for_predict'].to_excel(writer, sheet_name='predict_result', index=False)
        train_dataset_performance.to_excel(writer, sheet_name='train_dataset_performance')
    writer.save()
    writer.close()
grid_search_result_excel.to_excel('C:\\Users\\Tao\\Desktop\\机器学习\\机器学习\\output\\grid_search_result_for_SCV.xlsx')
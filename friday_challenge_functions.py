
import pandas as pd
import numpy as np 
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config

from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC


#####################################################################################################################################################

def default():
     """ READ DATA AND SPLIT """
     data_heart = pd.read_csv('heart.csv')

     x_variables = ['age', 'sex', 'cp',	'trtbps',	'chol',	'fbs', 'restecg',	'thalachh',	'exng',	'oldpeak',	'slp',	'caa',	'thall']
     y_variable  = 'output'

     X = data_heart[x_variables]
     y = data_heart[y_variable]

     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)                   #####################################

     """ INTANTIATE THE ALGORITHMS AND USE DEFAULT PARAMETERS.  SET THE RANDOM STATE TO 15 """

     tree_classifiers = {
     "Decision Tree":  DecisionTreeClassifier(),
     "Extra Trees":    ExtraTreesClassifier(),
     "Random Forest":  RandomForestClassifier(),
     "AdaBoost":       AdaBoostClassifier(),
     "Skl GBM":        GradientBoostingClassifier(),
     "XGBoost":        XGBClassifier(),
     "LightGBM":       LGBMClassifier(),
     'LogisticRegr':   LogisticRegression(),
     'SVC' :           SVC()
     }

     results = pd.DataFrame({'Model': [], "Accuracy_Default": [], 'MSE': [], 'MAB': [], " % error": [], 'Time': []})

     for model_name, model in tree_classifiers.items():
          start_time = time.time()
          model.fit(X_train, y_train)
          total_time = time.time() - start_time
               
          pred = model.predict(X_test)
          
          rang = abs(y_train.max()) - abs(y_train.min())

          results = results.append({"Model":          model_name,
                                        "Accuracy_Default": round(model.score(X_test,y_test) * 100, 1),
                                        "MSE":            metrics.mean_squared_error(y_test, pred),
                                        "MAB":            metrics.mean_absolute_error(y_test, pred),
                                        " % error":       metrics.mean_squared_error(y_test, pred) / rang,
                                        "Time":           total_time},
                                        ignore_index=True)


     results.sort_values(by=['Accuracy_Default'], ascending=False, ignore_index=True)
     results.index  = results.index + 1

     return results

#####################################################################################################################################################


def grid_search():
     data_heart = pd.read_csv('heart.csv')

     x_variables = ['age', 'sex', 'cp',	'trtbps',	'chol',	'fbs', 'restecg',	'thalachh',	'exng',	'oldpeak',	'slp',	'caa',	'thall']
     y_variable  = 'output'

     X = data_heart[x_variables]
     y = data_heart[y_variable]

     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)                    #####################################


     param_DecisionTree  = {'max_depth'   : [3, 4, 5, 6, 7, 8, 9]}
     param_ExtraTree     = {'n_estimators': [0.1, 1, 10, 100, 100, 1000],     'max_depth'     : [3, 4, 5, 6, 7, 8, 9]}
     param_RandomForest  = {'n_estimators': [0.1, 1, 10, 100, 100, 1000],     'max_depth'     : [3, 4, 5, 6, 7, 8, 9]}
     param_AdaBoost      = {'n_estimators': [0.1, 1, 10, 100, 100, 1000],     'learning_rate' : [0.01, 0.1, 1, 10, 100, 1000]}
     param_SklGBM        = {'n_estimators': [0.1, 1, 10, 100, 100, 1000],     'learning_rate' : [0.01, 0.1, 1, 10, 100, 1000]}
     param_XGBoost       = {'booster'     : ['gbtree', 'gblinear'],           'eta'           : [0.01, 0.1, 1, 10, 100, 1000] }
     param_LightGBM      = {'n_estimators': [0.01, 0.1, 1, 10, 100, 1000],    'learning_rate' : [0.01, 0.1, 1, 10, 100, 1000]}
     param_LogisticRegr  = {'C'           : [0.01, 0.1, 1, 10, 100, 1000],    'max_iter'      : [10, 50, 100, 150, 200, 1000]}
     param_SVC           = {'kernel'      : ['linear', 'rbf'],                  'C'           : [0.01, 0.1, 1, 10, 100, 1000]}          


     grid_search_DecisionTree      = GridSearchCV(DecisionTreeClassifier(), param_DecisionTree, cv=5)
     grid_search_ExtraTree         = GridSearchCV(ExtraTreesClassifier(), param_ExtraTree, cv=5)
     grid_search_RandomForest      = GridSearchCV(RandomForestClassifier(), param_RandomForest, cv=5)
     grid_search_AdaBoost          = GridSearchCV(AdaBoostClassifier(), param_AdaBoost, cv=5)
     grid_search_SklGBM            = GridSearchCV(GradientBoostingClassifier(), param_SklGBM, cv=5)
     grid_search_XGBoost           = GridSearchCV(XGBClassifier(), param_XGBoost, cv=5)
     grid_search_LightGBM          = GridSearchCV(LGBMClassifier(), param_LightGBM, cv=5)
     grid_search_LogisticRegr      = GridSearchCV(LogisticRegression(), param_LogisticRegr, cv=5)
     grid_search_SVC               = GridSearchCV(SVC(), param_SVC, cv=5)


     grid_search_DecisionTree.fit(X_train, y_train)
     grid_search_ExtraTree.fit(X_train, y_train)
     grid_search_RandomForest.fit(X_train, y_train)
     grid_search_AdaBoost.fit(X_train, y_train)
     grid_search_SklGBM.fit(X_train, y_train)
     grid_search_XGBoost.fit(X_train, y_train)
     grid_search_LightGBM.fit(X_train, y_train)
     grid_search_LogisticRegr.fit(X_train, y_train)
     grid_search_SVC.fit(X_train, y_train)


     print(f'Decision Tree    =>    Best Score : {round(grid_search_DecisionTree.best_score_ * 100, 2)}    Best Parameters :   {grid_search_DecisionTree.best_params_}  ')
     print(f'Extra Trees      =>    Best Score : {round(grid_search_ExtraTree.best_score_ * 100, 2)}    Best Parameters :   {grid_search_ExtraTree.best_params_}  ')
     print(f'Random Forest    =>    Best Score : {round(grid_search_RandomForest.best_score_ * 100, 2)}    Best Parameters :   {grid_search_RandomForest.best_params_}  ')
     print(f'AdaBoost         =>    Best Score : {round(grid_search_AdaBoost.best_score_ * 100, 2)}    Best Parameters :   {grid_search_AdaBoost.best_params_}  ')
     print(f'Skl GBM          =>    Best Score : {round(grid_search_SklGBM.best_score_ * 100, 2)}    Best Parameters :   {grid_search_SklGBM.best_params_}  ')
     print(f'XGBoost          =>    Best Score : {round(grid_search_XGBoost.best_score_ * 100, 2)}    Best Parameters :   {grid_search_XGBoost.best_params_}  ')
     print(f'LightGBM         =>    Best Score : {round(grid_search_LightGBM.best_score_ * 100, 2)}    Best Parameters :   {grid_search_LightGBM.best_params_}  ')
     print(f'LogisticRegr     =>    Best Score : {round(grid_search_LogisticRegr.best_score_ * 100, 2)}    Best Parameters :   {grid_search_LogisticRegr.best_params_}  ')
     print(f'SVC              =>    Best Score : {round(grid_search_SVC.best_score_ * 100, 2)}    Best Parameters :   {grid_search_SVC.best_params_}  ')

     return


#####################################################################################################################################################


def hyper_parameter_setting():
     """ READ DATA AND SPLIT """

     data_heart = pd.read_csv('heart.csv')

     x_variables = ['age', 'sex', 'cp',	'trtbps',	'chol',	'fbs', 'restecg',	'thalachh',	'exng',	'oldpeak',	'slp',	'caa',	'thall']
     y_variable  = 'output'

     X = data_heart[x_variables]
     y = data_heart[y_variable]

     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)                    #####################################


     """ INTANTIATE THE ALGORITHMS AND USE DEFAULT PARAMETERS.  SET THE RANDOM STATE TO 15 """

     tree_classifiers = {
   
     
     "Decision Tree":  DecisionTreeClassifier(max_depth=5),
     "Extra Trees":    ExtraTreesClassifier(max_depth=5, n_estimators=100),
     "Random Forest":  RandomForestClassifier(max_depth=5, n_estimators=100),
     "AdaBoost":       AdaBoostClassifier(learning_rate=0.1, n_estimators=1000),
     "Skl GBM":        GradientBoostingClassifier(learning_rate=1, n_estimators=1000),
     "XGBoost":        XGBClassifier(booster='gblinear', eta=0.1),
     "LightGBM":       LGBMClassifier(learning_rate=1, n_estimators=10),
     'LogisticRegr':   LogisticRegression(C=0.1, max_iter=100),
     'SVC' :           SVC(C=0.1, kernel='linear')
     }

     results = pd.DataFrame({'Model': [], "Accuracy_HyperParameter": [], 'MSE': [], 'MAB': [], " % error": [], 'Time': []})

     for model_name, model in tree_classifiers.items():
     
          start_time = time.time()
          model.fit(X_train, y_train)
          total_time = time.time() - start_time
               
          pred = model.predict(X_test)
          
          rang = abs(y_train.max()) - abs(y_train.min())

          results = results.append({"Model":          model_name,
                                        "Accuracy_HyperParameter": round(model.score(X_test,y_test) * 100, 1),
                                        "MSE":            metrics.mean_squared_error(y_test, pred),
                                        "MAB":            metrics.mean_absolute_error(y_test, pred),
                                        " % error":       metrics.mean_squared_error(y_test, pred) / rang,
                                        "Time":           total_time},
                                        ignore_index=True)


     results.sort_values(by=['Accuracy_HyperParameter'], ascending=False, ignore_index=True)
     results.index  = results.index + 1

     return results

#####################################################################################################################################################


def enhanced():
     """ READ DATA AND SPLIT """
     data_heart = pd.read_csv('heart.csv')

     X_original = data_heart.drop('output', axis=1)
     y_original = data_heart['output']
     
     X_train, X_test, y_train, y_test = train_test_split(X_original, y_original, random_state=0, stratify=y_original)  #####################################

     def data_enhancement(data):
          gen_data = data
          
          trtbps_std = data['trtbps'].std() / data['trtbps'].mean()
          chol_std = data['chol'].std() / data['chol'].mean()
          thalachh_std = data['thalachh'].std() / data['thalachh'].mean()
          oldpeak_std = data['oldpeak'].std() / data['oldpeak'].mean()
          
          if np.random.randint(2) == 1:
               gen_data['trtbps'] += trtbps_std
          else:
               gen_data['trtbps'] -= trtbps_std
               
          if np.random.randint(2) == 1:
               gen_data['chol'] += chol_std
          else:
               gen_data['chol'] -= chol_std
               
          if np.random.randint(2) == 1:
               gen_data['thalachh'] += thalachh_std
          else:
               gen_data['thalachh'] -= thalachh_std
               
          if np.random.randint(2) == 1:
               gen_data['oldpeak'] += oldpeak_std
          else:
               gen_data['oldpeak'] -= oldpeak_std
          return gen_data

     data_enhanced = data_enhancement(data_heart)


     extra_data = data_enhanced.sample(data_enhanced.shape[0] // 4)

     X_extra = extra_data.drop('output', axis=1)
     y_extra = extra_data['output']

     X_train = pd.concat([X_train, X_extra], axis=0 ) 
     y_train = pd.concat([y_train, y_extra], axis=0)


     """ INTANTIATE THE ALGORITHMS AND USE DEFAULT PARAMETERS.  SET THE RANDOM STATE TO 15 """

     tree_classifiers = {
     "Decision Tree":  DecisionTreeClassifier(max_depth=5),
     "Extra Trees":    ExtraTreesClassifier(max_depth=5, n_estimators=100),
     "Random Forest":  RandomForestClassifier(max_depth=5, n_estimators=100),
     "AdaBoost":       AdaBoostClassifier(learning_rate=0.1, n_estimators=1000),
     "Skl GBM":        GradientBoostingClassifier(learning_rate=1, n_estimators=1000),
     "XGBoost":        XGBClassifier(booster='gblinear', eta=0.1),
     "LightGBM":       LGBMClassifier(learning_rate=1, n_estimators=10),
     'LogisticRegr':   LogisticRegression(C=0.1, max_iter=100),
     'SVC' :           SVC(C=0.1, kernel='linear')

     }


     results = pd.DataFrame({'Model': [], "Accuracy_Enhanced": [], 'MSE': [], 'MAB': [], " % error": [], 'Time': []})

     for model_name, model in tree_classifiers.items():
     
          start_time = time.time()
          model.fit(X_train, y_train)
          total_time = time.time() - start_time
               
          pred = model.predict(X_test)
          
          rang = abs(y_train.max()) - abs(y_train.min())

          results = results.append({"Model":          model_name,
                                        "Accuracy_Enhanced": round(model.score(X_test,y_test) * 100, 1),
                                        "MSE":            metrics.mean_squared_error(y_test, pred),
                                        "MAB":            metrics.mean_absolute_error(y_test, pred),
                                        " % error":       metrics.mean_squared_error(y_test, pred) / rang,
                                        "Time":           total_time},
                                        ignore_index=True)


     results.sort_values(by=['Accuracy_Enhanced'], ascending=False, ignore_index=True)
     results.index  = results.index + 1

     return results

#####################################################################################################################################################



# data1 = default()
# data2 = hyper_parameter_setting()
# data3 = enhanced()


# print(data1)
# print(data2)
# print(data3)
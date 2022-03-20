
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


def default():
     """ READ DATA AND SPLIT """
     data_heart = pd.read_csv('heart.csv')

     x_variables = ['age', 'sex', 'cp',	'trtbps',	'chol',	'fbs', 'restecg',	'thalachh',	'exng',	'oldpeak',	'slp',	'caa',	'thall']
     y_variable  = 'output'

     X = data_heart[x_variables]
     y = data_heart[y_variable]

     X_train, X_test, y_train, y_test = train_test_split(X, y)

     """ INTANTIATE THE ALGORITHMS AND USE DEFAULT PARAMETERS.  SET THE RANDOM STATE TO 15 """

     tree_classifiers = {
     "Decision Tree":  DecisionTreeClassifier(random_state=15),
     "Extra Trees":    ExtraTreesClassifier(random_state=15),
     "Random Forest":  RandomForestClassifier(random_state=15),
     "AdaBoost":       AdaBoostClassifier(random_state=15),
     "Skl GBM":        GradientBoostingClassifier(random_state=15),
     "XGBoost":        XGBClassifier(random_state=15),
     "LightGBM":       LGBMClassifier(random_state=15),
     'LogisticRegr':   LogisticRegression(random_state=15),
     'SVC' :           SVC(random_state=15)
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



def hyper_parameter_setting():
     """ READ DATA AND SPLIT """

     data_heart = pd.read_csv('heart.csv')

     x_variables = ['age', 'sex', 'cp',	'trtbps',	'chol',	'fbs', 'restecg',	'thalachh',	'exng',	'oldpeak',	'slp',	'caa',	'thall']
     y_variable  = 'output'

     X = data_heart[x_variables]
     y = data_heart[y_variable]

     X_train, X_test, y_train, y_test = train_test_split(X, y)


     """ INTANTIATE THE ALGORITHMS AND USE DEFAULT PARAMETERS.  SET THE RANDOM STATE TO 15 """

     tree_classifiers = {
     
     "Decision Tree":  DecisionTreeClassifier(max_depth=3, random_state=15),
     "Extra Trees":    ExtraTreesClassifier(max_depth=3, n_estimators=10, random_state=15),
     "Random Forest":  RandomForestClassifier(max_depth=4, n_estimators=100, random_state=15),
     "AdaBoost":       AdaBoostClassifier(learning_rate=0.01, n_estimators=1000, random_state=15),
     "Skl GBM":        GradientBoostingClassifier(learning_rate=1, n_estimators=1000, random_state=15),
     "XGBoost":        XGBClassifier(booster='gblinear', eta=1, random_state=15),
     "LightGBM":       LGBMClassifier(learning_rate=1, n_estimators=10, random_state=15),
     'LogisticRegr':   LogisticRegression(C=0.1, max_iter=200, random_state=15),
     'SVC' :           SVC(C=100, kernel='linear', random_state=15)
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



def enhanced():
     """ READ DATA AND SPLIT """
     data_heart = pd.read_csv('heart.csv')

     X_original = data_heart.drop('output', axis=1)
     y_original = data_heart['output']
     
     X_train, X_test, y_train, y_test = train_test_split(X_original, y_original)

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
     
     "Decision Tree":  DecisionTreeClassifier(max_depth=3, random_state=15),
     "Extra Trees":    ExtraTreesClassifier(max_depth=3, n_estimators=10, random_state=15),
     "Random Forest":  RandomForestClassifier(max_depth=4, n_estimators=100, random_state=15),
     "AdaBoost":       AdaBoostClassifier(learning_rate=0.01, n_estimators=1000, random_state=15),
     "Skl GBM":        GradientBoostingClassifier(learning_rate=1, n_estimators=1000, random_state=15),
     "XGBoost":        XGBClassifier(booster='gblinear', eta=1, random_state=15),
     "LightGBM":       LGBMClassifier(learning_rate=1, n_estimators=10, random_state=15),
     'LogisticRegr':   LogisticRegression(C=0.1, max_iter=200, random_state=15),
     'SVC' :           SVC(C=100, kernel='linear', random_state=15)
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

data1 = default()
data2 = hyper_parameter_setting()
data3 = enhanced()

print(data1)
print(data2)
print(data3)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('heart.csv')

#df.head()
#df.info()
#df.describe()

#fig, ax = plt.subplots(figsize=(12,8)) 
#sns.heatmap(df.corr(), annot=True)

# Preprocessing
from sklearn import preprocessing 
from sklearn import impute
from sklearn import pipeline 
np.random.seed(0)

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

df2 = data_enhancement(df)
df2
df.head(5)

from sklearn.preprocessing import StandardScaler
scaler_Models = pipeline.Pipeline(steps=[('scaling' , StandardScaler())])

x = df.drop('output', axis=1)
y = df.output

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
x_train.shape

extra_sample = df2.sample(df2.shape[0] // 4)
x_train = pd.concat([x_train, extra_sample.drop(['output'], axis=1 ) ])
y_train = pd.concat([y_train, extra_sample['output'] ])
x_train.shape

from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier

classifiers = {
  "Decision Tree": DecisionTreeClassifier(),
  "Extra Trees":   ExtraTreesClassifier(n_estimators=100),
  "Random Forest": RandomForestClassifier(n_estimators=100),
  "AdaBoost":      AdaBoostClassifier(n_estimators=100),
  "Skl GBM":       GradientBoostingClassifier(n_estimators=100),
  #"XGBoost":       XGBClassifier(n_estimators=100),
  "LightGBM":      LGBMClassifier(n_estimators=100),
  "LogReg":        LogisticRegression(),
  "SVC":           SVC(),
  "KNN":           KNeighborsClassifier(n_neighbors=2)
}
classifiers = {name: pipeline.make_pipeline(scaler_Models, model) for name, model in classifiers.items()}

import time
from sklearn import metrics

results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], " % error": [], 'Time': []})

rang = abs(y_train.max()) - abs(y_train.min())

for model_name, model in classifiers.items():
    
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_test)
    
    results = results.append({"Model":    model_name,
                              "MSE": metrics.mean_squared_error(y_test, pred),
                              "MAB": metrics.mean_absolute_error(y_test, pred),
                              " % error": metrics.mean_squared_error(y_test, pred) / rang,
                              "Accuracy Score": model.score(x_test,y_test),
                              "Time":     total_time},
                              ignore_index=True)


results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['MSE', 'MAB'], vmin=0, vmax=100, color='#5fba7d')


















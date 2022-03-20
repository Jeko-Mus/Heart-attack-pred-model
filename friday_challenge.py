import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import friday_challenge_functions as fcf


# grid = fcf.grid_search()

data_default = fcf.default()
data_hyper_parameters = fcf.hyper_parameter_setting()
data_enhanced = fcf.enhanced()


data_default.to_csv('accuracy_default_parameters')
data_hyper_parameters.to_csv('accuracy_hyper_parameters')
data_enhanced.to_csv('accuracy_data_enhanced')


data1 = pd.read_csv('accuracy_default_parameters')
data2 = pd.read_csv('accuracy_hyper_parameters')
data3 = pd.read_csv('accuracy_data_enhanced')

# print(data1)
# print(data2)
# print(data3)


accuracy = pd.concat([data1[['Model',  'Accuracy_Default']], data2['Accuracy_HyperParameter'], data3['Accuracy_Enhanced']], axis=1)

data1 = accuracy['Accuracy_Default']
data2 = accuracy['Accuracy_HyperParameter']
data3 = accuracy['Accuracy_Enhanced']


X = np.arange(9)

# fig = plt.figure(figsize=(10, 8))

plt.figure(figsize=(14, 11))
# ax = fig.add_axes([0,0,1,1])

plt.bar(X + 0.0, data1, color = 'b', width = 0.2)
plt.bar(X + 0.2, data2, color = 'g', width = 0.2)
plt.bar(X + 0.4, data3, color = 'r', width = 0.2)

plt.ylabel('Accuracy', fontsize=20)
plt.title('A graph of the Accuracy a function of the Model', fontsize=20)
plt.ylim(60, 95)
plt.xticks(X, accuracy['Model'], rotation=30, fontsize=15)
plt.yticks(np.arange(60, 95, 5))
plt.legend(labels=['Accuracy_Default', 'Accuracy_HyperParameter', 'Accuracy_Enhanced'], fontsize=16)

plt.show()

# plt.savefig('zor')

# print(accuracy)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import friday_challenge_functions as fcf

# data_default = fcf.default()
# data_hyper_parameters = fcf.hyper_parameter_setting()
# data_enhanced = fcf.enhanced()

# data_default.to_csv('accuracy_default_parameters')
# data_hyper_parameters.to_csv('accuracy_hyper_parameters')
# data_enhanced.to_csv('accuracy_data_enhanced')


data1 = pd.read_csv('accuracy_default_parameters')
data2 = pd.read_csv('accuracy_hyper_parameters')
data3 = pd.read_csv('accuracy_data_enhanced')


accuracy = pd.concat([data1[['Model',  'Accuracy_Default']], data2['Accuracy_HyperParameter'], data3['Accuracy_Enhanced']], axis=1)

data1 = accuracy['Accuracy_Default']
data2 = accuracy['Accuracy_HyperParameter']
data3 = accuracy['Accuracy_Enhanced']


X = np.arange(9)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.0, data1, color = 'b', width = 0.2)
ax.bar(X + 0.2, data2, color = 'g', width = 0.2)
ax.bar(X + 0.4, data3, color = 'r', width = 0.2)

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vrs Model')
ax.set_xticks(X, accuracy['Model'])
ax.set_yticks(np.arange(0, 150, 2))
ax.legend(labels=['Accuracy_Default', 'Accuracy_HyperParameter', 'Accuracy_Enhanced'])


plt.show()
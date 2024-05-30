# # importing package
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # create data
# df = pd.DataFrame([
#     ['Accuracy', 0.99, 0.86, 0.96, 1.00, 0.96],
#     ['Precision', 0.98, 0.79, 0.95, 0.99, 0.94],
#     ['Recall', 1.00, 1.00, 0.98, 1.00, 1.00],
#     ['F1-Score', 0.99, 0.88, 0.96, 1.00, 0.97]],
#     columns=['Metrics', 'RF', 'SVC', 'AdaBoost', 'ET', 'QBoost'])
#
# df1 = pd.DataFrame([
#     ['Accuracy', 0.994178, 0.999799, 0.999799]],
#     columns=['Metrics', 'QBoost', 'Random Forest', 'AdaBoost'])
#
# # view data
# print(df)
#
# # plot grouped bar chart
# df1.plot(x='Metrics',
#          kind='bar',
#          stacked=False)
#         #title='Comparison of Quality Metrics for ML Algorithms')
#
# plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

qboost_accuracy = 0.994178
random_forest_accuracy = 0.999799
adaboost_accuracy = 0.999799

data = [qboost_accuracy, random_forest_accuracy, adaboost_accuracy]

labels = ['QBoost', 'Random Forest', 'AdaBoost']

ax = sns.barplot(x=labels, y=data)

plt.ylim(0.0, 1.18)

plt.ylabel('Accuracy')

for i, v in enumerate(data):
    ax.annotate(str(v), xy=(i, v), ha='center', va='bottom')


plt.show()




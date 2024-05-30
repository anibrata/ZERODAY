import matplotlib.pyplot as plt
import numpy as np

metrics = ("Accuracy", "Precision", "Recall", "F1-Score")
# PCA n=2
# penguin_means = {
#     'QBoost': (0.81, 0.86, 0.81, 0.80),
#     'PegaSoS QSVC': (0.70, 0.81, 0.70, 0.67),
#     'VQC': (0.61, 0.78, 0.61, 0.54)
# }

#PCA n=4
# penguin_means = {
#    #  'QBoost': (0.81, 0.86, 0.81, 0.80),
#     'PegaSoS QSVC': (0.77, 0.83, 0.77, 0.76),
#     'VQC': (0.74, 0.82, 0.74, 0.73)
# }

# PCA n=6
# penguin_means = {
#     'QBoost': (0.80, 0.86, 0.80, 0.79),
#     'PegaSoS QSVC': (0.76, 0.84, 0.76, 0.74),
#     'VQC': (0.67, 0.68, 0.67, 0.67)
# }

# PCA n=8
penguin_means = {
    'QBoost': (0.96, 0.96, 0.97, 0.96),
    #'PegaSoS QSVC': (0.76, 0.84, 0.76, 0.74),
    'Random Forest': (0.99, 0.99, 0.99, 0.99)
}

#Automotive AI
# penguin_means = {
#     'QBoost': (0.994, 0.997, 0.964, 0.979),
#     'Random Forest': (0.999, 0.999, 0.999, 0.999),
#     'AdaBoost': (0.999, 0.999, 0.999, 0.999)
# }

#metrics = ("Average Training Time", "Average Prediction Time")
# penguin_means = {
#     'SVC': (434.51, 37.72),
#     'RF': (92.44, 1.32),
#     'QBoost': (24.62, 0.068),
#     'AdaBoost':	(10.14, 0.25),
#     'ET': (8.37, 0.50)
# }
#penguin_means = {
#    'QBoost': (24.62, 0.068),
#    'VQC': (20406.25, 179.67)
#}
#penguin_means = {
#    'QBoost': (338.26, 7.15),
#    'Random Forest': (390.25, 12.06),
#    'AdaBoost': (12.96, 0.27)
#}

# PCA 2
#penguin_means = {
#    'QBoost': (13.24, 0.097),
#    'PegaSoS QSVC': (2.65, 5405.17),
#    'VQC': (14075.14, 194.18)
#}

# PCA 4
#penguin_means = {
#    'QBoost': (13.79, 0.095),
#    'PegaSoS QSVC': (4.27, 9347.38),
#    'VQC': (34478.57, 460.9)
#}

# PCA 6
#penguin_means = {
#    'QBoost': (16.47, 0.11),
#    'PegaSoS QSVC': (6.78, 13781.99),
#    'VQC': (62182.18, 859.68)
#}

# PCA 6
#penguin_means = {
#    'QBoost': (19.66, 0.12),
#    'PegaSoS QSVC': (10.71, 22589.22),
#    'VQC': (110296.37, 1575.01)
#}

x = np.arange(len(metrics))  # the label locations
width = 0.3  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=5, rotation=90)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Time(s)')
# ax.set_xlabel('Time Performance', weight='bold', fontsize=10)
ax.set_xlabel('', weight='bold', fontsize=15)
#ax.set_title('Quality metrics based comparison of Classifiers (5% Dataset)')
#ax.set_title('Time Performance based comparison of Classifiers (5% Dataset)')
ax.set_xticks(x + width, metrics)
ax.legend(loc='upper right', ncols=5)
# ax.text(0.025, 0.975, 'PCA with n=8',
#        horizontalalignment='left',
#        verticalalignment='top',
#        transform=ax.transAxes, weight='bold')

ax.set_ylim(0, 1.3)

plt.show()

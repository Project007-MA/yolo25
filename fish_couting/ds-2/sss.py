import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times new Roman'
plt.rcParams['font.size'] = 23
plt.rcParams['font.weight'] = 'bold'
# Confusion matrix elements
TP, FN, FP, TN = 8316, 12, 21, 5616

# Construct the confusion matrix
confusion_matrix = np.array([[TP, FN], 
                             [FP, TN]])

# Labels for "Detection" and "Non-Detection"
labels = ['Detection', 'Non-Detection']

# Display the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)

# plt.title('Confusion Matrix: Detection or Non-Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

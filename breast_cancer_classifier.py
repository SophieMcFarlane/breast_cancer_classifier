## Using K-Nearest Neighbor on breast cancer data to find the best K value based on the validation accuracies

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

## Loading in the data
breast_cancer_data = load_breast_cancer()

## Creating the training and validation sets
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

## making sure the training and validation sets are the same length
print(len(training_data))
print(len(training_labels))

## Getting a list of k's from 1 to 100 and their corresponding scores
k_list = []
accuracies = []
for k in range (1, 101): 
  k_list.append(k)
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data,   validation_labels))
  
## plotting k values against scores
plt.scatter(k_list, accuracies)
plt.xlabel('K')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()

# let's load the dataset and check its shape and head:

import pandas as pd

url = 'https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/HR_comma_sep.csv'
df = pd.read_csv(url)

print(df.shape)
print(df.head())

# check the data types and missing values:

print(df.dtypes)
print(df.isnull().sum())


# summary statistics:

print(df.describe())

# the categorical columns, starting with the department:

print(df['Department'].value_counts())

# salary column:

print(df['salary'].value_counts())




# One-hot encoding for department and salary columns
df = pd.get_dummies(df, columns=['Department', 'salary'])

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X = df.drop('left', axis=1)
y = df['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train a logistic regression model on the training data
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = lr.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# analyze our logistic regression model by looking at the confusion matrix and classification report:

from sklearn.metrics import confusion_matrix, classification_report

# Print the confusion matrix and classification report
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))


# ?graph

from sklearn.metrics import roc_curve, auc

# Calculate the probabilities and fpr/tpr for the ROC curve
y_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

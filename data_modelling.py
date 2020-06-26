import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("C:/Users/Ghajaanan Jeyakumara/Desktop/mental health/clean_data.csv",index_col=0)
#print(df.shape)
#df.head(5)

labelDict = {}
for feature in df:
    le = preprocessing.LabelEncoder()
    if feature != 'age':
        le.fit(df[feature].astype(str))
    if feature == 'age':
        le.fit(df[feature].astype(int))
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    if feature != 'age':
        df[feature] = le.transform(df[feature].astype(str))
    if feature == 'age':
        df[feature] = le.transform(df[feature].astype(int))

    labelKey = feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] = labelValue

#scaling age to fit into a scale of 0.0 to 1.0    
scaler = MinMaxScaler()
df['age'] = scaler.fit_transform(df[['age']])


#correlation matrix for features
correlation_matrix = df.corr()
sn.heatmap(correlation_matrix)
plt.tight_layout(rect=[0.95, 0.03, 1, 0.95])
plt.show()




#Correlation with output variable
cor_target = abs(correlation_matrix['treatment'])

#Selecting highly correlated features
#any feature with less than >.1 correlation will be useful for prediction
relevant_features = cor_target[cor_target > 0.1]
print(relevant_features)

#Specify X and Y using only relevant features
input_cols = ['gender', 'family_history', 'work_interfere', 'benefits', 'care_options', 'observed_consequence']
X = df[input_cols]
Y = df.treatment

#training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

#Dictionary for storing accuracy
accDict = {}

#Logistic Regression

#df.treatment.value_counts()
#sn.countplot(x = 'treatment', data = df, palette = 'hls')
#plt.show()
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
#make predictions on test data
Y_pred = logreg.predict(X_test)
#confusion matrix and predict target data
confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
confusion_matrix
#visualising consfusion matrix
fig, ax = plt.subplots()
tick_marks = np.arange(len(input_cols))
plt.xticks(tick_marks, input_cols)
plt.yticks(tick_marks, input_cols)
sn.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="Blues", fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0, 2])
plt.tight_layout(rect=[0.95, 0.03, 1, 0.95])
plt.title('Confusion matrix Logistic Regression', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
print("Logistic Regression")
#confusion matrix evaluation metrics
print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
print("Precision:", metrics.precision_score(Y_test, Y_pred))
print("Recall:", metrics.recall_score(Y_test, Y_pred))
accDict['Logistic Regression'] = (metrics.accuracy_score(Y_test, Y_pred)) *100

#Decision Trees
model = tree.DecisionTreeClassifier(max_depth=3, min_samples_split=8, max_features=6, criterion='entropy', min_samples_leaf=70)
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
#confusion matrix and predict target data
confusion_matrix = metrics.confusion_matrix(Y_test, Y_predict)
#visualising confusion matrix
fig, ax = plt.subplots()
tick_marks = np.arange(len(input_cols))
plt.xticks(tick_marks, input_cols)
plt.yticks(tick_marks, input_cols)
sn.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="viridis" ,fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0, 2])
plt.tight_layout(rect=[0.95, 0.03, 1, 0.95])   #[left,down,right,top]
plt.title('Confusion matrix Decision tree', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
print("Decision Tree")
#confusion matrix evaluation metrics
print("Accuracy:", metrics.accuracy_score(Y_test, Y_predict))
print("Precision:", metrics.precision_score(Y_test, Y_predict))
print("Recall:", metrics.recall_score(Y_test, Y_predict))
accDict['Decision Tree'] = (metrics.accuracy_score(Y_test, Y_predict)) *100

#Random Forest
clf = RandomForestClassifier(max_depth=None, min_samples_leaf=70, min_samples_split=8, n_estimators=30, random_state=1)
clf.fit(X_train, Y_train)
Y_predict = clf.predict(X_test)
#confusion matrix and predict target data
confusion_matrix = metrics.confusion_matrix(Y_test, Y_predict)
#visualising confusion matrix
fig, ax = plt.subplots()
tick_marks = np.arange(len(input_cols))
plt.xticks(tick_marks, input_cols)
plt.yticks(tick_marks, input_cols)
sn.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="Greens" ,fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0, 2])
plt.tight_layout(rect=[0.95, 0.03, 1, 0.95])
plt.title('Confusion matrix Random Forest', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
print("Random Forest")
#confusion matrix evaluation metrics
print("Accuracy:", metrics.accuracy_score(Y_test, Y_predict))
print("Precision:", metrics.precision_score(Y_test, Y_predict))
print("Recall:", metrics.recall_score(Y_test, Y_predict))
accDict['Random Forest'] = (metrics.accuracy_score(Y_test, Y_predict)) *100
#K Nearest Neighbors

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
Y_predict = knn.predict(X_test)
#confusion matrix and predict target data
confusion_matrix = metrics.confusion_matrix(Y_test, Y_predict)
confusion_matrix
#visualising consfusion matrix

fig, ax = plt.subplots()
tick_marks = np.arange(len(input_cols))
plt.xticks(tick_marks, input_cols)
plt.yticks(tick_marks, input_cols)
sn.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0, 2])
plt.tight_layout(rect=[0.95,0.03,1,0.95])
plt.title('Confusion matrix KNN', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
print("K Nearest Neighbors")
#confusion matrix evaluation metrics
print("Accuracy:", metrics.accuracy_score(Y_test, Y_predict))
print("Precision:", metrics.precision_score(Y_test, Y_predict))
print("Recall:", metrics.recall_score(Y_test, Y_predict))
accDict['K Nearest Neighbors'] = (metrics.accuracy_score(Y_test, Y_predict)) *100

#SVM Classifier
svm_clf = svm.SVC(kernel="linear")
svm_clf.fit(X_train, Y_train)
Y_predict = svm_clf.predict(X_test)
#confusion matrix and predict target data
confusion_matrix = metrics.confusion_matrix(Y_test, Y_predict)
confusion_matrix
#visualising consfusion matrix

fig, ax = plt.subplots()
tick_marks = np.arange(len(input_cols))
plt.xticks(tick_marks, input_cols)
plt.yticks(tick_marks, input_cols)
sn.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="BuPu", fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0, 2])
plt.tight_layout(rect=[0.95,0.03,1,0.95])
plt.title('Confusion matrix SVC', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
print("Support Vector Classifier")
#confusion matrix evaluation metrics
print("Accuracy:", metrics.accuracy_score(Y_test, Y_predict))
print("Precision:", metrics.precision_score(Y_test, Y_predict))
print("Recall:", metrics.recall_score(Y_test, Y_predict))
accDict['SVC'] = (metrics.accuracy_score(Y_test, Y_predict)) *100

#Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
Y_predict = gnb.predict(X_test)
#confusion matrix and predict target data
confusion_matrix = metrics.confusion_matrix(Y_test, Y_predict)
confusion_matrix
#visualising consfusion matrix

fig, ax = plt.subplots()
tick_marks = np.arange(len(input_cols))
plt.xticks(tick_marks, input_cols)
plt.yticks(tick_marks, input_cols)
sn.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="Oranges", fmt='g')
ax.xaxis.set_label_position("top")
ax.set_ylim([0, 2])
plt.tight_layout(rect=[0.95,0.03,1,0.95])
plt.title('Confusion matrix Naive Bayes', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
print("Naive Bayes")
#confusion matrix evaluation metrics
print("Accuracy:", metrics.accuracy_score(Y_test, Y_predict))
print("Precision:", metrics.precision_score(Y_test, Y_predict))
print("Recall:", metrics.recall_score(Y_test, Y_predict))
accDict['Naive Bayes'] = (metrics.accuracy_score(Y_test, Y_predict)) *100
#print(accDict)

def plotAccuracy():
    s = pd.Series(accDict)
    s = s.sort_values(ascending=False)
    #Colors
    ax = s.plot(kind='bar')
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(), 2)), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.ylim([70.0, 90.0])
    plt.xlabel('Model')
    plt.ylabel('Percentage')
    plt.title('Accuracy of models')
    plt.show()

plotAccuracy()

classifier = AdaBoostClassifier()
classifier.fit(X, Y)
dfTestPredictions = classifier.predict(X_test)

# Write predictions to csv file
# We don't have any significative field so we save the index
predictions = pd.DataFrame({'Index': X_test.index, 'Treatment': dfTestPredictions})
# Save to file
# This file will be visible after publishing in the output section
predictions.to_csv('C:/Users/Ghajaanan Jeyakumara/Desktop/mental health/predictions.csv', index=False)
print(predictions.head(20))


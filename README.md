<b>OBJECTIVE</b><br>
The main objective of this project is to practice data cleaning, apply classification algorithms and visualise the results <br>
<b>The original data has been obtained from the links below </b> : <br>
https://www.kaggle.com/osmi/mental-health-in-tech-2016     <br>
https://www.kaggle.com/osmi/mental-health-in-tech-survey <br>
<b> Preprocessing </b> <br>
Editted the poorly formatted csv files <br>
Joined the data sets survey_2014 and survey_2016 <br>
Editted the column names from survey questions to easily usable labels <br>
Removed superfluous data by dropping columns that wouldn't be used in modelling <br>
Treated missing values and anomalies <br>
Saved the clean data as a separate csv file for modelling <br>
Performed Label Encoding<br><br>
<b> ML Algorithms</b> <br><br>
Applied 6 Classification algorithms using sklearn library: <br>
<ol>
  <li>Logistic Regression</li><br>
 <li>Decision Trees </li><br>
 <li>Random Forest </li><br>
 <li>K Nearest Neighbors </li><br>
 <li>Support Vector Classifier </li><br>
 <li>Guassian Naive Bayes</li><br>
</ol><br>
Calculated and visualised the confusion matrix <br> <br>
Chose the best model using AdaBoostClassifier and printed the predictions <br><br>
Saved the predictions with the index number as a separate csv file

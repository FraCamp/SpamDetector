# importing dependencies
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

print("/-------------------SpamDetector for e-Mail-------------------/")
df1 = pd.read_csv("email_spam.csv")
df = df1[['label', 'text']]

# Categorize Spam as 1 and Not spam as 0
df.loc[df["label"] == 'ham', "Category"] = 0
df.loc[df["label"] == 'spam', "Category"] = 1

df = df.rename(columns={'text': 'Content'})
dff = df[['Content', 'Category']]

x = dff['Category']
y = dff['Content']

# splitting the original dataset (diveded by columns, x and y) into 4 different dataset
# x_train, x_test and y_train and y_test, using the train_test_split function (sklearn) is possible to decide the
# shuffling of the data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=3)

# feature extraction, conversion to lower case and removal of stop words using TFIDF VECTORIZER

# Finding the term frequency-inverse document frequency (tf-idf, useful into text analysis and in order to use machine
# learning algorithm for Natural Language Processing) by multiplying two metrics: how many times a
# word appears in a document, and the inverse document frequency of the word across a set of documents
tfvec = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

y_trainFeat = tfvec.fit_transform(y_train)
y_testFeat = tfvec.transform(y_test)

# Training and applying classifiers
# SVM
x_trainSvm = x_train.astype('int')
classifierSVM = LinearSVC()
classifierSVM.fit(y_trainFeat, x_trainSvm)
predResMailSVM = classifierSVM.predict(y_testFeat)

# MNB
x_trainGnb = x_train.astype('int')
classifierMNB = MultinomialNB()
classifierMNB.fit(y_trainFeat, x_trainGnb)
predResMailMNB = classifierMNB.predict(y_testFeat)

# KNN
x_trainKNN = x_train.astype('int')
classifierKNN = KNeighborsClassifier(n_neighbors=1)
classifierKNN.fit(y_trainFeat, x_trainKNN)
predResMailKNN = classifierKNN.predict(y_testFeat)

# RF
x_trainRF = x_train.astype('int')
classifierRF = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=None)
classifierRF.fit(y_trainFeat, x_trainRF)
predResMailRF = classifierRF.predict(y_testFeat)

# Adaboost
x_trainAdaB = x_train.astype('int')
classifierAdaB = AdaBoostClassifier(n_estimators=100)
classifierAdaB.fit(y_trainFeat, x_trainAdaB)
predResMailAdaB = classifierAdaB.predict(y_testFeat)

# Converting to int - solves - cant handle mix of unknown and binary
x_test = x_test.astype('int')
actual_Y = x_test.to_numpy()

# Metrics and results
print("\tSupport Vector Machine RESULTS")
# Accuracy score using SVM
print("Accuracy Score using SVM: {0:.4f}".format(accuracy_score(actual_Y, predResMailSVM) * 100))
# FScore MACRO using SVM
print("F Score using SVM: {0: .4f}".format(f1_score(actual_Y, predResMailSVM, average='macro') * 100))
cmSVM = confusion_matrix(actual_Y, predResMailSVM)
# [True negative  False Positive
# False Negative True Positive]
print("Confusion matrix using SVM:")
print(cmSVM)
print()
print("\tMultinomial Näive Bayes RESULTS")
# Accuracy score using MNB
print("Accuracy Score using MNB: {0:.4f}".format(accuracy_score(actual_Y, predResMailMNB) * 100))
# FScore MACRO using MNB
print("F Score using MNB:{0: .4f}".format(f1_score(actual_Y, predResMailMNB, average='macro') * 100))
cmMNb = confusion_matrix(actual_Y, predResMailMNB)
# [True negative  False Positive
# False Negative True Positive]
print("Confusion matrix using MNB:")
print(cmMNb)
print()
print("\tK Nearest Neighbors RESULTS")
print("Neighbors Number: 1")
# Accuracy score using KNN
print("Accuracy Score using KNN: {0:.4f}".format(accuracy_score(actual_Y, predResMailKNN) * 100))
# FScore MACRO using KNN
print("F Score using KNN:{0: .4f}".format(f1_score(actual_Y, predResMailKNN, average='macro') * 100))
cmKNN = confusion_matrix(actual_Y, predResMailKNN)
# [True negative  False Positive
# False Negative True Positive]
print("Confusion matrix using KNN:")
print(cmKNN)
print()
print("\tRandom Forest RESULTS")
# Accuracy score using MNB
print("Accuracy Score using RF: {0:.4f}".format(accuracy_score(actual_Y, predResMailRF) * 100))
# FScore MACRO using MNB
print("F Score using RF:{0: .4f}".format(f1_score(actual_Y, predResMailRF, average='macro') * 100))
cmRF = confusion_matrix(actual_Y, predResMailRF)
# [True negative  False Positive
# False Negative True Positive]
print("Confusion matrix using RF:")
print(cmRF)
print()
print("\tAdaboost RESULTS")
print("Estimators Number: 100")
# Accuracy score using MNB
print("Accuracy Score using AdaB: {0:.4f}".format(accuracy_score(actual_Y, predResMailAdaB) * 100))
# FScore MACRO using MNB
print("F Score using AdaB:{0: .4f}".format(f1_score(actual_Y, predResMailAdaB, average='macro') * 100))
cmAdaB = confusion_matrix(actual_Y, predResMailAdaB)
# [True negative  False Positive
# False Negative True Positive]
print("Confusion matrix using AdaB:")
print(cmAdaB)

print("\n")
print("/---------------------SpamDetector for SMS--------------------/")
# using encoding options in order to open and clean the csv, which has some empty columns
df2 = pd.read_csv("sms_spam.csv", encoding = "ISO-8859-1")
dfsp = df2[['v1', 'v2']]
dfsp.loc[dfsp["v1"] == 'ham', "Category"] = 0
dfsp.loc[dfsp["v1"] == 'spam', "Category"] = 1
dfsp = dfsp.rename(columns={'v2': 'Content'})
dfs = dfsp[['Content', 'Category']]

xs = dfs['Category']
ys = dfs['Content']
xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, train_size=0.8, test_size=0.2, random_state=3)
tfvecs = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
ys_trainFeat = tfvecs.fit_transform(ys_train)
ys_testFeat = tfvecs.transform(ys_test)

# SVM is used to model
xs_trainSvm = xs_train.astype('int')
classifierSVM.fit(ys_trainFeat, xs_trainSvm)
predResSmsSVM = classifierSVM.predict(ys_testFeat)

# MNB is used to model
xs_trainGnb = xs_train.astype('int')
classifierMNB.fit(ys_trainFeat, xs_trainGnb)
predResSmsMNB = classifierMNB.predict(ys_testFeat)

#KNN is used to model
xs_trainKNN = xs_train.astype('int')
# classifierKNN = KNeighborsClassifier(n_neighbors=1)
classifierKNN.fit(ys_trainFeat, xs_trainKNN)
predResSmsKNN = classifierKNN.predict(ys_testFeat)

#RF is used to model
xs_trainRF = xs_train.astype('int')
# classifierRF = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=None)
classifierRF.fit(ys_trainFeat, xs_trainRF)
predResSmsRF = classifierRF.predict(ys_testFeat)

#Adaboost is used to model
xs_trainAdaB = xs_train.astype('int')
# classifierAdaB = AdaBoostClassifier(n_estimators=100)
classifierAdaB.fit(ys_trainFeat, xs_trainAdaB)
predResSmsAdaB = classifierAdaB.predict(ys_testFeat)

# Converting to int - solves - cant handle mix of unknown and binary
xs_test = xs_test.astype('int')
actual_Ys = xs_test.to_numpy()

print("\tSupport Vector Machine RESULTS")
# Accuracy score using SVM
print("Accuracy Score using SVM: {0:.4f}".format(accuracy_score(actual_Ys, predResSmsSVM) * 100))
# FScore MACRO using SVM
print("F Score using SVM: {0: .4f}".format(f1_score(actual_Ys, predResSmsSVM, average='macro') * 100))
cmSVMs = confusion_matrix(actual_Ys, predResSmsSVM)
# [True negative  False Positive
# False Negative True Positive]
print("Confusion matrix using SVM:")
print(cmSVMs)
print()
print("\tMultinomial Näive Bayes RESULTS")
# Accuracy score using MNB
print("Accuracy Score using MNB: {0:.4f}".format(accuracy_score(actual_Ys, predResSmsMNB) * 100))
# FScore MACRO using MNB
print("F Score using MNB:{0: .4f}".format(f1_score(actual_Ys, predResSmsMNB, average='macro') * 100))
cmMNbs = confusion_matrix(actual_Ys, predResSmsMNB)
# [True negative  False Positive
# False Negative True Positive]
print("Confusion matrix using MNB:")
print(cmMNbs)
print()
print("\tK Nearest Neighbors RESULTS")
print("Neighbors Number: 1")
# Accuracy score using KNN
print("Accuracy Score using KNN: {0:.4f}".format(accuracy_score(actual_Ys, predResSmsKNN) * 100))
# FScore MACRO using KNN
print("F Score using KNN:{0: .4f}".format(f1_score(actual_Ys, predResSmsKNN, average='macro') * 100))
cmKNNs = confusion_matrix(actual_Ys, predResSmsKNN)
# [True negative  False Positive
# False Negative True Positive]
print("Confusion matrix using KNN:")
print(cmKNNs)
print()
print("\tRandom Forest RESULTS")
# Accuracy score using MNB
print("Accuracy Score using RF: {0:.4f}".format(accuracy_score(actual_Ys, predResSmsRF) * 100))
# FScore MACRO using MNB
print("F Score using RF:{0: .4f}".format(f1_score(actual_Ys, predResSmsRF, average='macro') * 100))
cmRFs = confusion_matrix(actual_Ys, predResSmsRF)
# [True negative  False Positive
# False Negative True Positive]
print("Confusion matrix using RF:")
print(cmRFs)
print()
print("\tAda Boost RESULTS")
print("Estimators Number: 100")
# Accuracy score using MNB
print("Accuracy Score using AdaB: {0:.4f}".format(accuracy_score(actual_Ys, predResSmsAdaB) * 100))
# FScore MACRO using MNB
print("F Score using AdaB:{0: .4f}".format(f1_score(actual_Ys, predResSmsAdaB, average='macro') * 100))
cmAdaBs = confusion_matrix(actual_Ys, predResSmsAdaB)
# [True negative  False Positive
# False Negative True Positive]
print("Confusion matrix using AdaB:")
print(cmAdaBs)

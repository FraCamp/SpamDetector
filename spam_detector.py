# importing dependencies
import pandas as pd
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
# df = df1.where((pd.notnull(df1)), '')

# Categorize Spam as 0 and Not spam as 1
df.loc[df["label"] == 'ham', "Category"] = 0
df.loc[df["label"] == 'spam', "Category"] = 1
# Leaving the original column "label" (ham, spam), adding a new column "Label"(1,0)
# df.loc[df["label"] == 'ham', "Label",] = 0
# df.loc[df["label"] == 'spam', "Label",] = 1

df = df.rename(columns={'text': 'Content'})
dff = df[['Content', 'Category']]

x = dff['Category']
y = dff['Content']

# splitting the original dataset (diveded by columns, x and y) into 4 different dataset
# x_train, x_test and y_train and y_test, using the train_test_split function (sklearn) is possible to decide the
# shuffling of the data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=3)
# print("x_train")
# print(x_train)
# print("y_train")
# print(y_train)
#
# print("x_test")
# print(x_test)
# print("y_test")
# print(y_test)

# feature extraction, coversion to lower case and removal of stop words using TFIDF VECTORIZER

# Finding the term frequency-inverse document frequency (tf-idf, useful into textanalysis and in order to use machine
# learning algorithm for Natural Language Processing) by multiplying two metrics: how many times a
# word appears in a document, and the inverse document frequency of the word across a set of documents
tfvec = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
# print("TfVectorizer")

y_trainFeat = tfvec.fit_transform(y_train)
y_testFeat = tfvec.transform(y_test)
# print("Fit_trasform")
# print(y_trainFeat.toarray())
# print("Trasform")
# print(y_testFeat.toarray())

# print(dff)

# SVM is used to model
x_trainSvm = x_train.astype('int')
classifierSVM = LinearSVC()
classifierSVM.fit(y_trainFeat, x_trainSvm)
predResMailSVM = classifierSVM.predict(y_testFeat)

# MNB is used to model
x_trainGnb = x_train.astype('int')
classifierMNB = MultinomialNB()
classifierMNB.fit(y_trainFeat, x_trainGnb)
predResMailMNB = classifierMNB.predict(y_testFeat)

#KNN is used to model
x_trainKNN = x_train.astype('int')
classifierKNN = KNeighborsClassifier(n_neighbors=1)
classifierKNN.fit(y_trainFeat, x_trainKNN)
predResMailKNN = classifierKNN.predict(y_testFeat)

# Calc accuracy,converting to int - solves - cant handle mix of unknown and binary
x_test = x_test.astype('int')
# print(x_test)
actual_Y = x_test.to_numpy()

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

# Calc accuracy,converting to int - solves - cant handle mix of unknown and binary
xs_test = xs_test.astype('int')
# print(x_test)
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
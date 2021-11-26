# importing dependencies
import pandas as pd
# to disable the warning for chained assignment when splitting the datasets for test and training sets
pd.options.mode.chained_assignment = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# functions definitions to train and classify with different algorithms
def SVM(x_train, y_trainFeat, y_testFeat):
    # x_train = x_train.astype('int')
    classifier = LinearSVC()
    classifier.fit(y_trainFeat, x_train)
    predRes = classifier.predict(y_testFeat)
    return predRes

def MNB(x_train, y_trainFeat, y_testFeat):
    # x_train = x_train.astype('int')
    classifier = MultinomialNB()
    classifier.fit(y_trainFeat, x_train)
    predRes = classifier.predict(y_testFeat)
    return predRes

def KNN(x_train, y_trainFeat, y_testFeat):
    # x_train = x_train.astype('int')
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(y_trainFeat, x_train)
    predRes = classifier.predict(y_testFeat)
    return predRes

def RF(x_train, y_trainFeat, y_testFeat):
    # x_train = x_train.astype('int')
    classifier = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=None)
    classifier.fit(y_trainFeat, x_train)
    predRes = classifier.predict(y_testFeat)
    return predRes

def Adab(x_train, y_trainFeat, y_testFeat):
    # x_train = x_train.astype('int')
    classifier = AdaBoostClassifier(n_estimators=100)
    classifier.fit(y_trainFeat, x_train)
    predRes = classifier.predict(y_testFeat)
    return predRes


# function to show results
def show_res(actual, predicted):
    # Accuracy score using SVM
    print("Accuracy Score: {0:.4f}".format(accuracy_score(actual, predicted) * 100))
    # FScore MACRO using SVM
    print("F Score: {0: .4f}".format(f1_score(actual, predicted, average='macro') * 100))
    cm = confusion_matrix(actual, predicted)
    # [True negative  False Positive
    # False Negative True Positive]
    print("Confusion Matrix:")
    print(cm)
    print()


# Data Preprocessing
# E-mail dataset
df1 = pd.read_csv("email_spam.csv")
df = df1[['label_num', 'text']]
df = df.rename(columns={'text': 'Content'})
df = df.rename(columns={'label_num': 'Category'})
x = df['Category']
y = df['Content']

# splitting the original dataset (diveded by columns, x and y) into random train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=3)

# TfidfVectorizer is equivalent to CountVectorizer followed by TfidfTransformer
tfvec = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Converting to int - solves - cant handle mix of unknown and binary
x_train = x_train.astype('int')
actual_x = x_test.to_numpy()
y_trainFeat = tfvec.fit_transform(y_train)
y_testFeat = tfvec.transform(y_test)


# SMS dataset
# using encoding options in order to open and clean the csv, which has some empty columns
df2 = pd.read_csv("sms_spam.csv", encoding = "ISO-8859-1")
dfsp = df2[['v1', 'v2']]
dfsp.loc[dfsp["v1"] == 'ham', "Category"] = 0
dfsp.loc[dfsp["v1"] == 'spam', "Category"] = 1
dfsp = dfsp.rename(columns={'v2': 'Content'})
dfs = dfsp[['Content', 'Category']]

xs = dfs['Category']
ys = dfs['Content']

# splitting the original dataset (diveded by columns, x and y) into random train and test sets
xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, train_size=0.8, test_size=0.2, random_state=3)

# TfidfVectorizer is equivalent to CountVectorizer followed by TfidfTransformer
tfvecs = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
ys_trainFeat = tfvecs.fit_transform(ys_train)
ys_testFeat = tfvecs.transform(ys_test)
xs_trainSvm = xs_train.astype('int')
# Converting to int - solves - cant handle mix of unknown and binary
xs_test = xs_test.astype('int')
actual_xs = xs_test.to_numpy()

# Training and classification
print("/-------------------SpamDetector for e-Mail-------------------/")
# SVM
predResMailSVM = SVM(x_train, y_trainFeat, y_testFeat)
# Metrics and results
print("\tSupport Vector Machine results")
show_res(actual_x, predResMailSVM)

# MNB
predResMailMNB = MNB(x_train, y_trainFeat, y_testFeat)
# Metrics and results
print("\tMultinomial Näive Bayes results")
show_res(actual_x, predResMailMNB)

# KNN
predResMailKNN = KNN(x_train, y_trainFeat, y_testFeat)
# Metrics and results
print("\tK Nearest Neighbors results")
print("Neighbors Number: 1")
show_res(actual_x, predResMailKNN)

# RF
predResMailRF = RF(x_train, y_trainFeat, y_testFeat)
# Metrics and results
print("\tRandom Forest results")
show_res(actual_x, predResMailRF)

# Adaboost
predResMailAdab = Adab(x_train, y_trainFeat, y_testFeat)
# Metrics and results
print("\tAdaboost results")
print("Estimators Number: 100")
show_res(actual_x, predResMailAdab)


print("\n/---------------------SpamDetector for SMS--------------------/")
# SVM
predResSmsSVM = SVM(xs_train, ys_trainFeat, ys_testFeat)
print("\tSupport Vector Machine results")
show_res(actual_xs, predResSmsSVM)

# MNB
predResSmsMNB = MNB(xs_train, ys_trainFeat, ys_testFeat)
# Metrics and results
print("\tMultinomial Näive Bayes results")
show_res(actual_xs, predResSmsMNB)

#KNN
predResSmsKNN = KNN(xs_train, ys_trainFeat, ys_testFeat)
# Metrics and results
print("\tK Nearest Neighbors results")
print("Neighbors Number: 1")
show_res(actual_xs, predResSmsKNN)

#RF
predResSmsRF = RF(xs_train, ys_trainFeat, ys_testFeat)
# Metrics and results
print("\tRandom Forest results")
show_res(actual_xs, predResSmsRF)

#Adaboost
predResSmsAdab = Adab(xs_train, ys_trainFeat, ys_testFeat)
# Metrics and results
print("\tAdaboost results")
print("Estimators Number: 100")
show_res(actual_xs, predResSmsAdab)
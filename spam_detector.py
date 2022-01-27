# importing dependencies
import pandas as pd
# to disable the warning for chained assignment when splitting the datasets for test and training sets
pd.options.mode.chained_assignment = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


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
    classifier = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=None)
    classifier.fit(y_trainFeat, x_train)
    predRes = classifier.predict(y_testFeat)
    return predRes

def Adab(x_train, y_trainFeat, y_testFeat):
    classifier = AdaBoostClassifier(n_estimators=100)
    classifier.fit(y_trainFeat, x_train)
    predRes = classifier.predict(y_testFeat)
    return predRes


# function to show results
def show_res(actual, predicted):
    print("Accuracy Score: {0:.4f}".format(accuracy_score(actual, predicted)))
    print("Precision: {0: .4f}".format(precision_score(actual, predicted)))
    print("Recall: {0: .4f}".format(recall_score(actual, predicted)))
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

training_data = y_train.tolist()
test_data = y_test.tolist()

#Counting the words occurences
count_vector = CountVectorizer() #tokenization, stopword filtering and relevant tokens identification
y_train_counts = count_vector.fit_transform(raw_documents=training_data)

print("Tokens extracted:")
print(count_vector.get_feature_names_out())
#print(count_vector.get_feature_names()) Deprecated function (get_features_names)
print("Description of the word occurences data stractures:")
print(type(y_train_counts))
print("(Documents, Tokens)")
print(y_train_counts.shape)
print("Word occurences of the first document:")
print(y_train_counts[1])

#TF-IDF extraction
tfidf_transformer = TfidfTransformer()
y_train_tfidf = tfidf_transformer.fit_transform(y_train_counts)
print("Values of features extracted from the first document:")
print(y_train_tfidf[0])

y_test_counts = count_vector.transform(raw_documents=test_data)
y_test_tfidf = tfidf_transformer.transform(y_test_counts)

# Converting to int - solves - cant handle mix of unknown and binary
x_train = x_train.astype('int')
actual_x = x_test.to_numpy()


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

training_data_sms = ys_train.tolist()
test_data_sms = ys_test.tolist()

#Counting the words occurences
count_vector = CountVectorizer()
y_train_counts_sms = count_vector.fit_transform(raw_documents=training_data_sms)

print("Tokens extracted:")
print(count_vector.get_feature_names_out())
#print(count_vector.get_feature_names()) Deprecated function (get_features_names)
print("Description of the word occurences data stractures:")
print(type(y_train_counts_sms))
print("(Documents, Tokens)")
print(y_train_counts_sms.shape)
print("Word occurences of the first document:")
print(y_train_counts_sms[1])

#TF-IDF extraction
tfidf_transformer = TfidfTransformer()
y_train_tfidf_sms = tfidf_transformer.fit_transform(y_train_counts_sms)
print("Values of features extracted from the first document:")
print(y_train_tfidf_sms[0])

y_test_counts_sms = count_vector.transform(raw_documents=test_data_sms)
y_test_tfidf_sms = tfidf_transformer.transform(y_test_counts_sms)

# Converting to int - solves - cant handle mix of unknown and binary
xs_test = xs_test.astype('int')
actual_xs = xs_test.to_numpy()

# Training and classification
print("/-------------------SpamDetector for e-Mail-------------------/")
# SVM
predResMailSVM = SVM(x_train, y_train_tfidf, y_test_tfidf)
# Metrics and results
print("\tSupport Vector Machine results")
show_res(actual_x, predResMailSVM)

# MNB
predResMailMNB = MNB(x_train, y_train_tfidf, y_test_tfidf)
# Metrics and results
print("\tMultinomial Näive Bayes results")
show_res(actual_x, predResMailMNB)

# KNN
predResMailKNN = KNN(x_train, y_train_tfidf, y_test_tfidf)
# Metrics and results
print("\tK Nearest Neighbors results")
print("Neighbors Number: 1")
show_res(actual_x, predResMailKNN)

# RF
predResMailRF = RF(x_train, y_train_tfidf, y_test_tfidf)
# Metrics and results
print("\tRandom Forest results")
show_res(actual_x, predResMailRF)

# Adaboost
predResMailAdab = Adab(x_train, y_train_tfidf, y_test_tfidf)
# Metrics and results
print("\tAdaboost results")
print("Estimators Number: 100")
show_res(actual_x, predResMailAdab)


print("\n/---------------------SpamDetector for SMS--------------------/")
# SVM
predResSmsSVM = SVM(xs_train, y_train_tfidf_sms, y_test_tfidf_sms)
print("\tSupport Vector Machine results")
show_res(actual_xs, predResSmsSVM)

# MNB
predResSmsMNB = MNB(xs_train, y_train_tfidf_sms, y_test_tfidf_sms)
# Metrics and results
print("\tMultinomial Näive Bayes results")
show_res(actual_xs, predResSmsMNB)

#KNN
predResSmsKNN = KNN(xs_train, y_train_tfidf_sms, y_test_tfidf_sms)
# Metrics and results
print("\tK Nearest Neighbors results")
print("Neighbors Number: 1")
show_res(actual_xs, predResSmsKNN)

#RF
predResSmsRF = RF(xs_train, y_train_tfidf_sms, y_test_tfidf_sms)
# Metrics and results
print("\tRandom Forest results")
show_res(actual_xs, predResSmsRF)

#Adaboost
predResSmsAdab = Adab(xs_train, y_train_tfidf_sms, y_test_tfidf_sms)
# Metrics and results
print("\tAdaboost results")
print("Estimators Number: 100")
show_res(actual_xs, predResSmsAdab)
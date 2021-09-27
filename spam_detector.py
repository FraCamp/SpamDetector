# importing dependencies
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

df1 = pd.read_csv("email_spam.csv")
df = df1[['label', 'text']]
# df = df1.where((pd.notnull(df1)), '')

# Categorize Spam as 0 and Not spam as 1
df.loc[df["label"] == 'ham', "Category",] = 0
df.loc[df["label"] == 'spam', "Category",] = 1
# Leaving the original column "label" (ham, spam), adding a new column "Label"(1,0)
# df.loc[df["label"] == 'ham', "Label",] = 0
# df.loc[df["label"] == 'spam', "Label",] = 1

df = df.rename(columns={'text':'Content'})
dff = df[['Content', 'Category']]

x = dff['Category']
y = dff['Content']


# splitting the original dataset (diveded by columns, x and y) into 4 different dataset
# x_train, x_test and y_train and y_test, using the train_test_split function (sklearn) is possible to decide the
# shuffling of the data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=3)
print("x_train")
print(x_train)
print("y_train")
print(y_train)

print("x_test")
print(x_test)
print("y_test")
print(y_test)

# feature extraction, coversion to lower case and removal of stop words using TFIDF VECTORIZER

# Finding the term frequency-inverse document frequency (tf-idf, useful into textanalysis and in order to use machine
# learning algorithm for Natural Language Processing) by multiplying two metrics: how many times a
# word appears in a document, and the inverse document frequency of the word across a set of documents
tfvec = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
print("TfVectorizer")

y_trainFeat = tfvec.fit_transform(y_train)
y_testFeat = tfvec.transform(y_test)
print("Fit_trasform")
print(y_trainFeat.toarray())
print("Trasform")
print(y_testFeat.toarray())

# print(dff)
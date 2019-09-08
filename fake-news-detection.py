import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#Loading Data

df = pd.read_csv('fake_or_real_news.csv')
df.set_index('Unnamed: 0',inplace=True)


print(df.isnull().sum())

y = df['label']
df.drop('label', axis=1, inplace=True)
print (y)

#Spliting the dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(df['text'], y, test_size = 0.2, random_state=42)


#Count Vectorizer Initialization
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(x_train)
count_test = count_vectorizer.transform(x_test)

#Training the Classifier
cclf = MultinomialNB()
cclf.fit(count_train, y_train)

#Predicting
cpredicted = cclf.predict(count_test)
print(accuracy_score(cpredicted, y_test))

#Tfid Vector Initialization
tfidf_vectorizer = TfidfVectorizer()
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)


#Training the classifer
tclf = MultinomialNB()
tclf.fit(tfidf_train, y_train)

#Predicting 
tpredicted = tclf.predict(tfidf_test)
print(accuracy_score(tpredicted, y_test))




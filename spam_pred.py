import pandas as pd

messages = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])

#print(messages)


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]'," ",messages['message'][i])

    review = review.lower()

    review = review.split()

    review = [ps.stem(word) for word in review if  not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)





from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)

x = cv.fit_transform(corpus).toarray()



y = pd.get_dummies(messages['label'])

y = y.iloc[:,1].values



from sklearn.model_selection import train_test_split

x_train,y_train,x_test,y_test = train_test_split(x,y,test_size=0.10,random_state=0)


from sklearn.naive_bayes import MultinomialNB

spam_detection = MultinomialNB().fit(x_train,y_train.ravel())

y_pred = spam_detection.predict(x_test)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,y_pred)


print(accuracy)

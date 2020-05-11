import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

messages=pd.read_csv("C:/Users/dell/Downloads/smsspamcollection/SMSSpamCollection",sep="\t"
                         ,names=['label','message'])

ps=PorterStemmer()
#data cleaning
corpus=[]
for i in range(0,len(messages)):
    sent=re.sub('^a-zA-Z'," ",messages['message'][i])
    
    sent=sent.lower()
    words=nltk.word_tokenize(sent)
    #words=sent.split()
    words=[ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
    words=' '.join(words)
    corpus.append(words)

#bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
X=cv.fit_transform(corpus).toarray()

#code

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#training
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train,y_train)

#testing
y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confuson_m=confusion_matrix(y_test,y_pred)

#accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

#accuracy=98.744%

    
    
    






                                                      



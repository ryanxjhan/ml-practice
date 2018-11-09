
### Read Data 


```python
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 

data_dir = "../input/"

df = pd.read_csv(data_dir + '/spam.csv', encoding='latin-1')  
```

### Spliting Data 


```python
from sklearn.model_selection import train_test_split

data_train, data_test, labels_train, labels_test = train_test_split(df.v2, df.v1, test_size=0.2, random_state=0)
```

### Count Frequencies of Words in Training Data 


```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

data_train_count = vectorizer.fit_transform(data_train)
data_test_count  = vectorizer.transform(data_test)
print(data_train_count.shape)
print(data_test_count.shape)
```

    (4457, 7612)
    (1115, 7612)


### How CountVectorizer Works 
fit_transform: count and locate unique words by assigning an id (ex. (1,2)). 

[Documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)


```python
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
example = ['Apple and pen', 'Apple pen pen']

result = vect.fit_transform(example)

print(result)
```

      (0, 2)	1
      (0, 0)	1
      (0, 1)	1
      (1, 2)	2
      (1, 1)	1



```python
print (vect.stop_words_)
```

    set()



```python
print (vect.vocabulary_)
```

    {'apple': 1, 'and': 0, 'pen': 2}


### Training Model


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

clf = MultinomialNB()
clf.fit(data_train_count, labels_train)
predictions = clf.predict(data_test_count)
print(predictions)
```

    ['ham' 'ham' 'ham' ... 'ham' 'ham' 'ham']



```python
print(accuracy_score(labels_test, predictions))
```

    0.9874439461883409


### Credit

Data Credit: [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

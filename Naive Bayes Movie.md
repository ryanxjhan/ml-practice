
### Read Data 


```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas as pd

train = pd.read_csv('../input/labeledTrainData.tsv', delimiter="\t")
test = pd.read_csv('../input/testData.tsv', delimiter="\t")

train.head()                
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>sentiment</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5814_8</td>
      <td>1</td>
      <td>With all this stuff going down at the moment w...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2381_9</td>
      <td>1</td>
      <td>\The Classic War of the Worlds\" by Timothy Hi...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7759_3</td>
      <td>0</td>
      <td>The film starts with a manager (Nicholas Bell)...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3630_4</td>
      <td>0</td>
      <td>It must be assumed that those who praised this...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9495_8</td>
      <td>1</td>
      <td>Superbly trashy and wondrously unpretentious 8...</td>
    </tr>
  </tbody>
</table>
</div>




```python
print (train.shape)
print (test.shape) # Test data does not have results
```

    (25000, 3)
    (25000, 2)


### Clean Data 


```python
import re  

def review_to_wordlist(review):
    #Keep English words only. 
    review_text = re.sub("[^a-zA-Z]"," ", review)
    
    #Turn words into lowercase
    words = review_text.lower()
    
    return(words)

# Seperate indices and text

# Indices
y_train = train['sentiment']

# Text
train_data = []
for review in train['review']:
    train_data.append(review_to_wordlist(review))
    
# Convert to numpy array      
train_data = np.array(train_data)

# Same for testing data
test_data = []
for review in test['review']:
    test_data.append(review_to_wordlist(review))
    
test_data = np.array(test_data)

print(train_data.shape)
print(test_data.shape)
```

    (25000,)
    (25000,)


### Vectorize Data 
* [ngram_range](https://www.kaggle.com/c/avito-demand-prediction/discussion/58819)
* [fit_transform vs transform](https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models)


```python
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer

# counting
# vectorizer = CountVectorizer()
# data_train_count = vectorizer.fit_transform(train_data)
# data_test_count  = vectorizer.transform(test_data)

# tf-idf
tfidf = TfidfVectorizer(
           ngram_range=(1, 3),  
           use_idf=1,
           smooth_idf=1,
           stop_words = 'english') # Romove stop words


data_train_count = tfidf.fit_transform(train_data)
data_test_count  = tfidf.transform(test_data)

print("Let's go!")
```

    Let's go!


### Multinomial Naive Bayes


```python
from sklearn.naive_bayes import MultinomialNB 

clf = MultinomialNB()
clf.fit(data_train_count, y_train)
pred = clf.predict(data_test_count)
print (pred)
```

    [1 0 1 ... 1 1 0]


### Save Results


```python
df = pd.DataFrame({"id": test['id'],"sentiment": pred})

df.to_csv('submission.csv',index = False, header=True)
```

### Credit

Data Credit: [Kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial/)

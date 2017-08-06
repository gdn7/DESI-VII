
"""
Created on Tue Feb 28 12:09:17 2017

@author: gdnir
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import nltk
nltk.download()

os.chdir("")

extracted_doc=open("extracted body.txt","r",encoding="utf8")
print(extracted_doc.read())


from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in extracted_doc]        
```




``
# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
                                    
```

```
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=50)

```


``````````````````````````````````````````````````````````````````````````
print(ldamodel.print_topics(num_topics=5, num_words=25))




import pyLDAvis.gensim as gensimvis
import pyLDAvis



vis_data = gensimvis.prepare(ldamodel, doc_term_matrix, dictionary)
pyLDAvis.show(vis_data)

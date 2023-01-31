#!/usr/bin/env python
# coding: utf-8

# # CIS 511 Final Project - Vishal Tejrao Patil

# 1. In the beginning, I have imported the libraries and the dataset(500 news articles).
# 2. Then I cleaned the  dataset by removing the stopwords, punctuation marks and lemmatized the words in the sentences of the        articles.
# 3. I have calculated the term frequencies and Inverse Document Frequency for the News article to extract top 10 Keywords.
# 4. Then I have implemented Luhn's Algorithm by calculating the weights of each sentence and sorting them in descending order.
# 5. Then I used top 3 weighted sentences to summarize the News article.
# 6. I collected the total sentences of the count in original article news and summarised into 40% of the original article.
#    For e.g.:There are 10 sentences in an article. The summary would contain 40% = 4 sentences. 
# 7. Then, I ROUGE as my evaluation metric and compared the machine-generated summary with the news highlights provided in the    dataset.
# 8. Finally, I plotted a histogram of Frequency vs ROGUE-1 score

# ### Importing Libraries

# In[1]:


import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from collections import Counter
import math
from rouge_score import rouge_scorer
from matplotlib import pyplot as plt


# ### Reading the Dataset
# 

# In[2]:


df = pd.read_csv('train.csv')



# ### Making a copy of dataframe

# In[3]:


corpus = df.copy()


# ### Removing stopwords and punctuations and convert to lowercase

# In[4]:



#creating a list of stopwords
stop_words = set(stopwords.words('english')) 


#tokenizing the text
for a in corpus['article']:
    tokens = word_tokenize(a)

#remove punctuation from each token
table = str.maketrans('', '', string.punctuation)
tokens = [w.translate(table) for w in tokens]

#filter out stop words
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]

#convert to lowercase
tokens = [word.lower() for word in tokens]

#Lemmatizing
lemmatizer = WordNetLemmatizer()        
for t in tokens:
    t = lemmatizer.lemmatize(t)

print(tokens)


# ### Calculating Term Frequencies
# 

# In[5]:



freq = Counter(tokens)
print(freq)


# ### Calculating IDF

# In[6]:


#counting the number of documents in the corpus and counting the number of documents that contain each token.

idf = {}
count = 0
for token in tokens:
    if token not in idf.keys():
        idf[token] = 1
        count +=1
    else:
        idf[token] +=1
for token in idf.keys():
    idf[token] = math.log(count/idf[token])

    print(idf)


# ### Calculating weights of each sentence in the article using Luhn Algorithm
# 

# In[7]:


#If the word is in the frequency dictionary, it adds the word's frequency multiplied by its IDF value to the weight variable.

def luhn(sentences):
    sentences_count = 0
    weights = []
    for sentence in sentences:
        sentences_count += 1
        word_list = word_tokenize(sentence)
        weight = 0
        
        for word in word_list:
            if word in freq.keys():
                weight += freq[word]*idf[word]
        weights.append(weight)
    return weights, sentences_count


# ### Getting the weights using Luhn Algorithm and Calculating ROGUE Score

# In[8]:


#loop to go through each article
highlights = corpus['highlights']
f_list = []

for idx_highlights in range(len(corpus)):
    
    #determining the keywords for each article
    keywords = {}
    
    keywords = sorted(idf, key=idf.get, reverse=True)[:10]
    print("Keywords: {}\n".format(keywords))
    
    #summarizing the article using Luhn Algorithm
    sentences = corpus['article'][idx_highlights].split('.')
    weights, sentence_count = luhn(sentences)
    print("Total Sentences: ",sentence_count)
    summary = ""
    
    factor = sentence_count*0.4
    factor = int(factor) 
    
    print("Summary sentences:",factor)
    for i in range(factor):
        index = weights.index(max(weights))
        summary += sentences[index]+'.'
        weights[index] = 0
    
    print("\nSummary:\n {}\n".format(summary))
    
#     print("Highlights:\n {}\n".format(highlights[idx_highlights])) 

    
    
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(summary, highlights[idx_highlights])
    
    for x in scores.values():
        x = x[2]
        f_list.append(x)        
    
    print("ROGUE:\n {}\n".format(scores))
    print("-------------------------------------------------------------------------------------------------------")
    


# In[9]:


#Evaluating average ROUGE-1 Score
avg_rg = sum(f_list)/len(f_list)
avg_rg


# ### Plotting a histogram

# In[10]:


# Plot Histogram on x
plt.hist(f_list, bins=50)
plt.gca().set(title='Frequency Histogram Luhn Algo Summarization', ylabel='Frequency',xlabel='ROUGE-1');


# In[ ]:





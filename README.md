# Text-Summarization-
Text summarization is the process of extracting important details from a text document. It is used to produce a succinct summary of the document, and this can be done in one of two ways depending on output: either by extraction or abstraction. The extractive approaches concentrate on choosing a group of words or phrases from the source text to produce the summary. In this work, I have implemented the extractive summary based on Standard TF-IDF based, and Centroid based summarization to summarize the text from an article from multiple newspaper articles.

Unsupervised Text Summarization methods (Extractive Summarization):

                                                                   i) TF-IDF Based Summarization

                                                                   ii) Centroid Based Summarization

TF-IDF Based Summarizer (Calculation):

            Term Frequency: For every word in given sentence we have calculated term frequency  

                             TF =Number of Repetition of Wi in  sentence/Number of words in sentence

            Inverse Document Frequency: For every word in complete text given IDF Score.

                             IDF=log(Total Number of Sentences/Number of sentences containg word Wi)
                          
                             TF-IDF: Assigning weight for each word based on TF and IDF score
                          
                             Weight(Wi) =TF*IDF
                                  
           Sentence Weightage: Sentence Weight=Sum of weight of all words in sentenceNumber of words in Sentence
           
           Sentence Extraction: sentence weight> average sentence weight * tuning factor  
           
           Those sentence will be part of summariztion, which have high sentence weight greater than the average sentence weight. 
           

Centroid Based Summarizer (Calculation):
           
            Centroid Vector: We have taken those words in a text which have higher tf-idf score than the average tf-idf score.

            Sentence Vector: All the words along with tf-idf score in a sentence act as vector

            Cosine Sentence Similarity: Weight of sentence given based on the cosine similarity between the centroid vector and sentence vector.
            
            (Refer: https://en.wikipedia.org/wiki/Cosine_similarity )

                           
            Sentence Extraction: Cosine sentence similarity>avg cosine sentence similarity*tuning factor

            Those sentence will be part of summarization which have higher cosine sentence similary greater then the average cosine similarity score.





                                                                   
                                                             

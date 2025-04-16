# %%
import pandas as pd 
import os
import csv
import numpy as np

import string, re, nltk
from string import punctuation
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import re
import xml.etree.ElementTree as ET

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.feature_extraction.text import TfidfVectorizer

#import tkinter as tk
#import customtkinter as ctk

######### Text Preprocessing Functions #########

def convert_to_lower(text):
  """Converts text to lowercase"""
  return text.lower()

def prune_whitespace(text):
  """Removes whitespaces """
  return text.strip()

def remove_punctuation(text):
  """Remove punctuation symbols"""
  punct_str = string.punctuation
  punct_str = punct_str.replace("'", "") #discarding apostrphe from the string
  return text.translate(str.maketrans("", "", punct_str))

def remove_file_extension(text):
    """Remove file extension"""
    return re.sub(r'\.[^.]+$', '', text)

def replace_under_scores_with_space(text):
    """Replace underscores with spaces"""
    new_text = ""
    new_text = re.sub(r'_|-', ' ', text)
    #return text.replace("_", " ")
    return new_text

def replace_new_lines_with_space(text):
    flatline = re.sub(r'\n', ' ', text)
    return flatline

def text_stemmer(text):
    text_stem = " ".join([stemmer.stem(word) for word in (text)])
    return text_stem

#Tokenization

def tokenize_text(text):
    """Tokenize the text"""
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)

#Remove stopwords
def remove_stopwords(text):
    """Remove stopwords"""
    generic_stop_words = stopwords.words('english')
    total_stop_words = generic_stop_words + ['pictures', 'picture']
    
    stop_words = set(total_stop_words)

    return " ".join([word for word in text.split() if word not in stop_words])

# %%
def remove_duplicate_words(text):
    """Remove duplicate words"""
    seen = set()
    return [x for x in text if not (x in seen or seen.add(x))]

def preprocess_text(text, do_stem=True):
    text = convert_to_lower(text)
    text = remove_file_extension(text)
    text = replace_under_scores_with_space(text)
    #text = prune_whitespace(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
   # text = remove_duplicate_words(text)

    if do_stem:
        return [stemmer.stem(word) for word in tokenize_text(text)]
    else :
        return tokenize_text(text)

###Inverted Index Class
###
class InvertedIndex:

    def __init__(self, corpus_df):
        self.corpus_df = corpus_df
        self.inverted_index = {}
        self.corpus_size = self.corpus_df.shape[0]
        self.pre_copmuted_idf = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0;
        self.total_word_count = 0

    def create_index(self):
        """Create inverted index from scratch """
        for index, row in self.corpus_df.iterrows():
            doc_id = index
            self.doc_lengths[doc_id] = len(row['processed_text'])
            for (position, word) in enumerate(row['processed_text']):
                self.total_word_count += 1
                if word not in self.inverted_index:
                    self.inverted_index[word] = {}
                
                if doc_id not in self.inverted_index[word]:
                    self.inverted_index[word][doc_id] = [1, position]
                else:
                    self.inverted_index[word][doc_id][0] += 1

        self.precompute_avg_doc_length()
        self.pre_compute_idf_for_all_terms()

    def add_do_index(self, doc_format):
        """Adding a document to the inverted index 
            This subroutine takes in a string at the moment although a better approach would 
            be either a file and then parsing the document or an xml pointer (the cranfield database in XML)"""
        self.corpus_size += 1
        for (position,word) in enumerate(set(processed_text(doc_format))):
            self.total_word_count += 1
            if word not in self.inverted_index:
                    self.inverted_index[word] = {}
                
            if doc_id not in self.inverted_index[word]:
                self.inverted_index[word][doc_id] = [1, position]
            else:
                self.inverted_index[word][doc_id][0] += 1

        self.precompute_avg_doc_length()


    def get_list_of_documents_containing_word(self, word):
        if word in self.inverted_index:
            docs = [doc for doc in self.inverted_index[word].keys()]
            return(docs)
        else:
            return ([])
        
        
    def get_term_freq_in_corpus(self, word):
        docs = self.get_list_of_documents_containing_word(word)
        return(sum([self.inverted_index[word][doc][0] for doc in docs]))

    def compute_idf(self, word):
        #idf = len(self.get_list_of_documents_containing_word(word)) / self.corpus_size
        doc_freq = len(self.get_list_of_documents_containing_word(word))
        idf = np.log(((self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5)) + 1)
        return(idf)

    def pre_compute_idf_for_all_terms(self):
        for word in self.inverted_index.keys():
            self.pre_copmuted_idf[word] = self.compute_idf(word)

    def get_idf(self, word):
        return(self.pre_copmuted_idf[word])

    def precompute_avg_doc_length(self):
        total_doc_length = sum([self.doc_lengths[k] for k in self.doc_lengths.keys()])
        self.avg_doc_length = total_doc_length / self.corpus_size

    def get_doc_length(self, doc_id):
        return(self.doc_lengths[doc_id])


    def get_term_freq_in_doc(self, term, doc):
        if(doc in self.inverted_index[term]):
            return (self.inverted_index[term][doc][0])
        else:
            return 0

#Search Engine VSM Class


#Search Engine LM Class
class search_engine_lm():
    def __init__(self, m_index):
        #self.corpus_df = corpus_df
        #self.num_docs = 1400
        self.inverted_index = m_index
        #self.word_count_dict = {}
        self.lambda_param = 0.5
        self.mu = 30


    def get_document_list(self, search_query):
        doc_list = []
        for word in preprocess_text(search_query):
            doc_list += self.inverted_index.get_list_of_documents_containing_word(word)
            doc_list = list(set(doc_list))

        return(doc_list)
            

    def search_for_query(self, search_query, top_n_items=20):
        #Collection of documents that contain the search query terms
        
        doc_scores = {}
        query_list = [word for word in preprocess_text(replace_new_lines_with_space(search_query), True) if word in self.inverted_index.inverted_index]
        query_words = set(query_list)
        query_length = len(query_words)

        #print(f"performing search for query {search_query} terms : {query_words}")

        doc_list = self.get_document_list(search_query)
        #print(f"query words = {set(query_words)}")
        #print(f"doc-list = {doc_list}")

        for doc_id in doc_list:
            score = 0
            #print(f"for doc_id = {doc_id}")
            for word in set(query_words):
                #print(f"for word = {word}")
                if doc_id in self.inverted_index.get_list_of_documents_containing_word(word):
                    
                    doc_length = self.inverted_index.get_doc_length(doc_id)
                    term_freq = self.inverted_index.get_term_freq_in_doc(word, doc_id)
                    term_prob_doc = term_freq / doc_length 

                    term_prob_corpus = self.inverted_index.get_term_freq_in_corpus(word) / self.inverted_index.total_word_count

                    # Jelinek-Mercer smoothing: P(q|d) = λ * P(q|d) + (1-λ) * P(q|C)
                    #smoothed_prob = (self.lambda_param * term_prob_doc) + ((1 - self.lambda_param) * term_prob_corpus)

                    # Dirichlet Prior Smoothing Formula
                    smoothed_prob = (term_freq + self.mu * term_prob_corpus) / (doc_length + self.mu)

                    #if smoothed_prob > 0:
                        #score += np.log(smoothed_prob)
                    score += smoothed_prob


                    
                    #print(f"word : {word} doc_id : {doc_id} term_prob_doc : {term_prob_doc} prob_collection : {term_prob_corpus} doc_length : {doc_length} smoothed_prob = {smoothed_prob}")

            doc_scores[doc_id] = score
            #print(f"Doc_score for {doc_id} : {score}")

        #sorted_doc_scores = [k for k,_ in sorted(doc_scores.items(), key=lambda x:x[1], reverse=True)][:top_n_items]
        sorted_doc_scores = {k : v for k,v in sorted(doc_scores.items(), key=lambda x:x[1], reverse=True)}

        return(sorted_doc_scores)            








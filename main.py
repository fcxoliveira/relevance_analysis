from os import mkdir
import spacy
import nltk
import os

from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt_tab')

RELEVANCE_THRESHOLD = 0.5

PATH = 'analysis'

phrases_file = open("phrases.txt", "r", encoding='utf-8') 
  
phrases_data = phrases_file.read() 
  
phrases = phrases_data.splitlines()

phrases_file.close()

models_file = open("models.txt", "r", encoding='utf-8') 
  
models_data = models_file.read() 
  
models = models_data.splitlines()

models_file.close()

def get_relevant_words(text, model_name, relevance_threshold=0.2):
    spacy_nlp = spacy.load("pt_core_news_sm")
    
    doc = spacy_nlp(text)
    
    words = [token.text for token in doc if token.pos_ in ["NOUN", "ADJ", "PROPN"] and token.is_stop == False]

    unique_words = []
    og_words = []

    for i in set(words):
        if i.lower() not in unique_words:
            unique_words.append(i.lower())
            og_words.append(i)
            
    model = SentenceTransformer(model_name)
    
    query_embeddings = model.encode(text, convert_to_tensor=True)
    corpus_embeddings = model.encode(og_words, convert_to_tensor=True)
    
    hits = util.cos_sim(query_embeddings, corpus_embeddings)[0]
    
    print(og_words)
    
    print(hits)
    
    relevant_words = get_analyse_by_cos(query_embeddings, corpus_embeddings, og_words, relevance_threshold)

    return relevant_words

def get_analyse_by_semantic(query_embeddings, corpus_embeddings, og_words, relevance_threshold=0.2):
    hits = util.semantic_search(query_embeddings, corpus_embeddings)[0]
    relevant_words = [og_words[hit['corpus_id']] for hit in hits if hit['score'] > relevance_threshold]

    return relevant_words

def get_analyse_by_cos(query_embeddings, corpus_embeddings, og_words, relevance_threshold=0.2):
    relevant_words = []
    hits = util.cos_sim(query_embeddings, corpus_embeddings)[0]
    for i in range(len(og_words)-1):
        if float(hits[i]) > relevance_threshold:
            relevant_words.append(og_words[i])
    return relevant_words

if os.path.isdir(PATH) == False:
    mkdir(PATH)

file = open("{}/relevance_analysis_cosseno_{}.txt".format(PATH, RELEVANCE_THRESHOLD), "w", encoding='utf-8') 

for model in models:
    file.write("### Tests with model '{}'".format(model))
     
    for phrase in phrases:  
        file.write("### Test with phrase: \n")
        file.write("### Phrase \n'{}'".format(phrase))
        file.write("\n")
        
        file.write("### Relevant words \n")
        relevant_words = get_relevant_words(phrase, model, RELEVANCE_THRESHOLD)
        file.write("\n".join(relevant_words, ))
        file.write("\n")
        file.write("\n")
        
file.close()
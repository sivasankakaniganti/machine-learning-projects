from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib
vectorizer = joblib.load('models/vectorizer.pkl')
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def remove_stop_lemit(text):
  
    word_tokens = word_tokenize(text) 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w.strip() not in stop_words: 
            filtered_sentence.append(w) 
    #print(filtered_sentence) 
    lemma_word = []
    for w in filtered_sentence:
        word1 = wordnet_lemmatizer.lemmatize(w, pos = "n")
        word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v")
        word3 = wordnet_lemmatizer.lemmatize(word2, pos = "a")
        lemma_word.append(word3)
    lemma_word = " ".join(lemma_word)
    text=lemma_word.strip().lower()
    return text
def preprocess(text):
    text = remove_stop_lemit(text)
    text = re.sub('\(.*\n*.*\)',' ',text) #removing (any words) 
    text = re.sub('[,.?\'\"#$@^&*]+',' ',text)
    text = re.sub('[\n\s\-\\\/]+',' ',text) #removing \n \s - \
    text = re.sub('[,.?\'\"#$@^&*]+',' ',text)
    return text.strip()
def final_preprocess(text):
    text = preprocess(text)
    text = vectorizer.transform([text])
    return text
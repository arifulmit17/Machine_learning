#Dataset preprocessing
import nltk
from nltk.corpus import stopwords
import re

def preprocessing(train_data):
     REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
     BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
     STOPWORDS = set(stopwords.words('english'))

     def clean_text(text):
         """
        text: a string
        
        return: modified initial string
         """
         #text = BeautifulSoup(text, "lxml").text # HTML decoding
         text = text.lower() # lowercase text
         text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
         text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
         text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
         return text

     train_data['RequirementText_string'] = train_data['RequirementText_string'].apply(clean_text)
     return train_data

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


def lemtext(train_data):
     lemmatizer=WordNetLemmatizer()      
     def lemsent(sent):
             token_words=word_tokenize(sent)
             stem_sentence=[]
             for word in token_words:
               stem_sentence.append(lemmatizer.lemmatize(word))
               stem_sentence.append(" ")

             sentence=""
             sentence=sentence.join(stem_sentence)
             sentence=sentence.rstrip()
             
             return sentence
             
     #for sent in train_data.RequirementText_string:
      # st=stemsent(sent)
       
     train_data['RequirementText_string'] = train_data['RequirementText_string'].apply(lemsent)
    
     #print (st)
     return train_data
 
from nltk import word_tokenize
from num2words import num2words

def numbertoword(train_data):
    def numbtoword(sent):
             token_words=word_tokenize(sent)
             stem_sentence=[]
             for word in token_words:
                 if word.isdigit():
                  stem_sentence.append(num2words(word))
                  stem_sentence.append(" ")
                 else:
                  stem_sentence.append(word)
                  stem_sentence.append(" ")
                  
             sentence=""
             sentence=sentence.join(stem_sentence)
             return sentence
     #for sent in train_data.RequirementText_string:
      # st=stemsent(sent)
       
    train_data['RequirementText_string'] = train_data['RequirementText_string'].apply(numbtoword)
    return train_data
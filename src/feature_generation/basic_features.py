import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.data import find
try:
    find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords

class BasicFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def get_cleaned_string(self,text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)   # remove punctuation/symbols
            text = ' '.join(text.split())  
            return text

    def get_cleaned_text_for_transformer(self,text):
        text=' '.join(text.split())
        text=text.lower()
        return text
        
    def add_stopword_counts(self,df):
        stop_words = set(stopwords.words('english'))   # Loaded once per function call

        def count_stopwords(text):
            text = ' '.join(text.split()).lower()
            words = text.split()
            return sum(1 for word in words if word in stop_words)

        df['topic_stopword_count'] = df['topic'].apply(count_stopwords)
        df['answer_stopword_count'] = df['answer'].apply(count_stopwords)
        return df

    def get_punctuation_count(self,text):
        text=' '.join(text.split())
        count=len(re.findall(r'[.,!?;:\'\"()\[\]{}\-\—…]', text))
        return count
        
    def get_avg_word_length(self,text):
        text = ' '.join(text.split())
        text_clean = re.sub(r'[^\w\s]', '', text).lower()
        word_list = text_clean.split()
        if not word_list:
            return 0.0
        total_chars = len(''.join(word_list))
        total_words = len(word_list)    
        avg_word_len = total_chars / total_words
        return avg_word_len


    def get_capital_words_count(self,text):
        text = ' '.join(text.split())
        text=re.sub(r'[^A-Za-z\s]', '', text)
        text_list=text.split()
        uppercase_words_count=len([word for word in text_list if word.isupper()])
        return uppercase_words_count

    def get_number_count(self,text):
        text = ' '.join(text.split())
        number_count=len(re.findall(r'\d+',text))
        return number_count

    def get_symbols_count(self,text):
        text=' '.join(text.split())
        count=len(re.findall(r'[@#$%^&*+=|\\/<>~_]', text))
        return count


    #characters count function
    def get_characters_count(self,text):
        return len(text)

    #Words count
    def get_words_count(self,text):
        return len(text.split())

    #Unique words count
    def get_unique_words_count(self,text):
        return len(set(text.split()))


    
    def transform(self,data: pd.DataFrame)->pd.DataFrame:
        df=data.copy()
        df['cleaned_topics'] = df['topic'].apply(self.get_cleaned_string)
        df['cleaned_answers'] = df['answer'].apply(self.get_cleaned_string)


        # Stopwords count
        df=self.add_stopword_counts(df)
        
        # Characters count
        df['topic_character_count'] = df['topic'].apply(self.get_characters_count)
        df['answer_character_count'] = df['answer'].apply(self.get_characters_count)

        # Words count
        df['topic_word_count'] = df['topic'].apply(self.get_words_count)
        df['answer_word_count'] = df['answer'].apply(self.get_words_count)

        # Unique words count
        df['topic_unique_word_count'] = df['topic'].apply(self.get_unique_words_count)
        df['answer_unique_word_count'] = df['answer'].apply(self.get_unique_words_count)


        # Punctuation count
        df['topic_punctuation_count'] = df['topic'].apply(self.get_punctuation_count)
        df['answer_punctuation_count'] = df['answer'].apply(self.get_punctuation_count)

        # Average word length
        df['topic_avg_word_length'] = df['topic'].apply(self.get_avg_word_length)
        df['answer_avg_word_length'] = df['answer'].apply(self.get_avg_word_length)

        # Capital words count
        df['topic_capital_words_count'] = df['topic'].apply(self.get_capital_words_count)
        df['answer_capital_words_count'] = df['answer'].apply(self.get_capital_words_count)

        # Number count
        df['topic_number_count'] = df['topic'].apply(self.get_number_count)
        df['answer_number_count'] = df['answer'].apply(self.get_number_count)

        # Symbol count
        df['topic_symbol_count'] = df['topic'].apply(self.get_symbols_count)
        df['answer_symbol_count'] = df['answer'].apply(self.get_symbols_count)
        

        return df    
    
    def fit(self, X, y=None):
        return self
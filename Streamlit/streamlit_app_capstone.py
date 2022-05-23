#streamlit run "C:\Users\yeosi\Documents\Python MAGES\07 DS106\Capstone\Streamlit\streamlit_app_capstone.py"
# Run the relevant libraries
from msilib.schema import CheckBox
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')

import re
import string
import time
import gensim
import gensim.downloader as api
import nltk
from cleantext import clean
from gensim.corpora.dictionary import Dictionary
from gensim.models.keyedvectors import KeyedVectors
from gensim.summarization import keywords
from gensim.test.utils import common_texts
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pylab import rcParams
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
from wordcloud import WordCloud

import unicodedata
import warnings
import os
import contractions
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.gensim_models as gensimvi
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from textblob import TextBlob

warnings.filterwarnings('ignore')

# Loading list of stop words from NLTK
stop_words = set(stopwords.words('english'))

# Remove word 'not' in stopwords as Not can depict emotions
stop_words.remove('not')

# Create function to define and remove custom stopwords 
def remove_custom_stopwords(text):
    stopset = set(stopwords.words("english"))
    for word in ['via', 'etc','very', 'hotel', 'room', 'stay', 'square', 'would', 'also', 'singapore', 'clarke', 'quay','ibis']:
        stopset.add(word)
  
    text = ' '.join([word for word in text.split() if word not in stopset])
    return text


# Create a function to map Part of Speech tags to the tokens
def pos_tag_wordnet(tagged_tokens):
    tag_map = {'j': wordnet.ADJ, 'v': wordnet.VERB, 'n': wordnet.NOUN, 'r': wordnet.ADV}
    new_tagged_tokens = [(word, tag_map.get(tag[0].lower(), wordnet.NOUN))
                            for word, tag in tagged_tokens]
    return new_tagged_tokens

# Create a function to process and clean texts 
def text_processing(review):
    
    #tokenize
    text = nltk.word_tokenize(review)
    
    #conver to lowercase 
    text = [t.lower() for t in text]
    
    # remove symbol
    text = [re.sub(r'^\d\w\s','',t) for t in text]
    
    #remove brackets
    text = [t.replace("(","").replace(")","") for t in text]
     
    # remove punctuation
    text = [t for t in text if t.isalnum()]
    
    # remove stopwords
    text  = [t for t in text if t not in  stop_words]
    
    #remove contractions
    text = [contractions.fix(t) for t in text]
    
    #Remove numbers
    text = [re.sub('\w*\d\w*', '', t) for t in text]
    
    #Remove ascii char
    text = [unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('utf-8', 'ignore') for t in text]
     
    #Remove empty tokens
    text = [t for t in text if t]
        
    #POS tagging
    text = nltk.pos_tag(text)
    text = pos_tag_wordnet(text)
    
    # Lemmatization
    lemma = WordNetLemmatizer()
    text = [lemma.lemmatize(t,tag) for t,tag in text]
    
    #Join tokens
    text = ' '.join(text)
    
    return text


topic_mapping = {0: 'Night Life', 1: 'Value for Money', 2: 'Service', 3: 'Location', 4: 'Room Amenities'}
    
def wordcloud(review_df, color, title):
    '''    
    INPUTS:
        reivew_df - dataframe, positive or negative reviews
        review_colname - column name, positive or negative review
        color - background color of worldcloud
        title - title of the wordcloud
    OUTPUT:
    Wordcloud visuazliation
    '''  
    text = review_df['Review'].to_string()
    #text_str = ' '.join(lemmatized_tokens(' '.join(text))) #call function "lemmatized_tokens"
    #text = review_df
    wordcloud = WordCloud(collocations = False,
                          background_color = color,
                          width=1600, 
                          height=800, 
                          margin=2,
                          min_font_size=20).generate(text)

    plt.figure(figsize = (10, 8))
    plt.imshow(wordcloud)
    #plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.figtext(.5,.8,title,fontsize = 20, ha='center')
    plt.show()
    st.pyplot()

# Create a function to analysis sentiments of reviews using TextBlob
def sentiment_analysis(df):
  
 #Create two new columns ‘Polarity’)
    df['TextBlob_Polarity'] = df['Review'].apply(lambda x:TextBlob(x).sentiment.polarity )
    
    def getAnalysis(score):
        if score <= -0.05:
            return 'Negative'
        elif score >= 0.15:
            return 'Positive'
        else:
            return 'Neutral'
     
    df['Sentiments'] = df['TextBlob_Polarity'].apply(getAnalysis )

    return  df

# Create a function to plot the countplots for each topic
def topic_sentiments(df, title):
    plt.figure(figsize = (12, 6))
    sns.countplot( x = 'Hotel_Name', data = df, hue = 'Sentiments', palette = 'Set2')
    plt.xticks(rotation=90, fontsize = 12)
    plt.title(title, fontsize = 18)
    plt.legend(loc = 2)


#Load model
model = pickle.load(open(r'C:\Users\yeosi\Documents\Python MAGES\07 DS106\Capstone\Streamlit\lsvc_pipe_model.pkl','rb'))


def main():
    global df
    global room_amen
    if 'df' not in st.session_state:
        st.session_state.df = df 
    # Create user interface on Streamlit
    st.title('ibis Budget Singapore')
    st.subheader('Topic Classification and Sentiment Analysis of Hotel Reviews')
    st.write('This app can be used to process hotel reviews and assign them to the most appropriate topic using supervised machine learning model. Sentiment analysis will then be done for the reviews to determine the overall sentiment of each topic.')

    # To upload CSV
    with st.sidebar:
        uploaded_file = st.file_uploader('Choose a file in .csv format', type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)	

    if st.checkbox('Show dataframe'):
        st.write('**The dataframe has been uploaded. The following are the columns:**')     
        if uploaded_file is not None:
            st.dataframe(df)
        else:
            pass

    # To Pre-process text and run model with sentiment analysis
    with st.sidebar:
        if st.button('Run Topic Allocation and Sentiment Analysis', key = '1'):
            df['Review'] = df['Review'].apply(text_processing)
            df['Review'] = df['Review'].apply(remove_custom_stopwords)
            df['Topic'] = model.predict(df['Review'])
            df['Topic'] = df['Topic'].map(topic_mapping)
            sentiment_analysis(df)
            # Create data subsets for each topic 

            room_amen = df.loc[df['Topic']== 'Room Amenities', (['Hotel_Name', 'Review', 'Sentiments'])]
            service = df.loc[df['Topic']== 'Service', (['Hotel_Name','Review', 'Sentiments'])]
            night_life = df.loc[df['Topic']== 'Night Life', (['Hotel_Name', 'Review', 'Sentiments'])]
            location = df.loc[df['Topic']== 'Location', (['Hotel_Name','Review', 'Sentiments'])]
            value = df.loc[df['Topic']== 'Value for Money', (['Hotel_Name', 'Review', 'Sentiments'])]

    with st.expander("Results"):
        st.dataframe(df)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    with st.expander("See Wordclouds"):
        st.write(wordcloud (room_amen, 'white', 'Room Amenities'))
        st.write(wordcloud (service, 'white', 'Service'))
        st.write(wordcloud (night_life, 'white', 'Night Life'))
        st.write(wordcloud (location, 'white', 'Location'))
        st.write(wordcloud (value, 'white', 'Value for Money'))
        #st.image(wc.to_image(), use_column_width=True)
    
    with st.expander("View Sentiments across Hotel Branches"):
        plt.figure(figsize = (15, 6))
        sns.countplot( x = 'Hotel_Name', data = df, hue = 'Sentiments', palette = 'Set2')
        plt.xticks(rotation=90, fontsize = 12)
        plt.title('Sentiments across Hotel Branches', fontsize = 18)
        st.pyplot()

    with st.expander("View Topics across Hotel Branches"):
        plt.figure(figsize = (15, 6))
        sns.countplot( x = 'Hotel_Name', data = df, hue = 'Topic', palette = 'Set2')
        plt.xticks(rotation=90, fontsize = 12)
        plt.title('Topics across Hotel Branches', fontsize = 18)
        st.pyplot()

    if 'df' not in st.session_state:
        st.session_state.df = df 

    #st.write(st.session_state.df)
    room_amen = st.session_state.df.loc[st.session_state.df['Topic']== 'Room Amenities', (['Hotel_Name', 'Review', 'Sentiments'])]
    service = st.session_state.df.loc[st.session_state.df['Topic']== 'Service', (['Hotel_Name','Review', 'Sentiments'])]
    night_life = st.session_state.df.loc[st.session_state.df['Topic']== 'Night Life', (['Hotel_Name', 'Review', 'Sentiments'])]
    location = st.session_state.df.loc[st.session_state.df['Topic']== 'Location', (['Hotel_Name','Review', 'Sentiments'])]
    value = st.session_state.df.loc[st.session_state.df['Topic']== 'Value for Money', (['Hotel_Name', 'Review', 'Sentiments'])]    

    with st.expander("See Sentiments for each Topic"):
        st.pyplot(topic_sentiments(room_amen, 'Room Amenities'))
        st.pyplot(topic_sentiments(service, 'Service'))
        st.pyplot(topic_sentiments(night_life, 'Night Life'))
        st.pyplot(topic_sentiments(location, 'Location'))
        st.pyplot(topic_sentiments(value, 'Value for Money'))

    
    
    
    
    
    
    
    with st.sidebar:
        with st.form("Select the topic to view the sentiment analysis"):
            option = st.radio('Topics',('Room Amenities', 'Service', 'Night Life', 'Location', 'Value for Money'))
            if 'df' not in st.session_state:
                st.session_state.df = df 
    # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")

    if submitted:
        if option == 'Room Amenities':
            room_amen = st.session_state.df.loc[st.session_state.df['Topic']== 'Room Amenities', (['Hotel_Name', 'Review', 'Sentiments'])]
            st.pyplot(topic_sentiments(room_amen, 'Room Amenities'))
        elif option == 'Service':
            service = st.session_state.df.loc[st.session_state.df['Topic']== 'Service', (['Hotel_Name','Review', 'Sentiments'])]
            st.pyplot(topic_sentiments(service, 'Service'))
        elif option == 'Night Life':
            night_life = st.session_state.df.loc[st.session_state.df['Topic']== 'Night Life', (['Hotel_Name', 'Review', 'Sentiments'])]
            st.pyplot(topic_sentiments(night_life, 'Night Life'))
        elif option == 'Location':
            location = st.session_state.df.loc[st.session_state.df['Topic']== 'Location', (['Hotel_Name','Review', 'Sentiments'])]
            st.pyplot(topic_sentiments(location, 'Location'))

        else:
            value = st.session_state.df.loc[st.session_state.df['Topic']== 'Value for Money', (['Hotel_Name', 'Review', 'Sentiments'])] 
            st.pyplot(topic_sentiments(value, 'Value for Money'))
    

if __name__=='__main__':
    main()
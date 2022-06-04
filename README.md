![alt_text](https://github.com/SiewPingYeo/Capstone-Project--End-to-End-ML-on-Topic-Classification-and-Sentiment-Analysis/blob/main/assets/hotelreview-lda-sent.png?raw=True)

#### Team Members
This is a team project with equal contribution with the following members:
- Yeo Siew Ping
- Edwin Wan
- Darren Hum

## End-to-End Machine Learning Project on Topic Modelling, Classifictaion and Sentiment Analysis

The aim of this project is to sieve out the underlying topics from unstructured text data (hotel reviews) followed by sentiment analysis of the reviews. The hotels that this project will focus on are budget hotels under the Ibis Singapore group. There are a total of 13 budget hotels under the Ibis Singapore brand. The reviews for the 13 budget hotels used in this project were scraped from TripAdvisor using Selenium (code can be found in repo).

The topic modelling and sentiment analysis of the data will be useful for hotel managers when it comes to identifying the pain points experienced by hotel guests, which increases the efficiency of their service recovery and improve the overall quality of the guests' experience. Good service quality and experience will usually lead to an increase in the number of repeat guests and attract new guests. This translates to higher revenue for the hotels.

The hotel reviews will first be pre-processed for an unsupervised machine learning technique - Latent Dirichlet Allocation (LDA). The topics will then be extracted from LDA, which will be used to label the reviews. Sentiment analysis is then performed with TextBlob to get a sense of the sentiments for each topic.

With the labeled reviews, supervised machine learning techniques will then be used to train a model to classify reviews into each of the topics. The model can then be used for deployment, where the hotel management can make use of the model to classify a large number of reviews at one go, followed by sentiment analysis of the reviews. This can potentially allow the hotel management to make informed and timely decisions for service recovery and hotel quality improvement, thus attracting more guests and increasing revenue in the long run.

To allow better visualisation of the situation, the team also created a Tableau dashboard for hotel management. 

**Flow of this project**

1. Import relevant libraries and dataset
2. Text pre-processing
3. Topic Modelling using Latent Dirichlet Allocation
4. Sentiment Analysis using TextBlob
5. Exploratory Data Analysis
6. Topic Classification - Model Training
7. Deployment on Streamlit and Data Visualisation on Tableau Dashboard

The final model is deployed on Streamlit and can be accessed over here [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/siewpingyeo/capstone-project--end-to-end-ml-on-topic-classification-and-sentiment-analysis/main/Streamlit/streamlit_app_capstone.py)

#### Technology Used
- Webscraping - Selenium
- Topic Modelling - Latent Dirichlet Allocation (LDA)
- Sentiment Analysis - TextBlob
- Topic Prediction - LinearSVC
- Deployment - Streamlit
- Dashboard - Tableau




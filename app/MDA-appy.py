import streamlit as st
from streamlit import components  

import pandas as pd 
import numpy as np 

import re
import string
import stanza
import stanza
from textblob import TextBlob 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint

import pyLDAvis
import pyLDAvis.sklearn

import matplotlib.pyplot as plt 

import seaborn as sns
sns.set_style('darkgrid')

from plotly.offline import plot 
import plotly.graph_objects as go 
import plotly.express as px 


# importing transcripts. NOTE: this one is not definitive version
df = pd.read_csv('/Users/danielbejarano/Documents/MSc. Information Management/2nd semester/Modern Data Analytics/consolidated-transcripts')
df.drop(columns = 'Unnamed: 0', inplace = True)
df.dropna(axis = 0, how = 'any', inplace = True)
df['Year'] = [int(i) for i in df['Year']]

st.image('/Users/danielbejarano/MDA/app/UN_General_Assembly_hall.jpeg')

## main header
st.title('Analysis of UN General Assemblies 1970-2018')

st.markdown(""" 
	In this application we dive deep into the contents of the UN General Assembly speeches for the past 50 years. 
	We first take a look at the frequently addressed issues, and then we rely on external data to see whether the topics addressed
	are also implemented in practice. 
	""")

# descriptive section --> including topic modelling here? 

st.write("""
	## Visualizing the speeches

	Here we can add other plots to describe the filings not in terms of content (can't think of any other metrics other
	than word count, which Khatchatur already did )
	""")

# setting up sidebar

st.sidebar.header('User Input Features')
unique_years = sorted(list(df['Year'].unique()), reverse = True)
unique_countries = list(df['Country'].unique())
unique_sessions = list(df['Session'].unique())
selected_years = st.sidebar.slider('Year', 1970, 2018, (1970, 2018))
selected_country = st.sidebar.multiselect('Country', unique_countries, unique_countries)
selected_session = st.sidebar.multiselect('Session', unique_sessions, unique_sessions)

filtered_df = df[(df.Country.isin(selected_country)) & (df.Session.isin(selected_session)) & (df.Year.isin(selected_years))]


# plotting transcript length histogram
fig = plt.figure(figsize = (10,6))
doc_lens = []
for i in filtered_df.Transcript.values:
	doc_lens.append(len(str(i)))
plt.hist(doc_lens, bins = 100)
plt.title('Distribution of transcript lengths')
plt.ylabel('Number of transcripts')
plt.xlabel('Transcript length')
sns.despine();
st.pyplot(fig)

st.write("""
	## Visualizing major themes addressed
	""")


vectorizer = CountVectorizer(analyzer = 'word',
	min_df = 3,
	stop_words = 'english',
	lowercase = True,
	token_pattern = '[a-zA-Z0-9]{3,}',
	max_features = 5000)

data_vectorized = vectorizer.fit_transform(filtered_df['Transcript'].values)
lda_model = LatentDirichletAllocation(n_components = 10, ## this is number of topics
	learning_method = 'online',
	random_state = 42,
	n_jobs = -1)

lda_output = lda_model.fit_transform(data_vectorized)
prepared_data = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds = 'tsne')
html_string = pyLDAvis.prepared_data_to_html(prepared_data)
components.v1.html(html_string, width = 1300, height = 800, scrolling = True)







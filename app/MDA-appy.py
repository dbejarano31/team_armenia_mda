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

from plotly.subplots import make_subplots
from plotly.offline import plot 
import plotly.graph_objects as go 
import plotly.express as px 

import boto3
import s3fs
import os

# before deploying the app I have to get the data from a remote server, it can't be stored in my local machine
# importing transcripts. NOTE: this one is not definitive version
df = pd.read_csv('https://s3grouparmenia.s3.eu-central-1.amazonaws.com/1405transcript_preprocessed_compact.csv')
df.dropna(axis = 0, how = 'any', inplace = True)
df['Year'] = [int(i) for i in df['Year']]

## setting up page width
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1200px;
        
    }}
    
</style>
""",
        unsafe_allow_html=True,
    )
## main header
st.title('Analysis of UN General Assemblies 1970-2018')

st.markdown(""" 
	In this application we dive deep into the contents of the UN General Assembly speeches for the past 50 years. 
	We first take a look at the frequently addressed issues, and then we rely on external data to see whether the topics addressed
	are also implemented in practice. 
	""")

# descriptive section

# setting up sidebar

st.sidebar.header('Filter the UN speeches')
unique_years = sorted(list(df['Year'].unique()), reverse = True)
unique_countries = list(df['Country'].unique())
unique_sessions = list(df['Session'].unique())
selected_years = st.sidebar.slider('Year', 1970, 2018, (1970, 2018))
country_expander = st.sidebar.beta_expander('Filter countries')
with country_expander:
	selected_country = st.multiselect('Country', unique_countries, unique_countries)
session_expander = st.sidebar.beta_expander('Filter sessions')
with session_expander:
	selected_session = st.multiselect('Session', unique_sessions, unique_sessions)

filtered_df = df[(df.Country.isin(selected_country)) & (df.Session.isin(selected_session)) & (df.Year.isin(selected_years))]


st.write("""
	## Visualizing major themes addressed
	Use the sidebar to the left to filter the transcripts per year, country or session!
	""")

# to retrieve n-grams. Default = 3
def get_top_k_ngram(corpus, k =10, n = 3):
	vec = CountVectorizer(ngram_range=(n,n), stop_words = 'english').fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word, sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
	words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
	return words_freq[:k]

n_gram_exp = st.beta_expander('Customize n-grams!')
with n_gram_exp:
	col1, col2 = st.beta_columns(2)
	with col1:
		n_gram = st.selectbox('Select n-grams to view', [2,3,4])
	with col2:
		top_k = st.selectbox('Select top options to view', [5,10,15,20])

if n_gram and top_k:
	common_words = get_top_k_ngram(filtered_df['Transcript'],int(top_k), int(n_gram))
	df1 = pd.DataFrame(common_words, columns = ['n-grams', 'count'])
	fig = go.Figure([go.Bar(x = df1['n-grams'], y = df1['count'], marker_color = 'lightskyblue')])

	if n_gram == 2:
		prefix = 'bi'
	elif n_gram == 3:
		prefix = 'tri'
	elif n_gram == 4:
		prefix = 'tetra'
	elif n_gram == 5:
		prefix == 'penta'

	fig.update_layout(title = go.layout.Title(text = 'Top {} {}-grams'.format(top_k, prefix)))
	st.plotly_chart(fig, use_container_width = True)

st.write("""
	### LDA topic modelling
	**Note:** The model shown below is pre-computed and hence the sidebar filters will not work. We freeze the LDA output
	in order to perform a trend analysis on the resulting topics. Otherwise, depending on the specified filters, the model may 
	output different topics which would then need to me named. 
	""")

fs = s3fs.S3FileSystem(anon = False)


@st.cache(ttl=600)
def read_file(filename):
	with fs.open(filename) as f:
		return f.read().decode('utf-8')

html_file = read_file('s3grouparmenia/lda.html')

#file = fs.open('https://s3grouparmenia.s3.eu-central-1.amazonaws.com/lda.html')
#html_file = file.read().decode('utf-8')

#s3 = boto3.client('s3')
#response = s3.get_object(Bucket = 's3grouparmenia', Key = 'lda.html')
#html_file = response['Body'].read()
#html_file = s3.download_file('s3grouparmenia', 'lda.html', 'lda.html')
#html_file = open('https://s3grouparmenia.s3.eu-central-1.amazonaws.com/lda.html', 'r', encoding = 'utf-8')
#source_code = html_file.read()
components.v1.html(html_file, width = 1200, height = 800, scrolling = True)


 
topic_nums = [1,2,3,4,5,6,7,8]
topic_names = ['Development of Africa', 'Human Rights', 'International Security', 'Nuclear Politics',
				'Economic Development', 'Israel-Palestine Conflict', 'World Peace', 'Sustainable Development' ]
topic_df = pd.DataFrame(
	{'Topic Number': topic_nums,
	'Topic Names': topic_names})

# setting up data for plotting topic trends
topic_df1 = pd.read_csv('https://s3grouparmenia.s3.eu-central-1.amazonaws.com/Transcripts_with_sentiment_topic')
res = {}
names = topic_df1['topic_name'].unique()
topic_dic = {key:None for key in names}
years = list(np.sort(topic_df1['Year_x'].unique()))
for i in list(topic_dic.keys()):
	topic_dic[i] = years

for topic in topic_dic.items():
	res[topic[0]] = {}
	for year in topic[1]:
		res[topic[0]][year] = len(topic_df1.loc[(topic_df1.topic_name == topic[0]) & (topic_df1.Year_x == year)])
df2 = pd.DataFrame(res)


st.write("""
	### Visualizing topic trends
	""")
topics = st.multiselect('Choose a topic name', topic_names, default = ['Development of Africa', 'Human Rights'])
topics_lc = [x.lower() for x in topics]
pd.options.plotting.backend = 'plotly'
filt_df = df2[topics_lc]
filt_df.index.name = 'Years'
fig = filt_df.plot()
#st.write(fig)

# this is a hack to center the table
col1, col2 = st.beta_columns((1,1))
with col1:
	st.write("""
 		#### Topic names

 		We manually label the topics based on the top 30 salient words shown above. The topic names are shown in the table below.  
		""")
	st.dataframe(topic_df)
with col2:
	st.write("""
		#### Topic trends """)
	st.write(fig)




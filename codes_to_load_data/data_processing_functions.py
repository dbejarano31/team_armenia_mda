from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import string

import re



class DataCleaners():

	def clean_text(text):
		text = re.sub(r'[0-9]', '', text)
		text = text.lower()
		text = re.sub(r'\[,!.*?\]', '', text)
		text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
		text = re.sub(r'\w*\d\w*', '', text)
		return text	

	def remove_stopwords(text):
		filtered = []
		stop_words = set(stopwords.words('english'))
		word_tokens = word_tokenize(text)
		for w in word_tokens:
			if w not in stop_words:
				filtered.append(w)
		filtered_doc = ' '.join(str(i) for i in filtered)
		return filtered_doc
    	
import os
import sys
import array
import codecs
import datetime
import time
from lxml import etree

import numpy as np
import pandas as pd
import scipy.sparse as sp

os.chdir('C:\\Users\songyifn\Desktop\mathoverflow_Recommender_System')

question_1=pd.read_csv('..\map_question.csv',header=None)
question_dict_1={} #this dictionary maps old question id (original id of the question stored in Posts.xml) to new user id (in user-question interactions matrix)
for i in range(question_1.shape[0]):
	question_dict_1[question_1.iloc[i,1]]=question_1.iloc[i,0]

answers_text=['']*question_1.shape[0] #a list container for storing all raw texts

with codecs.open('Raw_Data\Posts.xml', 'r', encoding='utf-8') as post_data:
	#go through every line of the XML file, if the id of the post is an element of the keys of question_dict_1, then the texts of this question can be mapped to list container to the position it is supposed to be at
	for line in post_data:
		try:
			datum = dict(etree.fromstring(line).items())
		except etree.XMLSyntaxError:
			continue

		is_question = datum.get('ParentId') is None
		if not is_question:
			continue

		question_id = int(datum['Id'])
		if question_id not in question_dict_1.keys():
			continue
		else:
			try:
				answers_text[question_dict_1[question_id]]+=' '+datum.get("Title")
			except TypeError:
				#for some questions the title is empty, in order for the dimension of .csv file that stores question texts to be correct, 'a' can be inserted into it ('a' is a stop word, and can be removed in subsequent steps)
				answers_text[question_dict_1[question_id]]+=' '+'a' 
			try:
				answers_text[question_dict_1[question_id]]+=' '+datum.get("Body")
			except TypeError:
				answers_text[question_dict_1[question_id]]+=' '+'a'

pd.DataFrame(np.array(answers_text)).to_csv("Questions_text.csv",header=False, index=False, encoding='utf-8')
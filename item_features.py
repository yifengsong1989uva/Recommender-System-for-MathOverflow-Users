import argparse
import array
import codecs
import datetime
import os
import subprocess
import sys
import time

from lxml import etree

import numpy as np
import pandas as pd

import progressbar

import requests

import scipy.sparse as sp

os.chdir('C:\\Users\songyifn\Desktop\mathoverflow_Recommender_System')

#user_1=pd.read_csv('user_ids_kept.csv',header=None)
question_1=pd.read_csv('..\map_question.csv',header=None)
print question_1.shape

question_dict_1={}
for i in range(question_1.shape[0]):
	question_dict_1[question_1.iloc[i,1]]=question_1.iloc[i,0]


os.chdir('C:\\Users\songyifn\Desktop\mathoverflow_Recommender_System')

answers_text=['']*question_1.shape[0]

with codecs.open('Raw_Data\Posts.xml', 'r', encoding='utf-8') as post_data:
	
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
				answers_text[question_dict_1[question_id]]+=' '+'a'
			try:
				answers_text[question_dict_1[question_id]]+=' '+datum.get("Body")
			except TypeError:
				answers_text[question_dict_1[question_id]]+=' '+'a'

"""
answers_by_user_new=[]
for u in user_dict:
	try:
		answers_by_user_new.append(answers_by_user[u])
	except KeyError:
		print u
		answers_by_user_new.append('')
"""
		
print len(answers_text)
pd.DataFrame(np.array(answers_text)).to_csv("Answers_text.csv",header=False, index=False, encoding='utf-8')
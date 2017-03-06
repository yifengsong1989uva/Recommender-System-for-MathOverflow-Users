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

user_1=pd.read_csv('user_ids_kept.csv',header=None)
user_2=pd.read_csv('..\map_user.csv',header=None)

user_dict_1={}
for i in range(user_1.shape[0]):
	user_dict_1[i]=user_1.iloc[i,0]
user_dict_2={}
for i in range(user_2.shape[0]):
	user_dict_2[user_2.iloc[i,0]]=user_2.iloc[i,1]

user_dict={}
for k in range(len(user_dict_1)):
	user_dict[user_dict_2[user_dict_1[k]]]=k

#pd.DataFrame(user_dict).to_csv('user_dict.csv',header=False,index=False)

os.chdir('C:\\Users\songyifn\Desktop\mathoverflow_Recommender_System')

about_me_by_user=['a']*4513
with codecs.open('Raw_Data\Users.xml', 'r', encoding='utf-8') as user_data:
	
	for line in user_data:
		try:
			datum = dict(etree.fromstring(line).items())
		except etree.XMLSyntaxError:
			continue

		try:
			user_id = int(datum['Id'])
			if user_id in user_dict.keys():
				try:
					about_me_by_user[user_dict[user_id]]+=' '+datum['AboutMe']
				except TypeError:
					about_me_by_user[user_dict[user_id]]+=' '
		except KeyError:
			continue

'''
answers_by_user_new=[]
for u in user_dict:
	try:
		answers_by_user_new.append(answers_by_user[u])
	except KeyError:
		print u
		answers_by_user_new.append('')
'''
		
print len(about_me_by_user)
pd.DataFrame(about_me_by_user).to_csv("User_aboutme.csv",header=False, index=False, encoding='utf-8')


'''
with codecs.open('Raw_Data\Users.xml', 'r', encoding='utf-8') as user_data:
	
	for line in post_data:
		try:
			datum = dict(etree.fromstring(line).items())
		except etree.XMLSyntaxError:
			continue


		try:
			user_id = int(datum['OwnerUserId'])
			answer = datum['Body']
			if user_id in answers_by_user.keys():
				answers_by_user[user_id]+=' '+answer
			else:
				answers_by_user[user_id]=answer
		except KeyError:
			continue
'''
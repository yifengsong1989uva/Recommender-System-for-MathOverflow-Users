import os
import sys
import codecs
from lxml import etree

import numpy as np
import pandas as pd

os.chdir('C:\\Users\songyifn\Desktop\mathoverflow_Recommender_System')

#build the same 0-4512 user id's mapping to old user ids (in the MathOverflow system) as in build_user_features_answers.py
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

about_me_by_user=['a']*user_1.shape[0] #'a' is the placeholder for the user ids which do not appear in Users.xml. It can be eliminated during the stop words removal step
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

print len(about_me_by_user) #verify if the number of aboutme texts is equal to the total number of users
pd.DataFrame(about_me_by_user).to_csv("User_aboutme.csv",header=False, index=False, encoding='utf-8')
import os
import sys
import codecs
from lxml import etree

import numpy as np
import pandas as pd

os.chdir('C:\\Users\songyifn\Desktop\mathoverflow_Recommender_System')

user_1=pd.read_csv('user_ids_kept.csv',header=None) #the 4,513 user ids that were kept in the case study
user_2=pd.read_csv('..\map_user.csv',header=None) #the mapping of new user ids to old user ids (user ids used in the MathOverflow system)

user_dict_1={} #the mapping of 0-4512 user ids to 0-11165 user ids in the entire pool
for i in range(user_1.shape[0]):
	user_dict_1[i]=user_1.iloc[i,0]
user_dict_2={} #the mapping of 0-11165 user ids to the old user ids
for i in range(user_2.shape[0]):
	user_dict_2[user_2.iloc[i,0]]=user_2.iloc[i,1]

user_dict=[] #combining user_dict_1 and user_dict_2 such that the mapping of 0-4512 user ids to the old user ids can be established: index is 0-4512 user ids, value is old user ids
for k in range(len(user_dict_1)):
	user_dict.append(user_dict_2[user_dict_1[k]])

pd.DataFrame(user_dict).to_csv('user_dict.csv',header=False,index=False) #save to file (reusable in subsequent steps)

answers_by_user={} #lookup table that maps 0-4512 user id to the corresponding collection of all answers provided by each user
with codecs.open('Raw_Data\Posts.xml', 'r', encoding='utf-8') as post_data:
	
	for line in post_data:
		try:
			datum = dict(etree.fromstring(line).items())
		except etree.XMLSyntaxError:
			continue

		is_answer = datum.get('ParentId') is not None
		if not is_answer:
			continue

		try:
			user_id = int(datum['OwnerUserId'])
			answer = datum['Body'] #the texts of the answers are stored in the 'Body' Field of each line
			if user_id in answers_by_user.keys():
				answers_by_user[user_id]+=' '+answer #concatenate the new answer to the previously stored answer texts for each user
			else:
				answers_by_user[user_id]=answer
		except KeyError:
			continue

answers_by_user_new=[] #the index of the list corresponds to 0-4512 user ids, the value corresponds to the collection of answer texts of user i (at index i)
for uid in user_dict:
	try:
		answers_by_user_new.append(answers_by_user[uid])
	except KeyError:
		answers_by_user_new.append('a')
		
#print len(answers_by_user_new) #verify if the number of answer collections is equal to the number of users
pd.DataFrame(answers_by_user_new).to_csv("User_answers.csv",header=False, index=False, encoding='utf-8')
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


def _process_post_tags(tags_string):

    return [x for x in tags_string.replace('<', ' ').replace('>', ' ').split(' ') if x]


def _read_interactions(post_data):

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
            question_id = int(datum['ParentId'])
            time_created = time.mktime(datetime.datetime
                                       .strptime(datum['CreationDate'],
                                                 '%Y-%m-%dT%H:%M:%S.%f')
                                       .timetuple())
        except KeyError:
            continue

        if user_id == -1:
            continue

        yield user_id, question_id, time_created


def _read_question_features(post_data):

    for line in post_data:

        try:
            datum = dict(etree.fromstring(line).items())
        except etree.XMLSyntaxError:
            continue

        is_question = datum.get('ParentId') is None

        if not is_question:
            continue

        question_id = int(datum['Id'])
        question_tags = _process_post_tags(datum.get('Tags', ''))

        yield question_id, question_tags


class IncrementalSparseMatrix(object):

    def __init__(self):

        self.row = array.array('i')
        self.col = array.array('i')
        self.data = array.array('f')

    def append(self, row, col, data):

        self.row.append(row)
        self.col.append(col)
        self.data.append(data)

    def tocoo(self, shape=None):

        row = np.array(self.row, dtype=np.int32)
        col = np.array(self.col, dtype=np.int32)
        data = np.array(self.data, dtype=np.float32)

        if shape is None:
            shape = (row.max() + 1, col.max() + 1)

        return sp.coo_matrix((data, (row, col)),
                             shape=shape)


class Dataset(object):

    def __init__(self):

        # Mappings
        self.user_mapping = {}
        self.question_mapping = {}
        self.tag_mapping = {}
		
        self.new_user_id_to_old = {}
        self.new_question_id_to_old = {}

        # Question features
        self.question_features = IncrementalSparseMatrix()

        # User-question matrix
        self.interactions = IncrementalSparseMatrix()

    def fit_features_matrix(self, question_features):

        self.question_features = IncrementalSparseMatrix()

        for (question_id, question_tags) in question_features:

            translated_question_id = (self.question_mapping
                                      .setdefault(question_id,
                                                  len(self.question_mapping)))
            self.new_question_id_to_old[translated_question_id] = question_id

            for tag in question_tags:
                translated_tag_id = (self.tag_mapping
                                     .setdefault(tag,
                                                 len(self.tag_mapping)))

                self.question_features.append(translated_question_id,
                                              translated_tag_id,
                                              1.0)

    def fit_interaction_matrix(self, interactions):

        self.interactions = IncrementalSparseMatrix()

        for (user_id, question_id, timestamp) in interactions:

            translated_user_id = (self.user_mapping
                                  .setdefault(user_id,
                                              len(self.user_mapping)))
											  
            self.new_user_id_to_old[translated_user_id] = user_id								  
											  
            translated_question_id = (self.question_mapping
                                      .setdefault(question_id,
                                                  len(self.question_mapping)))
            self.new_question_id_to_old[translated_question_id] = question_id
												  
            self.interactions.append(translated_user_id,
                                     translated_question_id,
                                     timestamp)

    def get_features_matrix(self):

        return self.question_features.tocoo(shape=(len(self.question_mapping),
                                                   len(self.tag_mapping)))

    def get_interaction_matrix(self):

        return self.interactions.tocoo(shape=(len(self.user_mapping),
                                              len(self.question_mapping)))

    def get_feature_labels(self):

        tags = sorted(self.tag_mapping.items(), key=lambda x: x[1])

        return np.array([x[0] for x in tags], dtype=np.dtype('|U50'))
		
    def get_new_user_id_to_old(self):
		
        return self.new_user_id_to_old
		
    def get_new_question_id_to_old(self):
		
        return self.new_question_id_to_old


def serialize_data(file_path, interactions, features, labels):

    arrays = {}

    for name, mat in (('interactions', interactions),
                      ('features', features)):
        arrays['{}_{}'.format(name, 'shape')] = (np.array(mat.shape,
                                                          dtype=np.int32)
                                                 .flatten()),
        arrays['{}_{}'.format(name, 'row')] = mat.row
        arrays['{}_{}'.format(name, 'col')] = mat.col
        arrays['{}_{}'.format(name, 'data')] = mat.data

    arrays['labels'] = labels

    np.savez_compressed(file_path, **arrays)
	
	


def read_data(data_path):
    """
    Construct a user-thread matrix, where a user interacts
    with a thread if they post an answer in it.
    """

    dataset = Dataset()

    with codecs.open(data_path, 'r', encoding='utf-8') as data_file:
        dataset.fit_features_matrix(_read_question_features(data_file))
	
    print dataset.get_features_matrix().shape[0]
    with codecs.open(data_path, 'r', encoding='utf-8') as data_file:
        dataset.fit_interaction_matrix(_read_interactions(data_file))

    question_features = dataset.get_features_matrix()
    interactions = dataset.get_interaction_matrix()
    feature_labels = dataset.get_feature_labels()
    map1 = dataset.get_new_user_id_to_old()
    map2 = dataset.get_new_question_id_to_old()

    assert question_features.shape[0] == interactions.shape[1]
    print question_features.shape[0]
    assert question_features.shape[1] == len(feature_labels)

    return interactions, question_features, feature_labels, map1, map2

	

if __name__ == '__main__':

    posts_path = 'C:\\Users\songyifn\Desktop\mathoverflow_Recommender_System\Raw_Data\Posts.xml'

    print('Reading data...')
    interactions, features, labels, map_user, map_question = read_data(posts_path)

    output_fname = 'stackexchange_{}.npz'.format('mathoverflow')
    pd.DataFrame.from_dict(map_user,orient='index').to_csv("map_user.csv",header=False)
    pd.DataFrame.from_dict(map_question,orient='index').to_csv("map_question.csv",header=False)
	
    print('Writing output...')
    output_path = os.path.join(os.path.dirname(posts_path), output_fname)
    serialize_data(output_path, interactions, features, labels)

    print('Done.')
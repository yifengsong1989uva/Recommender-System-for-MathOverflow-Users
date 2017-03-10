### Acknowledgement: This Python 2 script is created by editing/modifying the script available from this link: https://github.com/lyst/lightfm/blob/master/lightfm/datasets/stackexchange.py
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp



def load_mathoverflow_data(test_set_fraction=0.2,
						   min_training_interactions=1,
						   indicator_features=True, tag_features=False,
						   download_if_missing=True):
    """
    Load the mathoverflow data set into Python for building the recommender system models
	
	Arguments:
    ----------
    test_set_fraction: float, optional
        The fraction of the dataset used for testing. Splitting into the train and test set is done
        in a time-based fashion: all interactions before a certain time are in the train set and
        all interactions after that time are in the test set.
    min_training_interactions: int, optional
        Only include users with this amount of interactions in the training set, default value is 1.
    indicator_features: bool, optional
        Use an [n_items, n_items] identity matrix for item features. When True with genre_features,
        indicator and genre features are concatenated into a single feature matrix of shape
        [n_items, n_items + n_genres].

    Returns:
    -------
    train: sp.coo_matrix of shape [n_users, n_items]
         Contains training set interactions.
    test: sp.coo_matrix of shape [n_users, n_items]
         Contains testing set interactions.
    item_features: sp.csr_matrix of shape [n_items, n_item_features]
         Contains item features.
    item_feature_labels: np.array of strings of shape [n_item_features,]
         Labels of item features.
    """

    if not (indicator_features or tag_features):
        raise ValueError('At least one of item_indicator_features '
                         'or tag_features must be True')
    if not (0.0 < test_set_fraction < 1.0):
        raise ValueError('Test set fraction must be between 0 and 1')

    data = np.load('C:\\Users\songyifn\Desktop\mathoverflow_Recommender_System\Raw_Data\stackexchange_mathoverflow.npz')

    interactions = sp.coo_matrix((data['interactions_data'],
                                  (data['interactions_row'],
                                   data['interactions_col'])),
                                 shape=data['interactions_shape'].flatten())
    tag_features_mat = sp.coo_matrix((data['features_data'],
                                      (data['features_row'],
                                       data['features_col'])),
                                     shape=data['features_shape'].flatten())
    tag_labels = data['labels']

    test_cutoff_index = int(len(interactions.data) * (1.0 - test_set_fraction))
    test_cutoff_timestamp = np.sort(interactions.data)[test_cutoff_index]
    #split the training and testing part according to the their timestamps: all user-question interactions in the testing set occurred after the interactions in the training set
    in_train = interactions.data < test_cutoff_timestamp
    in_test = np.logical_not(in_train)

    train = sp.coo_matrix((np.ones(in_train.sum(), dtype=np.float32),
                           (interactions.row[in_train],
                            interactions.col[in_train])),
                          shape=interactions.shape)
    test = sp.coo_matrix((np.ones(in_test.sum(), dtype=np.float32),
                          (interactions.row[in_test],
                           interactions.col[in_test])),
                         shape=interactions.shape)

    if min_training_interactions > 0:
        include = np.squeeze(np.array(train.getnnz(axis=1))) > min_training_interactions #np.getnnz() return the counts of non-missing values in each row or column in the sparse matrix
        pd.DataFrame(np.where(include)[0]).to_csv("user_ids_kept.csv",index=False,header=False) #save all user ids that are kept in the training data (users with at least 2 total interactions)
        train = train.tocsr()[include].tocoo() #row index slicing
        test = test.tocsr()[include].tocoo()

    if indicator_features and not tag_features: #in this case, indicator_features argument is True, tag_features argument is False
        features = sp.identity(train.shape[1],
                               format='csr',
                               dtype=np.float32)
        labels = np.array(['question_id:{}'.format(x) for x in range(train.shape[1])])
    elif not indicator_features and tag_features: #in this case, indicator_features argument is False, tag_features argument is True 
        features = tag_features_mat.tocsr()
        labels = tag_labels
    else: #if both indicator features and tags features are True
        id_features = sp.identity(train.shape[1],
                                  format='csr',
                                  dtype=np.float32)
        features = sp.hstack([id_features, tag_features_mat]).tocsr()
        labels = np.concatenate([np.array(['question_id:{}'.format(x)
                                           for x in range(train.shape[1])]),
                                tag_labels])

    return {'train': train,
            'test': test,
			'entire_data': interactions, 
			#entire_data are the interactions data before being filtered accroding to the threshold for the total number of times a user has to interact with the questions; which includes all users who have at least one interaction with the question pool
            'item_features': features,
            'item_feature_labels': labels}
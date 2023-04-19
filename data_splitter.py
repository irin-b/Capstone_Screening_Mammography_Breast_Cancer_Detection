# importing required modules
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, StratifiedKFold


RANDOM_STATE = 42

dir_path = os.path.dirname(os.path.realpath(__file__))
METADATA = pd.read_csv(dir_path + '/' + 'train.csv') # path to metadata csv

class DataSplitter:

    def __init__(self, meta_df=METADATA, label_var='cancer', verbose=False):
            
        self.metadata = meta_df
        self.verbose = verbose
        self.train, self.test = self.get_train_test_split()
        self.trainset, self.calibset = self.get_train_calib_split()
        self.labels = self.get_labels(label_var)


    # aggregate cancer result at patient level for stratification
    def cancer_by_patient(self, meta_df=None, id_subset=None):
        
        if meta_df is None:
            # aggregate cancer result at patient level for stratification
            patients = self.metadata.groupby('patient_id')['cancer'].sum().reset_index()

        else:
            patients = meta_df.groupby('patient_id')['cancer'].sum().reset_index()
        
        patients['cancer_positive'] = np.where(patients.cancer>=1, 1, 0)

        if id_subset is not None:
          patients = patients[patients['patient_id'].isin(id_subset)]

        X = patients.drop(columns=['cancer_positive'])
        y = patients.cancer_positive

        return X, y


    # get a dictionary of patient_ids and their corresponding image_ids
    def get_image_id(self, X_df):

        img_ids = {}
        for patient_id in X_df.patient_id.unique():
            img_ids[patient_id] = self.metadata[self.metadata['patient_id']==patient_id].image_id.unique().tolist()

        return img_ids


    def get_train_img_id(self):

        img_ids = []
        for patient_id in self.train.keys():
            img_ids.extend(self.metadata[self.metadata['patient_id']==patient_id].image_id.unique().tolist())

        return img_ids
    

    def get_test_img_id(self):

        img_ids = []
        for patient_id in self.test.keys():
            img_ids.extend(self.metadata[self.metadata['patient_id']==patient_id].image_id.unique().tolist())

        return img_ids
    

    # train-test split >> constant throughout the project
    # hard-coded random state = 42 and proportion at 80:20
    def get_train_test_split(self):

        X, y = self.cancer_by_patient()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

        if self.verbose==True:
            print('Total patient_id in training set: ', X_train.patient_id.nunique())
            print('Total patient_id in test set: ', X_test.patient_id.nunique())

        train = self.get_image_id(X_train)
        test = self.get_image_id(X_test)
        
        if self.verbose==True:
            print('Total image_id in training set: ', sum([len(val) for key, val in train.items()]))
            print('Total image_id in test set: ', sum([len(val) for key, val in test.items()]))

        return train, test

    def get_train_calib_split(self, random_state=RANDOM_STATE):
        
        train_ids = self.train.keys()
        X, y = self.cancer_by_patient(id_subset=train_ids)
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.15, random_state=random_state, stratify=y)

        if self.verbose==True:
            print('Total patient_id in training set: ', X_train.patient_id.nunique())
            print('Total patient_id in calibration set: ', X_calib.patient_id.nunique())

        train = self.get_image_id(X_train)
        calib = self.get_image_id(X_calib)

        if self.verbose==True:
            print('Total image_id in training set: ', sum([len(val) for key, val in train.items()]))
            print('Total image_id in calibration set: ', sum([len(val) for key, val in calib.items()]))

        return train, calib


    # get a dictionary of image_ids and their corresponding target variable (label)
    def get_labels(self, label_var='cancer'):

        labels = {}
        for image_id in self.metadata.image_id.unique():
            labels[image_id] = self.metadata[self.metadata['image_id']==image_id][label_var]

        return labels      


    # get a list of image_ids from a set of patient_ids
    def extract_image_id(self, patient_ids):

        image_ids = []
        for patient_id in patient_ids:
            imgs = self.metadata[self.metadata['patient_id']==patient_id]['image_id'].to_list()
            image_ids.extend(imgs)

        return image_ids


    # split calibration set into training and validation sets
    def get_calib_split(self, random_state=RANDOM_STATE):

        calib_ids = self.calibset.keys()
        X, y = self.cancer_by_patient(id_subset=calib_ids)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)

        if self.verbose==True:
            print('Total patient_id in calibration training set: ', X_train.patient_id.nunique())
            print('Total patient_id in calibration validation set: ', X_valid.patient_id.nunique())

        train = self.get_image_id(X_train)
        valid = self.get_image_id(X_valid)

        if self.verbose==True:
            print('Total image_id in calibration training set: ', sum([len(val) for key, val in train.items()]))
            print('Total image_id in calibration validation set: ', sum([len(val) for key, val in valid.items()]))

        return train, valid


    # split data into cross-validation folds
    def get_cv(self, n_splits=5, random_state=RANDOM_STATE):

        patient_ids = self.trainset.keys()
        df = self.metadata[self.metadata.patient_id.isin(patient_ids)]
        
        X, y = self.cancer_by_patient(df)
        skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        
        if self.verbose==True:
            print('Splitting training set into {} stratified k-folds...'.format(skf.get_n_splits(X, y)))

        cv_patient = {}
        cv_img = {}

        for i, (train_indices, validate_indices) in enumerate(skf.split(X, y)):
            
            train_patients = X[X.index.isin(train_indices)]['patient_id'].to_list()
            validate_patients = X[X.index.isin(validate_indices)]['patient_id'].to_list()

            cv_patient[i] = {'train': train_patients, 'validate': validate_patients}

            train_img = self.extract_image_id(train_patients)
            validate_img = self.extract_image_id(validate_patients)
            
            cv_img[i] = {'train': train_img, 'validate': validate_img}

            if self.verbose==True:
                print('--Fold: ', i)
                print('----Total patient_id in training set: ', len(cv_patient[i]['train']))
                print('----Total image_id in training set: ', len(cv_img[i]['train']))

                print('----Total patient_id in validation set: ', len(cv_patient[i]['validate']))
                print('----Total image_id in validation set: ', len(cv_img[i]['validate']))

        return cv_patient, cv_img
    

    # split data into cross-validation folds
    def get_cv_train_img_by_class(self, cv_img):

        cv_img_class = {}
        for fold in range(len(cv_img)):

            train_img = cv_img[fold]['train']
            cancer_img = self.metadata[(self.metadata.image_id.isin(train_img))&(self.metadata.cancer==1)].image_id.unique()
            no_cancer_img = self.metadata[(self.metadata.image_id.isin(train_img))&(self.metadata.cancer==0)].image_id.unique()

            cv_img_class[fold] = {}
            cv_img_class[fold][0] = no_cancer_img
            cv_img_class[fold][1] = cancer_img

            if self.verbose==True:
                print('--Fold: ', fold)
                print('----Total image_id in training set with class 0: ', len(cv_img_class[fold][0]))
                print('----Total image_id in training set with class 1: ', len(cv_img_class[fold][1]))
        
        return cv_img_class





    
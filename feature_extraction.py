import numpy as np
from sklearn.decomposition import IncrementalPCA, MiniBatchNMF
import os
import pickle
import time

import data_loader as dl


EXTRACTOR_PATH = '' # directory to save trained models


# class to calculate and keep running mean and variance in batch training
# adapted from https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html

class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.var  = data.var(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newvar  = data.var(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.var  = m/(m+n)*self.var + n/(m+n)*newvar +\
                        m*n/(m+n)**2 * (tmp - newmean)**2

            self.nobservations += n





def train_feature_extractor(
        img_gen, model_name,
        n_components=None,
        init=None, beta_loss=None, alpha_W=None, l1_ratio=None,
        save_path=None, verbose=True):
    
    if model_name=='PCA':
        model = IncrementalPCA(n_components=n_components)
    elif model_name=='NMF':
        model = MiniBatchNMF(n_components=n_components, init=init, beta_loss=beta_loss, alpha_W=alpha_W, l1_ratio=l1_ratio)

    not verbose or print('Initialized {} with n_components = {}'.format(model_name, n_components))

    i = 1
    for batch in img_gen:
        not verbose or print('--Training batch ', i)
        model.partial_fit(batch)
        i+=1

    if save_path:
        with open(save_path,'wb') as f:
            pickle.dump(model, f)

    return model





def evaluate_feature_extractor(img_gen, trained_model, verbose=True):
    
    batch_stats = StatsRecorder()
    diff_stats = StatsRecorder()

    i = 1
    not verbose or print('Start evaluation...')

    for batch in img_gen:

        batch_stats.update(batch)

        not verbose or print('--Transforming batch ', i)
        X_transformed = trained_model.transform(batch)
            
        not verbose or print('---Reconstructing batch')
        X_reconstructed = trained_model.inverse_transform(X_transformed)

        diff_stats.update(batch - X_reconstructed)
        
        if i==1:
            diff2 = np.sum((batch - X_reconstructed)**2)
            x2 = np.sum(batch**2)

        else:
            diff2 = diff2 + np.sum((batch - X_reconstructed)**2)
            x2 = x2 + np.sum(batch**2)

        i+=1

    # generalized Kullback-Leibler divergence
    if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H))
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        
    not verbose or print('>> Calculating explained variance...')
    explained_var = np.mean(1 - (diff_stats.var/batch_stats.var))
    not verbose or print('--> ', explained_var)

    not verbose or print('>> Calculating reconstruction error...')
    reconstruction_err = diff2 / x2
    not verbose or print('--> ', reconstruction_err)

    return explained_var, reconstruction_err





def cv_train_feature_extractor(
        cv_img, patient_img_dict, batch_size, basepath, img_size, normalize, random_state,
        model_name,n_components, init=None, beta_loss=None, alpha_W=None, l1_ratio=None,
        from_numpy=True, extractor_path=EXTRACTOR_PATH, verbose=True, evaluate=False, return_none=False    
        ):

    models = {}
    training_time = {}
    explained_vars = {}
    reconstruction_errs = {}
    
    
    for fold in cv_img:
        
        not verbose or print('Fold: ', fold)
        savepath = extractor_path +  '/' + model_name + '_' + str(n_components) + '_' + str(random_state) + '_' + str(img_size[0]) + '_' +str(fold) + '.pkl'

        # check if already exist
        if os.path.exists(savepath):
            print('{} model trained on this cross-validation fold already exist'.format(model_name))
            print('-->', savepath)
            
            model = pickle.load(open(savepath, 'rb'))
            training_time[fold] = 0 
        
        else:

            train_img_ids = cv_img[fold]['train']  # list of image_id

            start = time.time()
                       
            train_img_gen = dl.ImgGenerator(
                list_IDs = train_img_ids,
                patient_img_dict = patient_img_dict,
                batch_size = batch_size,
                basepath = basepath,
                img_size = img_size,
                normalize = normalize,
                from_numpy = from_numpy
            )
                

            model = train_feature_extractor(
                img_gen = train_img_gen,
                model_name = model_name,
                n_components = n_components,
                init = init,
                beta_loss = beta_loss,
                alpha_W = alpha_W,
                l1_ratio = l1_ratio,
                save_path = savepath,
                verbose = verbose
                )
            
            stop = time.time()
            training_time[fold] = start - stop  

        models[fold] = model  
        not verbose or print('-->> Training time: ', (training_time[fold]))

        if evaluate:

            val_img_ids = cv_img[fold]['validate']  # list of image_id
            
            val_img_gen = dl.ImgGenerator(
                list_IDs = val_img_ids,
                patient_img_dict = patient_img_dict,
                batch_size = batch_size,
                basepath = basepath,
                img_size = img_size,
                normalize = normalize,
                from_numpy = from_numpy
            )

            explained_var, reconstruction_err = evaluate_feature_extractor(val_img_gen, model, verbose=verbose)

            explained_vars[fold] = explained_var
            reconstruction_errs[fold] = reconstruction_err

    if evaluate == True:
        return models, training_time, explained_vars, reconstruction_errs            
            
    if return_none: # ignored if evaluate = True
        return training_time
            
    return models, training_time




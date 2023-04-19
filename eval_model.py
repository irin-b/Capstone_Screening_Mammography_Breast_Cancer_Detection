import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, calibration

import pickle
import pydicom

import data_loader as dl
import run_model as rm

import matplotlib.pyplot as plt


# Set parameters
IMG_SIZE = (256, 256, 1) # if img_gen's feature_extractor == 'CNN', img_gen outputs 3 channels (duplicated from 1)
BATCH_SIZE = 32 # global batch size if multi-processing > local batch size = global batch size / processing cores
RANDOM_STATE = 42 # change to get different set of cross-validation folds, and image augmentations
BASEPATH = '' # DICOM file directory, structured like main_dir/paitient_id/imag_id.dcm
CORES = 4 # no. of processing cores

EXTRACTOR_PATH = ''


def vis_img_processing(dcm_path, numpy_path, img_id, img_size, labels, patient_img_dict, extractor_path):
    
    patient_id = [i for i in patient_img_dict if img_id in (patient_img_dict[i])][0] 
    dcm_file_path = dcm_path + '/' + str(patient_id) + '/' + str(img_id) + '.dcm'

    original = pydicom.dcmread(dcm_file_path)

    std_gen = dl.DataGenerator(
        list_IDs = [img_id],
        labels = labels,
        patient_img_dict = patient_img_dict,

        from_numpy = True,
        basepath = numpy_path,
        batch_size = 1,
        img_size = img_size,
        normalize = (-1, 1),
        feature_extractor=None,
        CNN_preprocess=None
        )
    
    std_img, std_label = std_gen[0]

    norm_gen = dl.DataGenerator(
        list_IDs = [img_id],
        labels = labels,
        patient_img_dict = patient_img_dict,

        from_numpy = True,
        basepath = numpy_path,
        batch_size = 1,
        img_size = img_size,
        normalize = (0, 1),
        feature_extractor = None,
        CNN_preprocess = None
        )
    
    norm_img, norm_label = norm_gen[0] 
    
    pca_path = extractor_path + '/PCA_50_42_' + str(img_size[0]) + '_0.pkl'
    pca = pickle.load(open(pca_path, 'rb'))

    pca_gen = dl.DataGenerator(
        list_IDs = [img_id],
        labels = labels,
        patient_img_dict = patient_img_dict,

        from_numpy = True,
        basepath = numpy_path,
        batch_size = 1,
        img_size = img_size,
        normalize = (0, 1),
        feature_extractor = pca,
        CNN_preprocess = None
        )
    
    pca_comp, pca_label = pca_gen[0]
    pca_img = pca.inverse_transform(pca_comp)
    
    nmf_path = extractor_path + '/NMF_26_42_' + str(img_size[0]) + '_0.pkl'
    nmf = pickle.load(open(nmf_path, 'rb'))

    nmf_gen = dl.DataGenerator(
        list_IDs = [img_id],
        labels = labels,
        patient_img_dict = patient_img_dict,

        from_numpy = True,
        basepath = numpy_path,
        batch_size = 1,
        img_size = img_size,
        normalize = (0, 1),
        feature_extractor = nmf,
        CNN_preprocess = None
        )
    
    nmf_comp, nmf_label = nmf_gen[0]
    nmf_img = nmf.inverse_transform(nmf_comp)

    cnn_gen = dl.DataGenerator(
        list_IDs = [img_id],
        labels = labels,
        patient_img_dict = patient_img_dict,

        from_numpy = True,
        basepath = numpy_path,
        batch_size = 1,
        img_size = img_size,
        normalize = (0, 1),
        feature_extractor = 'CNN',
        CNN_preprocess = tf.keras.applications.inception_v3.preprocess_input
        )
    
    cnn_img, cnn_label = cnn_gen[0]

    augment = rm.init_augment_layers(input_shape=img_size)

    augmented_img = [augment(cnn_img[0]) for _ in range(6)]

    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    axes[0, 0].imshow(original.pixel_array, cmap='gray')
    axes[0, 0].set_title('Original (label={})'.format(labels[img_id]))

    axes[0, 1].imshow(np.squeeze(std_img), cmap='gray')
    axes[0, 1].set_title('Standardized (label={})'.format(std_label[0]))

    axes[0, 2].imshow(np.squeeze(norm_img), cmap='gray')
    axes[0, 2].set_title('Normalized (label={})'.format(norm_label[0]))

    axes[1, 0].imshow(pca_img.reshape(img_size[0], img_size[1]), cmap='gray')
    axes[1, 0].set_title('PCA_reconstructed (label={})'.format(pca_label[0]))

    axes[1, 1].imshow(nmf_img.reshape(img_size[0], img_size[1]), cmap='gray')
    axes[1, 1].set_title('NMF_reconstructed (label={})'.format(nmf_label[0]))

    axes[1, 2].imshow(np.squeeze(cnn_img[0][:, :, 0]), cmap='gray')
    axes[1, 2].set_title('CNN_preprocessed (label={})'.format(cnn_label[0]))

    n = 0
    for i in [2, 3]:
        for j in range(3):
            axes[i, j].imshow(augmented_img[n][:, :, 0], cmap='gray')
            axes[i, j].set_title('CNN_augmented {}'.format(n))
            n = n +1


# plot training history of a model
def plot_history(history):
    
    metrics = ['loss', 'prc', 'precision', 'recall']
    metric_name = {'loss': 'Binary Cross Entropy Loss', 'prc':'PR AUC', 'precision': 'Precision', 'recall': 'Recall'}

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    n=0
    for i in range(2):
        for j in range(2):

            metric = metrics[n]
            val_metric = 'val_' + metric

            ax[i, j].plot(history.history[metric])
            ax[i, j].plot(history.history[val_metric])
            ax[i, j].set_title(metric_name[metric])
            ax[i, j].set_ylabel(metric)
            ax[i, j].set_xlabel('Epoch')
            ax[i, j].legend(['Train', 'Validate'], loc='upper left')

            n = n + 1

    fig.tight_layout(pad=1) 


# adapted from https://www.kaggle.com/code/sohier/probabilistic-f-score/comments#2143925
def proba_fscore(y_true, y_pred_proba, beta=1):
    
    y_positive_count = np.sum(y_true)

    # aggregated predicted probabiility of correct predictions
    proba_true_pos = np.sum(y_true*y_pred_proba)

    # aggregated predicted probabiility of incorrect predictions
    proba_false_pos = np.sum((1-y_true)*y_pred_proba)

    beta_squared = beta**2

    proba_precision = proba_false_pos / (proba_true_pos + proba_false_pos)
    proba_recall = proba_true_pos / y_positive_count

    if (proba_precision > 0 and proba_recall > 0):

        proba_fscore = (1 + beta_squared) * (proba_precision * proba_recall) / (beta_squared * proba_precision + proba_recall) 

        return proba_fscore
    else:
        return 0



# calculating metrics for each batch
def get_scores(y_true, y_pred_label, y_pred_proba):
    
    scores = {}

    scores['proba_f1-score'] = proba_fscore(y_true, y_pred_proba)
    
    scores['pr_auc'] = metrics.average_precision_score(y_true, y_pred_proba)
    try:
        scores['roc_auc'] = metrics.roc_auc_score(y_true, y_pred_proba)
    except:
        scores['roc_auc'] = 0.5
    
    scores['f1-score'] = metrics.f1_score(y_true, y_pred_label)
    scores['recall'] = metrics.recall_score(y_true, y_pred_label, zero_division=0)
    scores['precision'] = metrics.precision_score(y_true, y_pred_label, zero_division=0)
    scores['confusion_matrix'] = metrics.confusion_matrix(y_true, y_pred_label)
    scores['accuracy'] = metrics.accuracy_score(y_true, y_pred_label)

     # calculate stratified brier loss 
    pos_idx = np.nonzero(np.array(y_true)==1)
    scores['pos_brier_loss'] = metrics.brier_score_loss(np.array(y_true)[pos_idx], np.array(y_pred_proba)[pos_idx])

    neg_idx = np.nonzero(np.array(y_true)==0)
    scores['neg_brier_loss'] = metrics.brier_score_loss(np.array(y_true)[neg_idx], np.array(y_pred_proba)[neg_idx])

    return scores    


# calculate mean and standard deviation of metrics over all batches
def get_mean_scores(batch_scores):

    metrics = ['pr_auc', 'pos_brier_loss', 'neg_brier_loss','roc_auc', 'f1-score', 'recall', 'precision','confusion_matrix', 'accuracy']
    scores = {}
    for metric in metrics:
        scores[metric] = [batch_scores[batch][metric] for batch in batch_scores.keys()]

    mean_scores = {}                        
    for metric in metrics:
        mean_scores[metric] = {}
        mean_scores[metric]['mean'] = np.mean(scores[metric])
        mean_scores[metric]['std'] = np.std(scores[metric])
                            
    return mean_scores   



def get_predictions(model, X, pred_label=True, threshold=0.5):
    try:
        # scikit-learn model
        y_pred_proba = model.predict_proba(X)
    except:
        # Keras model
        y_pred_proba = model.predict(X)
    
    if pred_label:
        y_pred_label = np.where(y_pred_proba > threshold, 1, 0)
        return y_pred_proba, y_pred_label
    
    return y_pred_proba



# get predictions from generator
def get_aggregated_predictions(
        model, val_gen, pred_label=True, return_id=False,
        cnn=None, minibatch=False, minibatch_size=None,
        threshold=0.5, verbose=1):
    
    if pred_label:
        verbose!=1 or print('---Threshold: ', threshold)

    feature_extractor = val_gen.feature_extractor
    batch_size = val_gen.batch_size
    

    for i, data in enumerate(val_gen):

        if return_id:
            img_id, X_val, y_val = data
            if i==0:
                agg_img_id = img_id
            else:
                agg_img_id = np.concatenate((agg_img_id, img_id))
        else:
            X_val, y_val = data
        
        if (feature_extractor=='CNN') & (cnn is not None):
            verbose!=1 or print('--->>> Tranforming X with CNN...')

            if minibatch:
                minibatch_num = int(np.ceil(X_val.shape[0]/minibatch_size))
                idx_lower = 0
                for j in range(minibatch_num):

                    idx_upper = (j+1)*minibatch_size

                    if idx_upper > X_val.shape[0]:
                        x = X_val[idx_lower:]  
                    else:
                        x = X_val[idx_lower:idx_upper]

                    x = cnn(x, training=False)

                    if j== 0:
                        x_transformed = x
                    else:
                        x_transformed = np.vstack((x_transformed, x))  

                    idx_lower = idx_upper
                
                X_val = x_transformed
                
            else:
                X_val = cnn(X_val, training=False)


        if ((feature_extractor=='CNN') & (cnn is None))==False:
            X_val = X_val.reshape(batch_size, -1) # flatten array from IMG_SIZE shape
        
        verbose!=1 or print('---Getting predictions for batch ', i)
        if pred_label:
            y_pred_proba_batch, y_pred_label_batch = get_predictions(model, X_val, pred_label, threshold)

            if i==0:
                y_true = y_val
                y_pred_proba = y_pred_proba_batch
                y_pred_label = y_pred_label_batch
            else:
                y_true = np.concatenate((y_true, y_val))
                y_pred_proba = np.vstack((y_pred_proba, y_pred_proba_batch))
                y_pred_label = np.vstack((y_pred_label, y_pred_label_batch))

        else:
            y_pred_proba_batch = get_predictions(model, X_val, pred_label, threshold)

            if i==0:
                y_true = y_val
                y_pred_proba = y_pred_proba_batch
            else:
                y_true = np.concatenate((y_true, y_val))
                y_pred_proba = np.vstack((y_pred_proba, y_pred_proba_batch))


    verbose!=1 or print('Finish all batches!')
    if return_id:
        if pred_label:
            return agg_img_id, y_true, y_pred_proba, y_pred_label
        else:
            return agg_img_id, y_true, y_pred_proba
    
    elif pred_label:
        return y_true, y_pred_proba, y_pred_label
    
    else:
        return y_true, y_pred_proba



def load_trained_cnn(img_size, weight_path):

    inception = keras.applications.InceptionV3(
        input_shape=(img_size[0], img_size[1], 3),  # should have exactly 3 inputs channels because it is pre-trained on RBG images
        include_top=False
        )
    
    model = keras.Sequential([
        inception,
        keras.layers.Input(inception.output.shape[-3:]),
        keras.layers.GlobalAvgPool2D(),
        keras.layers.Dense(1, activation='sigmoid')
        ])
    
    model.load_weights(weight_path)
    model.trainable=False

    return model    
    


def get_prediction_from_pipeline(
    img_ids, label_img_dict, patient_img_dict,
    model_name, model_path,
    feature_extractor_name, extractor_path, n_components,
    img_size=IMG_SIZE, basepath=BASEPATH, batch_size=BATCH_SIZE,
    fold_range=range(5), cnn=None, CNN_preprocess=None, minibatch=False, minibatch_size=None,
    random_state=RANDOM_STATE, verbose=1
    ):
    
    img_id = {}
    y_true = {}
    y_pred_proba = {}

    for fold in fold_range:
        verbose!=1 or print('Fold: ', fold)

        if (feature_extractor_name=='PCA') | (feature_extractor_name=='NMF'):
            try:
                extractor_loadpath = extractor_path +  '/' + feature_extractor_name + '_' + str(n_components) + '_' + str(random_state) + '_' + str(img_size[0]) + '_' + str(fold) + '.pkl'
                feature_extractor = pickle.load(open(extractor_loadpath, 'rb'))
                verbose!=1 or print('---Loading {} feature extractor...'.format(feature_extractor_name))
                
            except:
                verbose!=1 or print('No feature extractor model pre-trained on the data found')
                return img_id, y_true, y_pred_proba
        
        elif (feature_extractor_name=='CNN_imagenet') | (feature_extractor_name=='CNN_rsna'):
            print('Feature extractor: ', feature_extractor_name)
            feature_extractor = 'CNN'
        
        else:
            feature_extractor = None
            verbose!=1 or print('Using no feature extractor')


        if model_name=='CNN':
            model = cnn
            cnn = None
            feature_extractor = 'CNN'
        else:
            try:
                model_loadpath = model_path +  '/' + feature_extractor_name + '_' + str(n_components) + '_' + model_name + '_' + str(random_state) + '_' + str(img_size[0]) + '_' + str(fold) + '.pkl'
                model = pickle.load(open(model_loadpath, 'rb'))
                verbose!=1 or print('---Loading {} model...'.format(model_name))
            except:    
                verbose!=1 or print('No feature extractor model pre-trained on the data found')
                return img_id, y_true, y_pred_proba
        

        val_gen = dl.DataGenerator(
            list_IDs = img_ids,
            labels = label_img_dict,
            patient_img_dict = patient_img_dict,
            
            from_numpy = True,
            basepath = basepath,
            batch_size = batch_size,
            img_size = img_size,
            
            shuffle = False,
            normalize = (0, 1),
            
            feature_extractor = feature_extractor,
            CNN_preprocess = CNN_preprocess,
            
            return_id = True
        )
        
        img_id_fold, y_true_fold, y_pred_proba_fold = get_aggregated_predictions(
            model = model,
            val_gen = val_gen,
            pred_label=False,
            return_id=True,
            cnn=cnn,
            minibatch=minibatch,
            minibatch_size=minibatch_size,
            verbose=verbose
        )
        
        img_id[fold] = img_id_fold
        y_true[fold] = y_true_fold
        y_pred_proba[fold] = y_pred_proba_fold
        
    return img_id, y_true, y_pred_proba



def make_prediction_df(img_ids, y_true, y_pred_proba, feature_extractor, model_name, img_size):

    for fold in range(len(img_ids)):

        if y_pred_proba[fold].shape[1]==2:
            y_pred_proba_fold = y_pred_proba[fold][:, 1]
        else:
            y_pred_proba_fold = np.squeeze(y_pred_proba[fold])

        print(img_ids[fold].shape, y_true[fold].shape, y_pred_proba_fold.shape)
        
        if feature_extractor is None:
            pred_col = model_name + '_' +  str(img_size[0]) + '_' +  str(fold)
        else:
            pred_col = feature_extractor + '_' + model_name + '_' +  str(img_size[0]) + '_' +  str(fold)
        
        cols = {'image_id': img_ids[fold], 'cancer': y_true[fold], pred_col: y_pred_proba_fold} 
        
        if fold==0:
            merged_df = pd.DataFrame(cols)
        else:
            df = pd.DataFrame(cols)
            merged_df = merged_df.merge(df, on=['image_id', 'cancer'], how='outer')

    
    if feature_extractor is None:
        filename = model_name + '_' +  str(img_size[0]) + '.csv'
    else:
        filename = feature_extractor + '_' + model_name + '_' +  str(img_size[0]) + '.csv'

    merged_df.to_csv(filename, index=False)
    print('Saved to ', filename)

    return merged_df



def get_best_threshold(y_true, y_pred_proba, verbose=False):
    # calculate pr curves
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred_proba)
    
    # to avoid division by zero
    precision_masked = np.ma.masked_where( (precision + recall)==0, precision)
    recall_masked = np.ma.masked_where( (precision + recall)==0, recall)

    # convert to f score
    fscore = (2 * precision_masked * recall_masked) / (precision_masked + recall_masked)
    
    # locate the index of the largest f score
    ix = np.nanargmax(fscore)
    threshold = thresholds[ix]
    
    not verbose or print('Best Threshold=%f, F-Score=%.3f' % (threshold, fscore[ix]))
    y_pred_label = np.where(y_pred_proba > threshold, 1, 0)
    
    scores = get_scores(y_true, y_pred_label, y_pred_proba)
    not verbose or print(scores)
    
    return threshold, scores



def get_folds_best_thresholds(pred_df, col_name, fold_range):

    y_true = pred_df.cancer

    best = {}
    for fold in fold_range:
        fold_col = col_name + '_' + str(fold)
        y_pred_proba = pred_df[fold_col]
        threshold, scores = get_best_threshold(y_true, y_pred_proba)

        best[fold] = {}
        best[fold]['threshold'] = threshold
        best[fold]['scores'] = scores
    
    # print(best)
    return best


def make_score_df(pred_df, col_name, fold_range):

    scores = get_folds_best_thresholds(pred_df, col_name, fold_range)
    df = []
    for fold in fold_range:
        fold_col = col_name + '_' + str(fold)
        cols = {'model': fold_col, 'threshold': scores[fold]['threshold']}

        for k, v in scores[fold]['scores'].items():
            cols[k] = [v]
            if k=='confusion_matrix':
                cols['tn'] = cols[k][0][0][0]
                cols['fp'] = cols[k][0][0][1]
                cols['fn'] = cols[k][0][1][0]
                cols['tp'] = cols[k][0][1][1]
        
        df.append(pd.DataFrame(cols))

    merged_df = pd.concat(df)

    return merged_df
    


# calculate metrics 
def run_evaluation(model, val_gen, cnn=None, minibatch=False, minibatch_size=128, threshold=0.5, aggregate=False, verbose=1):

    feature_extractor = val_gen.feature_extractor
    batch_size = val_gen.batch_size

    verbose!=1 or print('---Evaluating model')

    if aggregate:
        y_true, y_pred_proba, y_pred_label = get_aggregated_predictions(
            model = model,
            val_gen = val_gen,
            cnn = cnn,
            minibatch = minibatch,
            minibatch_size = minibatch_size,
            threshold = threshold,
            verbose = verbose
            )
        
        scores = get_scores(y_true, y_pred_label, y_pred_proba[:, 1])
    
    else:
        batch_scores = {}
        for i, (X_val, y_val) in enumerate(val_gen):
            
            if feature_extractor=='CNN':
                if minibatch:
                    minibatch_num = int(np.ceil(X_val.shape[0]/minibatch_size))
                    idx_lower = 0
                    for i in range(minibatch_num):

                        idx_upper = (i+1)*minibatch_size

                        if idx_upper > X_val.shape[0]:
                            x = X_val[idx_lower:]  
                        else:
                            x = X_val[idx_lower:idx_upper]

                        x = cnn(x, training=False)

                        if i== 0:
                            x_transformed = x
                        else:
                            x_transformed = np.vstack((x_transformed, x))  

                        idx_lower = idx_upper
                    
                    X_val = x_transformed
                    
                else:
                    X_val = cnn(X_val, training=False)

            X_val = X_val.reshape(batch_size, -1) # flatten array from IMG_SIZE shape
            # y_val = y_val[:, 1] # the output is one-hot encoded, select only second column which is for cancer=1
            
            verbose!=1 or print('------Getting predictions for batch ', i)
            y_pred_label = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)
            
            verbose!=1 or print('------Calculating scores of batch ', i)
            batch_score = get_scores(y_val, y_pred_label, y_pred_proba[:, -1])

            batch_scores[i] = batch_score

        scores = get_mean_scores(batch_scores)

    return scores


def vis_pr_curve(pred_df, col_name, fold_range=range(5)):

    y_true = pred_df.cancer

    fig, ax = plt.subplots(figsize=(10, 10))

    for fold in fold_range:
        fold_col = col_name + '_' + str(fold)
        y_pred_proba = pred_df[fold_col]

        # calculate pr curves
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred_proba)
        
        # to avoid division by zero
        precision_masked = np.ma.masked_where( (precision + recall)==0, precision)
        recall_masked = np.ma.masked_where( (precision + recall)==0, recall)

        # convert to f score
        fscore = (2 * precision_masked * recall_masked) / (precision_masked + recall_masked)
        
        # locate the index of the largest f score
        ix = np.nanargmax(fscore)
        threshold = thresholds[ix]

        metrics.PrecisionRecallDisplay.from_predictions(
            y_true = y_true,
            y_pred = y_pred_proba,
            pos_label = 1,
            name = 'Fold: {} Threshold: {:.2f} F1-score: {:.2f}'.format(fold, threshold, fscore[ix]),
            ax = ax
        )

        ax.scatter(recall[ix], precision[ix], marker='x', s=100)

    ax.grid()
    ax.set_title("Precision-Recall curve: {}".format(col_name))
    plt.legend(loc='upper right')
    


def vis_confusion_matrix(pred_df, col_name, fold_range=range(5)):

    scores = get_folds_best_thresholds(pred_df, col_name, fold_range)
    
    num_folds = max([i for i in fold_range]) + 1
    fig, axes = plt.subplots(1, num_folds, figsize=(5*num_folds, 5))
    
    for fold in fold_range:

        threshold = scores[fold]['threshold']
        precision = scores[fold]['scores']['precision']
        recall = scores[fold]['scores']['recall']
        cm = scores[fold]['scores']['confusion_matrix']  

        if fold_range==range(1):
            metrics.ConfusionMatrixDisplay(cm).plot(ax=axes, colorbar=False, text_kw={'fontsize': 16})
            axes.set_title('Threshold: {:.2f} Precision: {:.2f} Recall: {:.2f}'.format(threshold, precision, recall))
        
        else:
            metrics.ConfusionMatrixDisplay(cm).plot(ax=axes[fold], colorbar=False, text_kw={'fontsize': 16})
            axes[fold].set_title('Threshold: {:.2f} Precision: {:.2f} Recall: {:.2f}'.format(threshold, precision, recall))



# adapted from https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#calibration-curves
def vis_calibration_curve(pred_df, col_name, fold_range=range(5), n_bins=10, histogram=True, zoom_in=False):
    
    y_true = pred_df.cancer

    fig, ax = plt.subplots(figsize=(10, 10))

    calibration_displays = {}
    for fold in fold_range:
        fold_col = col_name + '_' + str(fold)
        y_pred_proba = pred_df[fold_col]

        # calculate stratified brier loss 
        pos_idx = np.nonzero(np.array(y_true)==1)
        pos_brier_loss = metrics.brier_score_loss(np.array(y_true)[pos_idx], np.array(y_pred_proba)[pos_idx])

        neg_idx = np.nonzero(np.array(y_true)==0)
        neg_brier_loss = metrics.brier_score_loss(np.array(y_true)[neg_idx], np.array(y_pred_proba)[neg_idx])

        display = calibration.CalibrationDisplay.from_predictions(
            y_true = y_true,
            y_prob = y_pred_proba,
            name = 'Fold: {} Brier loss (+): {:.4f} (-): {:.4f}'.format(fold, pos_brier_loss, neg_brier_loss),
            n_bins=n_bins, 
            pos_label=1, 
            ax=ax,
        )

        calibration_displays[fold] = display

    ax.grid()
    ax.set_title("Calibration curves: {}".format(col_name))
    plt.legend(loc='upper left')

    if histogram:
        fig, ax = plt.subplots(figsize=(10, 5))
        for fold in fold_range:
            ax.hist(calibration_displays[fold].y_prob, histtype='step')

        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Count')

    plt.tight_layout()


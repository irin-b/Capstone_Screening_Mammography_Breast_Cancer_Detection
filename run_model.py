import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingRandomSearchCV

import os
import pickle
import time

import data_loader as dl
import eval_model as ev


# Set parameters
IMG_SIZE = (256, 256, 1) # if img_gen's feature_extractor == 'CNN', img_gen outputs 3 channels (duplicated from 1)
BATCH_SIZE = 32 # global batch size if multi-processing > local batch size = global batch size / processing cores
RANDOM_STATE = 42 # change to get different set of cross-validation folds, and image augmentations
BASEPATH = '' # DICOM file directory, structured like main_dir/paitient_id/imag_id.dcm
CORES = 32 # no. of processing cores

EXTRACTOR_PATH = ''
PCA_N_COMPONENTS = 50 # pre-determined no. of components of PCA to use as feature extractor
NMF_N_COMPONENTS = 26 # pre-determined no. of components of NMF to use as feature extractor

CHECKPOINT_PATH = ''
if CHECKPOINT_PATH == '':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    CHECKPOINT_PATH = dir_path + '/' + 'checkpoint' # path to save model checkpoints

EPOCH = 100
LEARNING_RATE = 0.001
LOSS = tf.keras.losses.BinaryCrossentropy(),
OPTIMIZER = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

# image augmentation layers
def init_augment_layers(input_shape=IMG_SIZE):
    aug_layers = tf.keras.Sequential([
        keras.layers.RandomFlip(seed=RANDOM_STATE),
        keras.layers.RandomRotation(0.2, seed=RANDOM_STATE),
        keras.layers.RandomCrop(input_shape[0], input_shape[1], seed=RANDOM_STATE),
        keras.layers.RandomContrast((0.3, 0.7), seed=RANDOM_STATE)
        ])
    return aug_layers


def get_model(model_layers, input_shape=IMG_SIZE,
              augment_layers=True, base_model=None, trainable=False,
              loss_func=LOSS, optimizer=OPTIMIZER, metrics=METRICS):

    if augment_layers:
        if base_model:
            base_model.trainable=trainable     
            
            aug_layers = init_augment_layers(input_shape)

            # Design model
            model = tf.keras.Sequential([
                keras.layers.Input(shape=(input_shape[0], input_shape[1], 3)), 
                aug_layers,
                base_model,
                model_layers
            ])
    
        else:

            aug_layers = init_augment_layers(input_shape)

            # Design model
            model = tf.keras.Sequential([
                aug_layers,
                model_layers
                ])
            
    elif base_model:

        base_model.trainable=trainable

        model = tf.keras.Sequential([
            base_model,
            model_layers
            ])
    else:
        model = model_layers

    model.compile(
        optimizer = optimizer,
        loss = loss_func,
        metrics= metrics
    )

    return model



def make_or_restore_model(
        model_layers, input_shape=(IMG_SIZE[0], IMG_SIZE[1]),
        augment_layers=True, base_model=None, trainable=False,
        loss_func=LOSS, optimizer=OPTIMIZER, metrics=METRICS,
        checkpoint_path=CHECKPOINT_PATH
        ):
    
    print('Checkpoint path: ', checkpoint_path)
    # check if checkpoint folder exists if not create it
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        print('---Checkpoint folder created: ', checkpoint_path)
    else:
        print('---Checkpoint folder already exists')

    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_path + "/" + name for name in os.listdir(checkpoint_path)]

    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)

        return keras.models.load_model(latest_checkpoint)
    

    print("Creating a new model...")

    model = get_model(
        model_layers = model_layers,
        input_shape = input_shape,
        augment_layers = augment_layers,
        base_model = base_model,
        trainable = trainable,
        loss_func = loss_func,
        optimizer = optimizer,
        metrics = metrics
        )

    return model



def get_train_val_generators(
        train_img_ids, val_img_ids,
        label_img_dict, patient_img_dict,
        train_img_by_class=None, train_class_ratio=None, 
        from_numpy=False, basepath=BASEPATH, batch_size=BATCH_SIZE,
        img_size=IMG_SIZE, shuffle=True, normalize=(-1, 1),
        feature_extractor=None, CNN_preprocess=None
        ):
        
    # define DataGenerators for training split
    train_gen = dl.DataGenerator(
        list_IDs = train_img_ids,
        labels = label_img_dict,
        patient_img_dict = patient_img_dict,
        IDs_by_class = train_img_by_class,
        class_ratio = train_class_ratio,
        from_numpy = from_numpy,
        basepath = basepath,
        batch_size = batch_size,
        img_size = img_size,
        shuffle = shuffle,
        normalize = normalize,
        feature_extractor = feature_extractor,
        CNN_preprocess = CNN_preprocess,
        verbose = False
        )

    # define DataGenerators for validation split
    val_gen = dl.DataGenerator(
        list_IDs = val_img_ids,
        labels = label_img_dict,
        patient_img_dict = patient_img_dict,
        from_numpy = from_numpy,
        basepath = basepath,
        batch_size = batch_size,
        img_size = img_size,
        shuffle = shuffle,
        normalize = normalize,
        feature_extractor = feature_extractor,
        CNN_preprocess = CNN_preprocess,
        verbose = False
        )

    return train_gen, val_gen

def on_train_begin(self, logs={}):
    self.metrics = {}
    for metric in logs:
        self.metrics[metric] = []




def run_training_cnn(
        train_generator, val_generator,

        model_layers,
        input_shape=IMG_SIZE, 
        augment_layers=True, base_model=None, trainable=False,
        loss_func=LOSS, optimizer=OPTIMIZER, metrics=METRICS,

        checkpoint_path = CHECKPOINT_PATH,
        strategy=None, epochs=EPOCH,
        callbacks=None, class_weight=None,
        use_multiprocessing=True, workers=CORES,
        verbose=1, return_model=False
        ):
    
    #if not os.path.exists(CHECKPOINT_PATH):
    #    os.makedirs(CHECKPOINT_PATH)

    if strategy: # strategy for distributed training

        # Open a strategy scope and create/restore the model
        with strategy.scope():
            
            model_layers = model_layers
            augment_layers = augment_layers
            base_model = base_model
            loss_func = loss_func
            optimizer = optimizer
            metrics = metrics

            model = make_or_restore_model(
                model_layers = model_layers,
                input_shape = input_shape,
                augment_layers = augment_layers,
                base_model = base_model,
                trainable = trainable,
                loss_func = loss_func,
                optimizer = optimizer,
                metrics = metrics,
                checkpoint_path = checkpoint_path
                )
               

    else:
        model = make_or_restore_model(
                model_layers = model_layers,
                input_shape = input_shape,
                augment_layers = augment_layers,
                base_model = base_model,
                trainable = trainable,
                loss_func = loss_func,
                optimizer = optimizer,
                metrics = metrics,
                checkpoint_path = checkpoint_path
                )


    if callbacks is None:
        callbacks = [
            # This callback saves a SavedModel every epoch
            # We include the current epoch in the folder name.
             keras.callbacks.ModelCheckpoint(filepath=checkpoint_path + "/ckpt-{epoch}", save_freq="epoch")
    ]
        
    history = model.fit(
        train_generator,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_generator,
        class_weight=class_weight,
        use_multiprocessing = use_multiprocessing,
        workers = workers,
        verbose = verbose
    )

    if return_model==True:
        return model, history
    
    return  None, history 



# for training CNN model using cross-validation
def run_cv_training_cnn(
        model_layers,
        cv_img,
        label_img_dict,
        patient_img_dict,

        input_shape=IMG_SIZE,
        augment_layers=True, base_model=None, trainable=False,
        loss_func=LOSS, optimizer=OPTIMIZER, metrics=METRICS,

        train_img_by_class = None,
        train_class_ratio = None,

        from_numpy = False,
        img_basepath = BASEPATH,
        batch_size = BATCH_SIZE, 
        img_size = IMG_SIZE,
        shuffle = True,
        normalize = (-1, 1),

        feature_extractor_name = None,
        CNN_preprocess=None,
        n_components = None,
        extractor_path = EXTRACTOR_PATH,
        random_state = RANDOM_STATE,

        checkpoint_path = CHECKPOINT_PATH,
        strategy = None,
        epoch = EPOCH,
        callbacks = None,
        class_weight=None,
        use_multiprocessing = True,
        workers = CORES,
        verbose = 1,

        return_model = False
        ):

    history = {}
    if return_model==True:
        models = {}

    for fold in cv_img:
        
        verbose!=1 or print('Fold: ', fold)
        if (feature_extractor_name == 'PCA') | (feature_extractor_name == 'NMF'):
            loadpath = extractor_path +  '/' + feature_extractor_name + '_' + str(n_components) + '_' + str(random_state) + '_' + str(img_size[0]) + '_' +str(fold) + '.pkl'
            try:
                verbose!=1 or print('---Loading feature extractor...')
                feature_extractor = pickle.load(open(loadpath, 'rb'))
            except:
                print('No feature extractor model pre-trained on the data found')
        elif feature_extractor_name == 'CNN':
            feature_extractor = 'CNN'
            verbose!=1 or print('---Feature extractor: Pre-trained CNN')
        else:
            feature_extractor = None
            verbose!=1 or print('---No feature extractor')


        verbose!=1 or print('---Creating training and validation data generators...')

        train_gen, val_gen = get_train_val_generators(
            train_img_ids = cv_img[fold]['train'],
            val_img_ids = cv_img[fold]['validate'],
            label_img_dict = label_img_dict,
            patient_img_dict = patient_img_dict,
            train_img_by_class = train_img_by_class,
            train_class_ratio = train_class_ratio,
            from_numpy = from_numpy,
            basepath = img_basepath,
            batch_size = batch_size,
            img_size = img_size,
            shuffle = shuffle,
            normalize = normalize,
            feature_extractor = feature_extractor,
            CNN_preprocess = CNN_preprocess
        )


        verbose!=1 or print('---Start runing model...')
        
        fold_model, fold_history = run_training_cnn(
            strategy = strategy,
            train_generator = train_gen,
            val_generator = val_gen,

            model_layers = model_layers,
            input_shape = input_shape,
            augment_layers = augment_layers,
            base_model = base_model,
            trainable = trainable,
            loss_func = loss_func,
            optimizer = optimizer,
            metrics = metrics,

            checkpoint_path = checkpoint_path,
            callbacks = callbacks,
            epochs = epoch,
            class_weight=class_weight,
            use_multiprocessing = use_multiprocessing,
            workers = workers,
            verbose = verbose,
            return_model=return_model
            )
        
        verbose!=1 or print('>>> Finish training model!')
        history[fold] = fold_history
        
        if return_model==True:
            models[fold] = fold_model
    
    if return_model==True:
        return models, history
    
    return None, history



# function for running hyperparameter search
def hp_search(X, y, pipeline, param_grid, random_state, min_resources, verbose=2):
    
    grid = HalvingRandomSearchCV(
    pipeline,
    cv = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True),
    param_distributions = param_grid,
    
    factor = 5, # 1/5 candidates selected in each iteration
    aggressive_elimination = True,
    n_candidates = 'exhaust',
    min_resources = min_resources,
    scoring = metrics.make_scorer(metrics.average_precision_score, needs_proba=True),
    refit = False,
    
    n_jobs = -1,
    random_state = 42,
    verbose = verbose
    )
    
    grid.fit(X, y)
    
    print('Best hyperparameters: ', grid.best_params_)
    print('Best average precision score: ', grid.best_score_)
    
    return grid



# train model from generator
def run_training(model, model_name, train_gen, cnn=None, minibatch=False, minibatch_size=128, verbose=1):
    
    feature_extractor = train_gen.feature_extractor
    batch_size = train_gen.batch_size
    
    verbose!=1 or print('---Start training model')
    start = time.time()

    for i, (X_train, y_train) in enumerate(train_gen):

        verbose!=1 or print('------Training batch ', i)

        if feature_extractor=='CNN':
            verbose!=1 or print('------>>> Tranforming X with CNN...')
            if minibatch:
                minibatch_num = int(np.ceil(X_train.shape[0]/minibatch_size))
                idx_lower = 0
                for i in range(minibatch_num):

                    idx_upper = (i+1)*minibatch_size

                    if idx_upper > X_train.shape[0]:
                        x = X_train[idx_lower:]  
                    else:
                        x = X_train[idx_lower:idx_upper]

                    x = cnn(x, training=False)

                    if i== 0:
                        x_transformed = x
                    else:
                        x_transformed = np.vstack((x_transformed, x))  

                    idx_lower = idx_upper
                
                X_train = x_transformed

            else:
                X_train = cnn(X_train, training=False)
                

        # verbose!=1 or print('X shape: ', X_train.shape)
        X_train = X_train.reshape(batch_size, -1) # flatten array from to 2 dimension  
        # verbose!=1 or print('X flattened shape: ', X_train.shape)

        if model_name == 'rfc':
            if i>0:
                model.n_estimators = model.n_estimators +1          
                verbose!=1 or print('------Current n_estimators: ', model.n_estimators)
    
        
        model.fit(X_train, y_train)

    stop = time.time()
    verbose!=1 or print('>>> Training time: ', (stop - start))

    return model



# for training logistic regression and random forest classifier using cross-validation
def run_cv_training(    
        model_name, 
        model_params,
        
        cv_img,
        label_img_dict,
        patient_img_dict,
    
        train_img_by_class = None,
        train_class_ratio = None,

        from_numpy = False,
        img_basepath = BASEPATH,
        batch_size = BATCH_SIZE, 
        img_size = IMG_SIZE,
        shuffle = True,
        normalize = (-1, 1),

        feature_extractor_name = None,
        CNN_preprocess=None,
        n_components = None,
        extractor_path = EXTRACTOR_PATH,
        minibatch = False,
        minibatch_size = 128,
        random_state = RANDOM_STATE,
 
        threshold = 0.5,
        aggregate = False,
        verbose = 1,
        savepath = EXTRACTOR_PATH,
        return_model = False
        ):

    history = {}
    if return_model==True:
        models = {}

    for fold in cv_img:
        
        not verbose or print('Fold: ', fold)

        model_savepath = savepath +  '/' + feature_extractor_name + '_' + str(n_components) + '_' + model_name + '_' + str(random_state) + '_' + str(img_size[0]) + '_' + str(fold) + '.pkl'

        # check if already exist
        if os.path.exists(model_savepath):
            print('{} model trained on this cross-validation fold already exist'.format(model_name))
            print('-->', model_savepath)
            
            #model = pickle.load(open(savepath, 'rb'))
            continue


        if (feature_extractor_name == 'PCA') | (feature_extractor_name == 'NMF'):
            loadpath = extractor_path +  '/' + feature_extractor_name + '_' + str(n_components) + '_' + str(random_state) + '_' + str(img_size[0]) + '_' +str(fold) + '.pkl'
            try:
                verbose!=1 or print('---Loading feature extractor...')
                feature_extractor = pickle.load(open(loadpath, 'rb'))
                cnn = None

            except:
                print('No feature extractor model pre-trained on the data found')

                
        elif feature_extractor_name == 'CNN_imagenet':
            feature_extractor = 'CNN'
            verbose!=1 or print('---Feature extractor: CNN with ImageNet weights')            

            cnn = keras.applications.InceptionV3(   
                weights='imagenet',   # load weights pre-trained on ImageNet. 
                input_shape=(img_size[0], img_size[1], 3),  # should have exactly 3 inputs channels because it is pre-trained on RBG images
                include_top=False
            )

            
        elif feature_extractor_name == 'CNN_rsna':
            feature_extractor = 'CNN'
            verbose!=1 or print('---Feature extractor: CNN pre-trained on mammogram images')
            
            loadpath = extractor_path +  '/' + 'inceptionv3_512_' + str(random_state)
            
            cnn = keras.models.load_model(loadpath)

            
        else:
            feature_extractor = None
            verbose!=1 or print('---No feature extractor')


        verbose!=1 or print('---Creating training and validation data generators...')
 
        train_gen, val_gen = get_train_val_generators(
            train_img_ids = cv_img[fold]['train'],
            val_img_ids = cv_img[fold]['validate'],
            label_img_dict = label_img_dict,
            patient_img_dict = patient_img_dict,
            train_img_by_class = train_img_by_class,
            train_class_ratio = train_class_ratio,
            from_numpy = from_numpy,
            basepath = img_basepath,
            batch_size = batch_size,
            img_size = img_size,
            shuffle = shuffle,
            normalize = normalize,
            feature_extractor = feature_extractor,
            CNN_preprocess = CNN_preprocess
        )

        
        if model_name  == 'logit':
            
            verbose!=1 or print('---Initializing logistic regression model...')
            if model_params['penalty']=='elasticnet':
                l1_ratio = 0.5
            else:
                l1_ratio = None

            model = LogisticRegression(
                C = model_params['C'],
                tol = model_params['tol'],
                penalty = model_params['penalty'],
                class_weight=model_params['class_weight'],
                l1_ratio = l1_ratio,

                warm_start = True,
                solver = 'saga',
                max_iter = 10000,
                random_state = 42,
            )
            
        elif model_name == 'rfc':
            
            verbose!=1 or print('---Initializing random forest classifier...')
            model = RandomForestClassifier( 
                criterion = model_params['criterion'],
                max_samples = model_params['max_samples'],
                max_features = model_params['max_features'],
                class_weight=model_params['class_weight'],
                
                warm_start=True, # enable incremental learning
                n_estimators=1, # starting, +1 every batch
                max_depth=5, # control for complexity
                bootstrap = True,
                
                min_samples_split=3,
                min_samples_leaf=2,
                max_leaf_nodes=None,
                
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=1
            )
            
        
        model = run_training(
            model = model,
            model_name = model_name,
            train_gen = train_gen,
            cnn = cnn,
            minibatch = minibatch,
            minibatch_size = minibatch_size,
            verbose = verbose
            )
        
        history[fold] = ev.run_evaluation(
            model = model,
            val_gen = val_gen,
            cnn = cnn,
            minibatch = minibatch,
            minibatch_size = minibatch_size,
            threshold = threshold,
            aggregate = aggregate,
            verbose = verbose
            )

        if return_model==True:
            models[fold] = model


        if verbose==1:
            print('>>> Finish training model!')
            print('>>> Mean scores over batches: ')
            print(history[fold])
            print('---Saving model...')

        with open(model_savepath,'wb') as f:
            pickle.dump(model, f)

    if return_model==True:
        return models, history
    
    return None, history

import numpy as np
import tensorflow as tf
import keras
# from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# from sklearn.pipeline import Pipeline
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

import data_splitter as ds
import data_loader as dl
import feature_extraction as fex


# Set parameters
IMG_SIZE = (32, 32, 1) # if img_gen's feature_extractor == 'CNN', img_gen outputs 3 channels (duplicated from 1)
BATCH_SIZE = 8 # global batch size if multi-processing > local batch size = global batch size / processing cores
RANDOM_STATE = 42 # change to get different set of cross-validation folds, and image augmentations
BASEPATH = '' # DICOM file directory, structured like main_dir/paitient_id/imag_id.dcm
CORES = 4 # no. of processing cores

EXTRACTOR_PATH = ''
PCA_N_COMPONENTS = 15 # pre-determined no. of components of PCA to use as feature extractor
NMF_N_COMPONENTS = 15 # pre-determined no. of components of NMF to use as feature extractor

CHECKPOINT_PATH = ''
if CHECKPOINT_PATH == '':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    CHECKPOINT_PATH = dir_path + '/' + 'checkpoint' # path to save model checkpoints

EPOCH = 1
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
        from_numpy=False, basepath=BASEPATH, batch_size=BATCH_SIZE,
        img_size=IMG_SIZE, n_classes=2, shuffle=True, normalize=(-1, 1),
        feature_extractor=None, CNN_preprocess=None
        ):
        
    # define DataGenerators for training split
    train_gen = dl.DataGenerator(
        list_IDs = train_img_ids,
        labels = label_img_dict,
        patient_img_dict = patient_img_dict,
        from_numpy = from_numpy,
        basepath = basepath,
        batch_size = batch_size,
        img_size = img_size,
        n_classes = n_classes,
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
        n_classes = n_classes,
        shuffle = shuffle,
        normalize = normalize,
        feature_extractor = feature_extractor,
        CNN_preprocess = CNN_preprocess,
        verbose = False
        )

    return train_gen, val_gen



def run_training_cnn(
        train_generator, val_generator,

        model_layers,
        input_shape=IMG_SIZE, 
        augment_layers=True, base_model=None, trainable=False,
        loss_func=LOSS, optimizer=OPTIMIZER, metrics=METRICS,

        checkpoint_path = CHECKPOINT_PATH,
        strategy=None, epochs=EPOCH,
        callbacks=None, verbose=1,
        use_multiprocessing=True, workers=CORES,
        return_model=False
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
        use_multiprocessing = use_multiprocessing,
        workers = workers,
        verbose = verbose
    )

    if return_model==True:
        return model, history
    
    return  history 



# for training CNN model using cross-validation
def run_cv_training_cnn(
        model_layers,
        cv_img,
        label_img_dict,
        patient_img_dict,

        input_shape=IMG_SIZE,
        augment_layers=True, base_model=None, trainable=False,
        loss_func=LOSS, optimizer=OPTIMIZER, metrics=METRICS,
      
        from_numpy = False,
        basepath = BASEPATH,
        batch_size = BATCH_SIZE, 
        img_size = IMG_SIZE,
        n_classes = 2,
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
        use_multiprocessing = True,
        workers = CORES,
        verbose = 1,

        return_none = False
        ):

    history = {}
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
            from_numpy = from_numpy,
            basepath = basepath,
            batch_size = batch_size,
            img_size = img_size,
            n_classes = n_classes,
            shuffle = shuffle,
            normalize = normalize,
            feature_extractor = feature_extractor,
            CNN_preprocess = CNN_preprocess
        )


        verbose!=1 or print('---Start runing model...')
        
        fold_history = run_training_cnn(
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
            use_multiprocessing = use_multiprocessing,
            workers = workers,
            verbose = verbose
            )
        
        verbose!=1 or print('>>> Finish training model!')
        history[fold] = fold_history
    
    if return_none:
        return None
    
    return history
  


# calculating metrics for each batch
def get_scores(y_true, y_pred_label, y_pred_proba):
    scores = {}
    scores['pr_auc'] = metrics.average_precision_score(y_true, y_pred_proba)
    scores['brier_loss'] = metrics.brier_score_loss(y_true, y_pred_proba)
    scores['roc_auc'] = metrics.roc_auc_score(y_true, y_pred_proba)
    
    scores['f1-score'] = metrics.f1_score(y_true, y_pred_label)
    scores['recall'] = metrics.recall_score(y_true, y_pred_label)
    scores['precision'] = metrics.precision_score(y_true, y_pred_label)
    scores['confusion_matrix'] = metrics.confusion_matrix(y_true, y_pred_label)
    scores['accuracy'] = metrics.accuracy_score(y_true, y_pred_label)
    
    return scores    


# calculate mean and standard deviation of metrics over all batches
def get_mean_scores(batch_scores):

    metrics = ['pr_auc', 'brier_loss', 'roc_auc', 'f1-score', 'recall', 'precision','confusion_matrix', 'accuracy']
    scores = {}
    for metric in metrics:
        scores[metric] = [batch_scores[batch][metric] for batch in batch_scores.keys()]

    mean_scores = {}                        
    for metric in metrics:
        mean_scores[metric] = {}
        mean_scores[metric]['mean'] = np.mean(scores[metric])
        mean_scores[metric]['std'] = np.std(scores[metric])
                            
    return mean_scores   


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




# for training logistic regression and random forest classifier using cross-validation
def run_cv_training(    
        model_name, 
        model_params,
        
        cv_img,
        label_img_dict,
        patient_img_dict,
    
        from_numpy = False,
        basepath = BASEPATH,
        batch_size = BATCH_SIZE, 
        img_size = IMG_SIZE,
        n_classes = 2,
        shuffle = True,
        normalize = (-1, 1),

        feature_extractor_name = None,
        CNN_preprocess=None,
        n_components = None,
        model_path = EXTRACTOR_PATH,
        random_state = RANDOM_STATE,
 
        verbose = 1,
        return_none = False
        ):

    history = {}
    
    for fold in cv_img:
        
        not verbose or print('Fold: ', fold)

        savepath = model_path +  '/' + feature_extractor_name + '_' + str(n_components) + '_' + model_name + '_' + str(random_state) + '_' + str(img_size[0]) + '_' + str(fold) + '.pkl'

        # check if already exist
        if os.path.exists(savepath):
            print('{} model trained on this cross-validation fold already exist'.format(model_name))
            print('-->', savepath)
            
            #model = pickle.load(open(savepath, 'rb'))
            continue


        if (feature_extractor_name == 'PCA') | (feature_extractor_name == 'NMF'):
            loadpath = model_path +  '/' + feature_extractor_name + '_' + str(n_components) + '_' + str(random_state) + '_' + str(img_size[0]) + '_' +str(fold) + '.pkl'
            try:
                verbose!=1 or print('---Loading feature extractor...')
                feature_extractor = pickle.load(open(loadpath, 'rb'))
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
            
            loadpath = model_path +  '/' + 'inceptionv3_512_' + str(random_state)
            
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
            from_numpy = from_numpy,
            basepath = basepath,
            batch_size = batch_size,
            img_size = img_size,
            n_classes = n_classes,
            shuffle = shuffle,
            normalize = normalize,
            feature_extractor = feature_extractor,
            CNN_preprocess = CNN_preprocess
        )

        
        if model_name  == 'logit':
            
            verbose!=1 or print('---Initializing logistic regression model...')
            model = LogisticRegression(
                C = model_params['C'],
                tol = model_params['tol'],
                penalty = model_params['penalty'],
                
                warm_start = True,
                solver = 'saga',
                max_iter = 10000,
                # l1_ratio = 0.5,
                random_state = 42
            )
            
        elif model_name == 'rfc':
            
            verbose!=1 or print('---Initializing random forest classifier...')
            model = RandomForestClassifier( 
                criterion = model_params['criterion'],
                max_samples = model_params['max_samples'],
                max_features = model_params['max_features'],
                
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
            
        
        verbose!=1 or print('---Start training model')
        for i, (X_train, y_train) in enumerate(train_gen):

            verbose!=1 or print('------Training batch ', i)

            if feature_extractor=='CNN':
                X_train = cnn(X_train, training=False)

            X_train = X_train.reshape(batch_size, -1) # flatten array from to 2 dimension  
            #y_train = y_train[:, 1] # the output is one-hot encoded, select only second column which is for cancer=1

            if model_name == 'rfc':
                if i>0:
                    model.n_estimators = model.n_estimators +1          
                    verbose!=1 or print('------Current n_estimators: ', model.n_estimators)
        
            
            model.fit(X_train, y_train)
            
            
        verbose!=1 or print('---Evaluating model')
        batch_scores = {}
        for i, (X_val, y_val) in enumerate(val_gen):
            
            if feature_extractor=='CNN':
                X_val = cnn(X_val, training=False)

            X_val = X_val.reshape(batch_size, -1) # flatten array from IMG_SIZE shape
            # y_val = y_val[:, 1] # the output is one-hot encoded, select only second column which is for cancer=1
            
            verbose!=1 or print('------Getting predictions for batch ', i)
            y_pred_label = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)
            
            verbose!=1 or print('------Calculating scores of batch ', i)
            scores = get_scores(y_val, y_pred_label, y_pred_proba[:, -1])

            batch_scores[i] = scores

        history[fold] = get_mean_scores(batch_scores)


        if verbose==1:
            print('>>> Finish training model!')
            print('>>> Mean scores over batches: ')
            print(history[fold])
            print('---Saving model...')

        with open(savepath,'wb') as f:
            pickle.dump(model, f)

    if return_none:
        return None
    
    return history





# uncomment all below to run as script 'python3 run_model_example.py'

# create cross-validation folds (controlled by RANDOM_STATE)
# splitter = ds.DataSplitter(verbose=False)
# cv_patient, cv_img, label_by_img = splitter.get_cv(label_var='cancer', n_splits=5, random_state=RANDOM_STATE)


# from https://keras.io/api/callbacks/model_checkpoint/
# Prepare a directory to store all the checkpoints.

# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath = CHECKPOINT_PATH,
#     save_weights_only = False,
#     monitor = 'val_accuracy',
#     mode = 'max',
#     save_best_only = True
#     )




# if not already pre-trained and save, train feature extractor (NMF/PCA) on the train split of each cv fold
# for model in ['PCA', 'NMF']:

# if model == 'NMF':
#     normalize = (0,1)
#     n_components = NMF_N_COMPONENTS
# else:
#     normalize = (-1, 1)
#     n_components = PCA_N_COMPONENTS

# for fold in cv_img:

#     fex.cv_train_feature_extractor(
#         cv_img = cv_img,
#         patient_img_dict = splitter.train,
#         batch_size = BATCH_SIZE, # batch size for IncrementalPCA must be larger than n_components
#         basepath = BASEPATH,
#         img_size = (IMG_SIZE[0], IMG_SIZE[1]),
#         normalize = normalize,
#         random_state = RANDOM_STATE,
#         model_name = model,
#         n_components = n_components, 
#         extractor_path = EXTRACTOR_PATH,
#         verbose=True,
#         evaluate=False,
#         return_none=False   
#     )


# pre-trained CNN of choice
# base_model = keras.applications.Xception(
#     weights='imagenet',  # load weights pre-trained on ImageNet.
#     input_shape=(150, 150, 3),  # should have exactly 3 inputs channels because it is pre-trained on RBG images
#     include_top=False
#     ) 


# model architecture of choice
# model_layers = tf.keras.Sequential([ 
#     keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1])),
#     # keras.layers.MaxPooling2D((2, 2)))
#     # keras.layers.Conv2D(64, (3, 3), activation='relu'))
#     # keras.layers.MaxPooling2D((2, 2)))
#     # keras.layers.Conv2D(64, (3, 3), activation='relu'))
#     keras.layers.Flatten(),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(1)
#     ])



# run_cv_training(
#         cv_img = cv_img,
#         model_layers = model_layers,

#         label_img_dict = label_by_img,
#         patient_img_dict = splitter.train,
#         basepath = BASEPATH,
#         batch_size = BATCH_SIZE, 
#         img_size = (IMG_SIZE[0], IMG_SIZE[1]),
#         n_classes = 2,
#         shuffle = True,
#         normalize = (-1, 1),

#         input_shape=(IMG_SIZE[0], IMG_SIZE[1]),
#         augment_layers=True, base_model=None, trainable=False,
#         loss_func=LOSS, optimizer=OPTIMIZER, metrics=METRICS,

#         feature_extractor_name = 'CNN',
#         extractor_path = EXTRACTOR_PATH,
#         random_state = RANDOM_STATE,

#         strategy = tf.distribute.MirroredStrategy(),
#         epoch = EPOCH,
#         callbacks = [model_checkpoint_callback],
#         use_multiprocessing = True,
#         workers = CORES,
#         verbose = 2,

#         return_none = True
#         )
# pip install pydicom

import numpy as np
import tensorflow as tf
import keras

from zipfile import ZipFile
import os
import pydicom
import cv2

from kaggle.api.kaggle_api_extended import KaggleApi



# Set parameters

IMG_SIZE = (1024, 1024, 1)
NUMPY_SIZE = (1024, 1024)
BATCH_SIZE = 64
RANDOM_STATE = 42
# file directory where the images are stored (structured as in Kaggle)
BASEPATH = '' 



# functions to read DICOM file from directory and preprocess
# img is 2D image data

def crop_dark_area(img):
# adapted from https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
    
    mask = img > img.min()
    m,n = img.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    
    return img[row_start:row_end,col_start:col_end]


def crop_light_area(img):

    mask = img < img.max()
    m,n = img.shape
    mask0, mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()

    return img[row_start:row_end,col_start:col_end]


def read_and_preprocess(basepath, image_id, img_size=IMG_SIZE, normalize=(-1, 1)):
    
    file_path = basepath + '/' + str(image_id) + '.dcm'
    dcm = pydicom.dcmread(file_path)
    
    img = dcm.pixel_array
    img = crop_dark_area(img)
    img = crop_light_area(img)
    img = img.reshape(img.shape[0], img.shape[1], 1)

    if normalize == (-1, 1):
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = tf.image.per_image_standardization(img)
    
    elif normalize == (0, 1):
        img = img / 255
        img = tf.convert_to_tensor(img, dtype=tf.float32)

    img = tf.image.resize(img, [img_size[0], img_size[1]])

    return img



def read_numpy(basepath, image_id, img_size, normalize=(0, 1), numpy_size=NUMPY_SIZE):

    file_path = basepath + '/' + str(image_id) + '.npy' 
    img = np.load(file_path)
            
    if normalize==(-1, 1):
        img = (img - np.mean(img)) / np.std(img)
    
    if img_size[:2] != numpy_size[:2]:
        img = cv2.resize(img, dsize=(img_size[0], img_size[1]), interpolation=cv2.INTER_LINEAR)

    return img



# generator class for feature extraction (output only X)
# adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class ImgGenerator(tf.keras.utils.Sequence):

    def __init__(self, list_IDs, patient_img_dict, batch_size, basepath=BASEPATH, img_size=IMG_SIZE, normalize=(-1, 1), from_numpy=False):
        self.list_IDs = list_IDs  # all image_id in the training set
        self.batch_size = batch_size
        self.patient_img = patient_img_dict # splitter.train containing {patient_id: [image_id, ...]} of the training set
        self.basepath = basepath
        self.img_size = img_size
        self.normalize = normalize
        self.from_numpy = from_numpy

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / float(self.batch_size)))

    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, img_sizes)
        # Initialization
        X = np.empty((self.batch_size, self.img_size[0]*self.img_size[1]))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
          
          # Store sample
          if self.from_numpy == True:
            X[i,] = read_numpy(self.basepath, ID, self.img_size, self.normalize, numpy_size=NUMPY_SIZE).flatten()
    
          else:
            patient_id = [i for i in self.patient_img if ID in (self.patient_img[i])][0]
            folder_path = self.basepath + '/' + str(patient_id)  
            X[i,] = read_and_preprocess(folder_path, ID, self.img_size, self.normalize).numpy().flatten()

        return X

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        list_IDs_temp = self.list_IDs[index*self.batch_size : (index+1)*self.batch_size]

        # Generate data
        X_batch = self.__data_generation(list_IDs_temp)

        return X_batch
    


class ImgGen_from_numpy(tf.keras.utils.Sequence):

    def __init__(self, list_IDs, batch_size, basepath=BASEPATH, img_size=IMG_SIZE, normalize=(0, 1)):
        self.list_IDs = list_IDs  # all image_id in the training set
        self.batch_size = batch_size
        self.basepath = basepath
        self.img_size = img_size
        self.normalize = normalize

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / float(self.batch_size)))

    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, img_sizes)
        # Initialization
        X = np.empty((self.batch_size, self.img_size[0]*self.img_size[1]))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # print(ID)
            # Store sample
            file_path = self.basepath + '/' + str(ID) + '.npy' 
            x = np.load(file_path)
            
            if self.normalize==(-1, 1):
                x = (x - np.mean(x)) / np.std(x)
            
            if self.img_size[:2] != NUMPY_SIZE[:2]:
                x = cv2.resize(x, dsize=(self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_LINEAR)    
                
            X[i,] = x.flatten()

        return X

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        list_IDs_temp = self.list_IDs[index*self.batch_size : (index+1)*self.batch_size]

        # Generate data
        X_batch = self.__data_generation(list_IDs_temp)

        return X_batch
    


    
# generator class for predictive model (output both X and y)

class DataGenerator(keras.utils.Sequence):
  
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, patient_img_dict,
                from_numpy=False, basepath=BASEPATH,
                batch_size=BATCH_SIZE, img_size=IMG_SIZE, n_classes=2,
                shuffle=True, normalize=(-1, 1),
                feature_extractor=None, CNN_preprocess=None,
                verbose=False):
    
        'Initialization'
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.labels = labels
        self.patient_img = patient_img_dict
        self.img_size = img_size
        self.batch_size = batch_size

        self.feature_extractor = feature_extractor
        # feature extraction using either PCA or NMF >> input pre-trained model
        # feature extraction using CNN >> input string 'CNN'

        if feature_extractor is None:
            self.output_shape = IMG_SIZE
        elif feature_extractor=='CNN':
            self.output_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
            self.preprocess = CNN_preprocess
        else:
            self.output_shape = feature_extractor.n_components_

        self.from_numpy = from_numpy
        self.basepath = basepath
        self.shuffle = shuffle
        self.normalize = normalize
        self.verbose = verbose
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, img_sizes)
    
        y = np.empty((self.batch_size), dtype=int)

        # no feature extraction
        if self.feature_extractor is None:

            # Initialization
            X = np.empty((self.batch_size, self.img_size[0], self.img_size[1]))
 
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
            
                # Store sample
                if self.from_numpy == True:
                    X[i,] = read_numpy(self.basepath, ID, self.img_size, self.normalize, numpy_size=NUMPY_SIZE)

                else:
                    patient_id = [i for i in self.patient_img if ID in (self.patient_img[i])][0]
                    folder_path = self.basepath + '/' + str(patient_id)
                    X[i,] = tf.squeeze(read_and_preprocess(folder_path, ID, self.img_size, self.normalize))
            
                # Store class
                y[i] = self.labels[ID]
          

        # feature extraction using pre-trained CNN
        elif self.feature_extractor == 'CNN':

            # Initialization
            X = np.empty((self.batch_size, self.img_size[0], self.img_size[1], 3))
      
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
            
                # Store sample
                if self.from_numpy == True:
                    x = read_numpy(self.basepath, ID, self.img_size, self.normalize, numpy_size=NUMPY_SIZE)
                    x = np.repeat(x[:, :, np.newaxis], 3, axis=2)

                    if self.preprocess:
                        x = self.preprocess(np.squeeze(x))

                    X[i,] = x

                else:
                    patient_id = [i for i in self.patient_img if ID in (self.patient_img[i])][0]
                    folder_path = self.basepath + '/' + str(patient_id)
                
                    x = read_and_preprocess(folder_path, ID, self.img_size, self.normalize)
                    x = tf.tile(x, [1, 1, 3])

                    if self.preprocess:
                        x = self.preprocess(x)

                    X[i,] = x

                # Store class
                y[i] = self.labels[ID]


        # feature extraction using either PCA or NMF (input pre-trained model)
        else:
            # Initialization
            X = np.empty((self.batch_size, self.feature_extractor.n_components_))

            # Generate data
            for i, ID in enumerate(list_IDs_temp):

                # Store sample
                if self.from_numpy == True:
                    x = read_numpy(self.basepath, ID, self.img_size, self.normalize, numpy_size=NUMPY_SIZE)
                    X[i,] = self.feature_extractor.transform(x.flatten().reshape(1, -1))

                else:
                    patient_id = [i for i in self.patient_img if ID in (self.patient_img[i])][0]
                    folder_path = self.basepath + '/' + str(patient_id)
                
                    x = read_and_preprocess(folder_path, ID, self.img_size, self.normalize)
                    X[i,] = self.feature_extractor.transform(x.numpy().flatten().reshape(1, -1))

                # Store class
                y[i] = self.labels[ID]



        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y



    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y






# the rest is probably not used if the dataset is already stored in some directory

os.environ['KAGGLE_USERNAME'] = "<your-kaggle-username>"
os.environ['KAGGLE_KEY'] = "<your-kaggle-api-key>"

api = KaggleApi()
api.authenticate()

def unzip(file_name):

    # opening the zip file in READ mode
    with ZipFile(file_name, 'r') as zip:
        # printing all the contents of the zip file
        zip.printdir()
    
        # extracting all the files
        print('Extracting {}'.format(file_name))
        zip.extractall()
        os.remove(file_name)


def load(save_path, patient_id, image_id):
    
    folder = save_path + '/' + str(patient_id)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    os.chdir(folder)

    if os.path.exists(str(image_id)+'.dcm'):
        print('File already exists')
    else:
        file_path = 'train_images' + '/' + str(patient_id) + '/' + str(image_id) + '.dcm'
        

    try:
        unzip(str(image_id) + '.dcm.zip')
    except:
        print('File is not zip')

      
def load_by_cv(cv, fold, split, patient_img_dict, basepath):

    for patient_id in cv[fold][split]:
    
        print('PATIENT_ID: {}'.format(patient_id))
        new_folder = basepath + '/' + str(patient_id)
    
        if os.path.exists(new_folder):
            print('Patient {} data already exists'.format(patient_id))
    
        else:
            os.makedirs(new_folder)
            os.chdir(new_folder)
            #print('Downloading to {}'.format(os.getcwd()))

        for image_id in patient_img_dict[patient_id]:
            file_path = 'train_images' + '/' + str(patient_id) + '/' + str(image_id) + '.dcm'
            api.competition_download_file('rsna-breast-cancer-detection', file_path, path=new_folder)
        
            try:
                 unzip(str(image_id) + '.dcm.zip')
            except:
                print('File is not zip')




# Capstone-Project-Screening-Mammography-Breast-Cancer-Detection

Overview

The aim of our project is to develop interpretable predictive models, to aid radiologists and other related clinical personnels in detecting early signs of breast cancer from screening mammography by providing a probability of malignancy.

Additionally, the trade-off between performance, training cost, and interpretability between a variety of models with different level of complexity and interpretability, including InceptionV3, random forest  and logistic regression. Other than InceptionV3, principal component analysis (PCA) nd non-negative matrix factorization (NMF) will also extract features from mammogram images with the hope that they may be able to learn interpretable features that align with domain knowledge (image features that radiologist consider when interpreting mammograms).



Dataset
The datasets consist of 54,713 files in total and are 314.72 GB in size. They can be directly downloaded from Kaggle by joining the RSNA Screening Mammography Breast Cancer Detection competition organized by Radiological Society of North America.

Link to datasets: https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data


How to run the experiment?

There are 5 utility modules created in this project to help automate machine learning pipeline from image preprocessing to model evaluation. To use them in your notebook or script, place them in the same folder and import them by name.

- data_splitter.py is mainly used to provide consistent data partitioning into training, calibration, and test sets.
- data_loader.py contains the data generator classes that are used to read, preprocess and genearate image data with or without their corresponding target label to be used in batch training.
- feature_extraction.py contains functions to automate the training and evaluation of PCA and NMF models.
- run_model.py contains a variety of function to help automated predictive model training pipeline.
- eval_model.py contains functions to generate predictions from pipeline, evaluate them with a range of metrics and create visualizations.


Feel free to explore the  Jupyter notebooks in the notebook folder to see how these modules can be used in model development!

Note: PCA and NMF trained on 512 x 512 and 1024 x 1024 images are too larget to be stored in the repository. They are available [here](https://drive.google.com/drive/folders/17bDPY74nv2s0e1BCvJQh0ECKpZYZR3Az?usp=sharing).

P.S. This project was not successful. If you would like to learn more about why, [here](https://drive.google.com/file/d/1oSDM29Z2ETLBEibrpd0sdwnkEKNsNaRJ/view?usp=share_link) is the full report.

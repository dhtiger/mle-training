# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To create the environment
```
conda env create -f env.yml
conda activate mle-dev
```

## To install the housinglib library
first download the wheel file from the repo, then run the following command.
```
pip install housinglib-0.1.0-py3-none-any.whl
```
import it by using 
```
import housinglib
```

## The scripts folder contains the scripts to train and check scores of the model
To load the data use the ingest_data.py script by running following command in the shell
```shell
(mle-dev) :~$ python ingest_data.py
```
To train the datasets use train.py 
```shell
(mle-dev) :~$ python train.py
```
To see the performance of the model use score.py
```shell
(mle-dev) :~$ python score.py 

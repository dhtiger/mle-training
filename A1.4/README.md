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

## To excute the script

After cloning this repo, move to the folder mle-training.

Step 1: Create conda environment using the file given in the repo with extension .yml, using the below command.
'conda env create -f <Path to .yml file>'

Step 2: Activate the created python environment:
'conda activate mle-dev'

Step 3: Run the python file using the below command:
'python nonstandardcode.py'
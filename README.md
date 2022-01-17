# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The project and module performs 3 tasks:
1. EDA on specified columns  of a dataframe (specified in the config)
2. Data pre-processing to obtain a X & y data
3. modelling and reporting
   a. train test split
   b. fitting rf & logistic regression via CV (saves best models)
   c. reporting (plots, stats, features importance all saved as images)

## Running Files

1. modify the `config.yml` as appropriate/if needed
2. from terminal, navigate to `assignment1` folder
3. run `python churn_library.py`
4. expect the following outputs:
   - `log` folder: with logs
   - `models` folders: saved best model for rf & logistic
   - `images/eda` folder: eda images saved
   - `images/results` folder: model performance results 
     

Alternatively you can import the various functions of `churn_library.py` in a 
notebook.


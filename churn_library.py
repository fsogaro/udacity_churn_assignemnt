# library doc string
"""
Collection of functions needed for the Churn assignment (Udacity Nanodegree)

Author : FMS
Date: 12-01-2022
"""

# import libraries

import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
sns.set()


logging.basicConfig(
    filename='./logs/churn_library_run.log',
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def read_config(config_path):
    '''
        returns the content of a yml config

        input:
                pat to yaml file
        output:
                config
        '''

    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def eda_plot_and_save(df, col, plot_type, outfolder=None):
    '''
    switches plot type based on plot_type
    input:
            s: pandas series to plot
            plot_type: type of plot to use
            savepath: if not none, where to save it

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    if plot_type == "hist":
        df[col].hist()

    elif plot_type == "value_counts_bar":
        df[col].value_counts('normalize').plot(kind='bar')

    elif plot_type == "distplot":
        sns.distplot(df[col])

    elif plot_type == "corr":
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)

    else:
        plt.close()
        print(f"plot type {plot_type} not implemented")

    if outfolder is not None:
        check_dir(outfolder)
        plt.savefig(f'{outfolder}/{plot_type}_{col}.png')
        plt.close()


def check_dir(dir_path):
    """
    checks if a dir exists otherwise it creates
    input:
            dir_path: path to the dir to check
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def perform_eda(df, params_dict, savepath):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            params_dict:  parameters to standardise plotting
            savepath: saving path output

    output:
            None
    '''
    for plot_type in params_dict.keys():
        cols_to_plot = params_dict.get(plot_type, [])
        for col in cols_to_plot:
            if (col in df.columns) or (plot_type == "corr"):
                eda_plot_and_save(df, col, plot_type,
                                  outfolder=f"{savepath}")


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
     notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for cat_col in category_lst:

        if cat_col not in df.columns:
            print(f"column {cat_col} not found in df - skipping")
            continue

        # encoded column
        val_lst = []
        val_groups = df.groupby(cat_col).mean()[response]

        for val in df[cat_col]:
            val_lst.append(val_groups.loc[val])

        df[f'{cat_col}_{response}'] = val_lst

    return df


def perform_feature_engineering(df, response, keep_cols, test_size=0.3):
    '''
    input:
              df: pandas dataframe
              keep_cols: list of features
              response: string of response name [optional argument that could be
               used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    if keep_cols is None:
        keep_cols = df.columns

    x = df[keep_cols]
    y = df[response]
    return train_test_split(x, y, test_size=test_size, random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                folder='./images/results/'):
    '''
    produces classification report for training and testing results and stores
     report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    check_dir(folder)

    print('random forest results')
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace
    plt.axis('off')
    plt.savefig(f'{folder}/random_forest_clf_report.png', bbox_inches='tight')
    plt.close()

    print('logistic regression results')
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace
    plt.axis('off')
    plt.savefig(f'{folder}/logistic_regression_clf_report.png',
                bbox_inches='tight')
    plt.close()


def classification_performance_plot(models, X_test, y_test, folder="./",
                                    name="roc_auc"):
    """
    plot a roc_auc for different models into a single plot
    input:
        models: list of models
        X_test: features data
        y_test: ground truth
        folder: to save images
        name: optional, roc_auc

    output:
        none

    """
    plt.figure(figsize=(15, 8))
    for model in models:
        ax = plt.gca()
        plot_roc_curve(model, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig(f'{folder}/{name}.png', bbox_inches='tight')
    plt.close()


def feature_importance_plot(model, X_data, output_pth,
                            save_name="features_importance"):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure
            save_name: name of image

    output:
             None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    check_dir(output_pth)
    plt.savefig(f'{output_pth}/{save_name}', bbox_inches='tight')
    plt.close()


def train_models(X_train, X_test, y_train, y_test, savepath="./"):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(random_state=42)

    classifier_grid_search = {
        "random_forest": {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }
    }

    cv_rfc = GridSearchCV(estimator=rfc,
                          param_grid=classifier_grid_search["random_forest"],
                          cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # save best model
    check_dir(f'{savepath}models/')
    joblib.dump(cv_rfc.best_estimator_, f'{savepath}models/rfc_model.pkl')
    joblib.dump(lrc, f'{savepath}models/logistic_model.pkl')

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # classification reports
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                folder=f"{savepath}images/results/")
    # ROC auc
    classification_performance_plot(
        [cv_rfc.best_estimator_, lrc],
        X_test, y_test, folder=f"{savepath}images/results/"
    )

    # features importance
    feature_importance_plot(cv_rfc.best_estimator_, X_train,
                            f"{savepath}images/results/",
                            save_name="RF_features_importance")


if __name__ == "__main__":
    config = read_config("config.yml")
    SAVEPATH = config.get("savepath", "./")
    df = import_data(config["csv_path"])
    # custom step
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    perform_eda(df, config['eda_params'], savepath="./images/eda/")

    df2 = encoder_helper(df, config['category_to_encode'], response="Churn")

    out = perform_feature_engineering(df2,
                                      response="Churn",
                                      keep_cols=config["keep_cols"])
    train_models(*out, savepath="./")





"""
testing module for the Churn assignment (Udacity Nanodegree)

Author : FMS
Date: 12-01-2022
"""
import os
import numpy as np
import logging
# import churn_library_solution as cls
import pandas as pd
import churn_library as cl


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear "
                      "to have rows and columns")
        raise err


def test_eda_config(read_config):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        c = read_config("./config.yml")
        logging.info("Testing read_config: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing read_config: The file wasn't found")
        raise err

    try:
        assert len(c.keys()) > 0
    except AssertionError as err:
        logging.error("Config has no keys")
        raise err


def test_eda(perform_eda):
    """
    test perform eda function
    """
    df = cl.import_data("./data/bank_data.csv")
    params_dict = {"hist": ["Attrition_Flag", "Customer_Age"],
                   "corr": [""],
                   "value_counts_bar": ["Marital_Status"],
                   "distplot": ["Total_Trans_Ct"],
                   }
    logging.info(params_dict)
    perform_eda(df, params_dict, './testing/images/eda/')
    try:
        assert os.path.isfile(
            f"./testing/images/eda/distplot_Total_Trans_Ct.png")
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("ida image file not found")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df = cl.import_data("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    df_out = encoder_helper(df, category_lst=["Gender"], response="Churn")
    try:
        assert "Gender_Churn" in df_out.columns
        logging.info("Testing encoder_helper: SUCCESS, new columns found")
    except AssertionError as err:
        logging.error("expected new column, but not found")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''

    df = pd.DataFrame(
        {"f1": np.arange(100),
         "f2": np.arange(100),
         "f3": np.arange(100),
         "y": np.random.randint(0, 2, 100)
         }
    )

    out = perform_feature_engineering(df,
                                      response='y', keep_cols=["f1", "f3"],
                                      test_size=0.3)
    try:
        assert len(out) == 4
        assert out[0].shape[1] == out[1].shape[1] == 2
        assert len(out[0]) == len(out[2]) == 70
        assert len(out[1]) == len(out[3]) == 30
        logging.info("Testing test_perform_feature_engineering: SUCCESS, "
                     "correct shapes")
    except AssertionError as err:
        logging.error("train test split did not return the right shapes")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    df = pd.DataFrame(
        {"f1": np.arange(100),
         "f2": np.arange(100),
         "f3": np.arange(100),
         "y": np.random.randint(0, 2, 100)
         }
    )

    out = cl.perform_feature_engineering(df,
                                         response='y', keep_cols=["f1", "f3"],
                                         test_size=0.3)
    savepath = "./testing/"
    train_models(*out, savepath="./testing/")

    try:
        assert os.path.isfile(f"{savepath}models/rfc_model.pkl")
        assert os.path.isfile(f"{savepath}models/logistic_model.pkl")
        assert os.path.isfile(
            f"{savepath}images/results/random_forest_clf_report.png")
        logging.info("Testing train_models: SUCCESS,")
    except AssertionError as err:
        logging.error("train test split did not return the right shapes")
        raise err


if __name__ == "__main__":
    test_import(cl.import_data)
    test_eda_config(cl.read_config)
    test_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)
    test_train_models(cl.train_models)

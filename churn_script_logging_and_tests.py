'''
Test and logging module for Project Predict Customer Churn.
'''
import os
import logging
import churn_library as cl


cl.mkdir_or_exist('./logs')

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
        test data import - this example is completed for you to assist with the other test functions
        '''
    try:
        data_frame = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return data_frame


def test_eda(perform_eda, data_frame):
    '''
    test perform eda function
    '''
    try:
        perform_eda(data_frame)
        logging.info("Testing perform eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform eda: No such file or directory")
        raise err

    try:
        assert os.path.exists('./images/eda/churn_distribution.png')
        assert os.path.exists('./images/eda/customer_age_distribution.png')
        assert os.path.exists('./images/eda/heatmap.png')
        assert os.path.exists('./images/eda/marital_status_distribution.png')
        assert os.path.exists(
            './images/eda/total_transaction_distribution.png')
    except AssertionError as err:
        logging.error("Testing perform eda: The output file wasn't found")
        raise err


def test_encoder_helper(encoder_helper, data_frame,
                        category_lst, response=None):
    '''
    test encoder helper
    '''
    try:
        data_frame = encoder_helper(data_frame, category_lst, response)
        logging.info("Testing encoding: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoding: FAILURE")
        raise err

    col_names = data_frame.columns
    for category in category_lst:
        try:
            if response:
                assert category + '_' + response in col_names
            else:
                assert category + '_Churn' in col_names
        except AssertionError as err:
            logging.error(
                "Testing encoding: there is no %s encoded column",
                category)
            raise err

    return data_frame


def test_perform_feature_engineering(perform_feature_engineering,
                                     data_frame,
                                     keep_cols,
                                     response=None):
    '''
    test perform_feature_engineering
    '''
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            data_frame, keep_cols, response
        )
        logging.info("Testing creation of features: SUCCESS")
    except Exception as err:
        logging.error("Testing creation of features: FAILURE")
        raise err

    try:
        assert (x_train.shape[0] > 0) or (x_train.shape[1] > 0)
        assert (x_test.shape[0] > 0) or (x_test.shape[1] > 0)
        assert (y_train.shape[0] > 0) or (y_train.shape[1] > 0)
        assert (y_test.shape[0] > 0) or (y_test.shape[1] > 0)
    except AssertionError as err:
        logging.error(
            "Testing creation of features: The split process is not correct")
        raise err

    return x_train, x_test, y_train, y_test


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing models training: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing models training: No such file or directory")
        raise err

    try:
        assert os.path.exists('./images/results/rfc_results.png')
        assert os.path.exists('./images/results/lr_results.png')
        assert os.path.exists('./images/results/roc_curve_result.png')
        assert os.path.exists('./models/rfc_model.pkl')
        assert os.path.exists('./models/logistic_model.pkl')
        assert os.path.exists('./images/results/feature_importance_rfc.png')
        assert os.path.exists('./images/results/feature_importance_lrc.png')
        assert os.path.exists('./images/results/rfc_results.png')
        assert os.path.exists('./images/results/lr_results.png')
    except AssertionError as err:
        logging.error("Testing model training: The output file wasn't found")
        raise err


if __name__ == "__main__":
    df = test_import(cl.import_data)
    test_eda(cl.perform_eda, df)
    test_encoder_helper(cl.encoder_helper, df, cl.cat_columns)
    X_train_df, X_test_df, y_train_df, y_test_df = test_perform_feature_engineering(
        cl.perform_feature_engineering, df, cl.keep_cols)
    test_train_models(
        cl.train_models,
        X_train_df,
        X_test_df,
        y_train_df,
        y_test_df)

# library doc string
'''
Library with helper functions for Predict Customer Churn project.
'''

# import libraries
import os
# Below import have to be on the top of imports
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']


def mkdir_or_exist(path):
    '''
    func for make directory, or if exist the skip
    '''
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    mkdir_or_exist('images/eda/')

    plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    plt.savefig('images/eda/churn_distribution.png')

    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.savefig('images/eda/customer_age_distribution.png')

    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/marital_status_distribution.png')

    plt.figure(figsize=(20, 10))
    sns.displot(dataframe['Total_Trans_Ct'])
    plt.savefig('images/eda/total_transaction_distribution.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/eda/heatmap.png')


def encoder_worker(dataframe, column_name):
    '''
    helper function to turn categorical column into a new column with
    propotion of churn for each category

    input:
            df: pandas dataframe
            column_name: list of columns that contain categorical features
            response: string of response name [optional argument that could be
                      used for naming variables or index y column]

    output:
            list: list of encoded observations (length of list is equal len(df))
    '''
    result_list = []
    res_group = dataframe.groupby(column_name).mean()['Churn']

    for val in dataframe[column_name]:
        result_list.append(res_group.loc[val])

    return result_list


def encoder_helper(dataframe, category_lst, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming
                      variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        if response:
            dataframe[category + '_' + response] = encoder_worker(dataframe, category)
        else:
            dataframe[category + '_Churn'] = encoder_worker(dataframe, category)

    return dataframe


def perform_feature_engineering(dataframe, columns2keep, response=None):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming
                        variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    x_data = pd.DataFrame()
    y_data = pd.DataFrame()
    if response:
        y_data[response] = dataframe['Churn']
    else:
        y_data = dataframe['Churn']
    x_data[columns2keep] = dataframe[columns2keep]
    return train_test_split(x_data, y_data, test_size=0.3, random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
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
    plt.figure(figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/rfc_results.png')

    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/lr_results.png')


def feature_importance_plot(model, x_data, col_names, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.coef_[0]
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [col_names[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(list(range(len(importances))), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)

def roc_comparsion_plot(lrc, rfc, x_test, y_test, output_pth):
    '''
    creates and stores the roc of all clfs
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(rfc, x_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig(output_pth)

def train_models(x_train, x_test, y_train, y_test, col_names):
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
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='liblinear')

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    mkdir_or_exist('./images/results')
    mkdir_or_exist('./models')

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    roc_comparsion_plot(
        lrc,
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        './images/results/roc_curve_result.png')

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Save feature importance from Random Forest model
    plt.figure(figsize=(20, 5))
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    plt.savefig('./images/results/feature_importance_rfc.png')

    # Save feature importance from LR model
    feature_importance_plot(
        lrc, x_test,
        col_names, './images/results/feature_importance_lrc.png')

    # classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

def main():
    '''
    Main func which is running whole pipeline
    '''
    dataframe = import_data('data/bank_data.csv')
    perform_eda(dataframe)
    dataframe = encoder_helper(dataframe, category_lst=cat_columns)
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        dataframe, keep_cols)
    train_models(x_train, x_test, y_train, y_test, keep_cols)


if __name__ == "__main__":
    main()

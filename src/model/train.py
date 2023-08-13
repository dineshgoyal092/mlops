# Import libraries

import argparse
import glob
import os
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.workspace import Workspace
# import mlflow.sklearn
# from mlflow import MlflowClient
azure_tenant_id = os.environ.get("AZURE_TENANT_ID")
azure_client_id = os.environ.get("AZURE_APP_ID")
azure_client_secret = os.environ.get("AZURE_CLIENT_SECRET")

print("------------------")
print(azure_tenant_id)
print(azure_client_id)
print(azure_client_secret)
print("-----------")
    
# auth = ServicePrincipalAuthentication(azure_tenant_id, azure_client_id, azure_client_secret)
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
# mlflow.set_tracking_uri("azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/50fa0132-1a23-44e7-8af9-1d776d5b4c2a/resourceGroups/odsp_idc_ds_ab/providers/Microsoft.MachineLearningServices/workspaces/odsp_idc_ds_ab_ws")
# https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri


# TO DO: add function to split data
def split_data(df):
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values
    return train_test_split(X, y, test_size=0.30, random_state=0)

# define functions
def main(args):
    # TO DO: enable autologging
    mlflow.autolog()

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)





def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # mlflow.start_run()
    # mlflow.log_param("my", "param")
    # mlflow.log_metric("score", 100)
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
    # mlflow.end_run()

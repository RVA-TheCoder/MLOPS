import warnings
import argparse
import logging

import numpy as np, pandas as pd
from pathlib import Path


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Get arguments from the command line
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)

args = parser.parse_args()

# Evaluation function
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def eval_metrics(actual, pred):

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2


if __name__ =="__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from local
    data = pd.read_csv("../data/winequality.csv", sep=";")

    print(data.head())

    # Split the data into training and test sets. (0.75, 0.25) split.
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(data, test_size=0.2)

    # The predicted column is "quality" which is a scalar from [3, 9]
    X_train = train.drop(["quality"], axis=1)
    X_test = test.drop(["quality"], axis=1)
    y_train = train[["quality"]]
    y_test = test[["quality"]]

    # Hyperparamters
    alpha = args.alpha
    l1_ratio = args.l1_ratio

    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import ElasticNet


    # Tracking is the first thing that we should be specified before any experiment or runs
    mlflow.set_tracking_uri(uri="")  # default directory
    print("The set URI : ", mlflow.get_tracking_uri())


    exp = mlflow.set_experiment(experiment_name="Exp_AutoLog01")

    print("Exp Name : ", exp.name)
    print("Exp ID : ", exp.experiment_id)
    print("Exp Tags: ", exp.tags)
    print("Exp Artifact Location : ", exp.artifact_location)
    print("Exp Lifecycle Stage : ", exp.lifecycle_stage)

    mlflow.start_run(experiment_id=exp.experiment_id)

    # Define the autolog function before Model training because tracking happens during Model Training
    # log_input_examples=True, will log only few examples from the input data

    """
    mlflow.autolog() function can log the standard model defined in the library
    but cannot log the customized models.
    Therefore, for custom  model use the manual logging.
    """
    mlflow.autolog(log_input_examples=True,
                   log_model_signatures=True,
                   log_models=True)

    # Model Creation
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(X_train, y_train)

    # Predictions
    y_pred = lr.predict(X_test)

    (rmse, mae, r2) = eval_metrics(y_test, y_pred)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # mlflow.autolog() function will take care of this.
    # Logging trained model
    # mlflow.sklearn.log_model(lr , "trained_LR_Model")

    active_run = mlflow.active_run()

    print("Active Run Name : ",active_run.info.run_name)
    print("Active Run Id : ", active_run.info.run_id)

    mlflow.end_run()











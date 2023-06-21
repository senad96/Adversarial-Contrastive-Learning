import prefect
import requests
from prefect.task_runners import SequentialTaskRunner

from prefect import task, flow

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import os
import boto3



@task(name="load-data", retries=2, retry_delay_seconds=5)
def load_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    return X, y



@task(name="train-model")
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test



@task(name="evaluate-model")
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


@flow(name="Demo Prefect", description="Demo Prefect")
def develop_decision_tree():
    X, y = load_data()
    model, X_test, y_test = train_model(X, y)
    accuracy = evaluate_model(model, X_test, y_test)
    return accuracy



if __name__ == "__main__":
    print(develop_decision_tree())
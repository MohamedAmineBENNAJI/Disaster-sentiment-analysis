"""This module includes the Machine learning pipeline for disaster messages."""
# import libraries
import logging
import pickle
import re
import sys
from typing import Any, Dict, List, Tuple, Union

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from xgboost import XGBClassifier

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
# set the logger config
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def load_data_from_database(
    database_path: str,
) -> Union[Tuple[Any, Any, List[Any]], None]:
    """This utility function loads disaster data from a SQL database
        and return features and targets.

    Args:
        database_path: The path of the SQL database.

    Returns:
        X: The input messages.
        y: The output classes.
        categories: List of target names.
    """

    # Load data from database and read it
    try:

        engine = create_engine(f"sqlite:///{database_path}")
        df = pd.read_sql("messages", engine)

        # Select the Input features and the output classes
        # and define categories
        X = df["message"]
        y = df.drop(columns=["message", "id", "genre", "original"])
        categories = list(y.columns)
        logging.info("Data loaded successfully")
        return X, y, categories

    except FileNotFoundError:
        logging.error("Check the database path")

        return None


def tokenize(text: str) -> List[str]:
    """This utility function will clean and tokenize the input messages.

    Args:
        text: The input messages.

    Returns:
        clean_tokens: A list of clean tokens.
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:

        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(params: Dict[str, Any], verbose: bool = True):
    """This function will prepare our Scikit-learn pipeline that will
    transform our data and fit it to targets.
    Args:
        params: A dictionary containing the parameters used to
        fine-tune the model with GridSearchCV algorithm.
        verbose: A boolean used to control displaying the logs of training.

    Returns:
        cv_model: The cross-validation model.
    """
    classifier = XGBClassifier(tree_method="hist", seed=42)
    pipeline = Pipeline(
        [
            (
                "vect",
                TfidfVectorizer(
                    tokenizer=tokenize,
                    use_idf=True,
                    smooth_idf=True,
                    sublinear_tf=False,
                ),
            ),
            (
                "clf",
                MultiOutputClassifier(classifier),
            ),
        ]
    )
    cv_model = GridSearchCV(
        pipeline,
        param_grid=params,
        verbose=1,
        cv=3,
        refit=True,
        return_train_score=True,
    )

    return cv_model


def evaluate_model(
    model: Pipeline,
    test_features: pd.Series,
    y_test: pd.DataFrame,
    categories: List[str],
) -> None:
    """This function evaluates the trained model and returns the
    confusion matrix and the classification report.

    Args:
        model: The trained model.
        test_features: The evaluation messages..
        y_test: The evaluation labels.
        categories: A list containing the used categories.
    """
    y_pred = model.predict(test_features)
    predictions_df = pd.DataFrame(y_pred, columns=categories)
    for i, col in enumerate(categories):

        class_report = classification_report(y_test[col], predictions_df[col])
        logging.info(f"***Classifcation report for {col}***\n {class_report}")


def save_model(model: Pipeline, model_filepath: str) -> None:
    """This function is used to save the trained model.

    Args:
        model: The trained classifier.
        model_filepath: The output path of the classifier.
    """

    model = model.best_estimator_
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)


def main():
    """This function executes the main training and evaluation pipelines."""
    if len(sys.argv) == 4:
        logging.info("Starting the machine learning pipeline")
        # Load the data
        database_path, model_filepath, fine_tune = sys.argv[1:]
        fine_tune = eval(fine_tune)
        logging.info(f"Loading the data from Database {database_path}")
        X, y, category_names = load_data_from_database(database_path)
        # Split the data
        logging.info("Splitting the data into train,test splits")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Define parameters for GridSearchCV fine-tuning
        if fine_tune:
            print("Fine-tuning the model is enabled")

            params = {
                "clf__estimator__max_depth": [3, 5],
                "clf__estimator__n_estimators": [100, 200],
                "clf__estimator__learning_rate": [0.1, 0.01],
            }
        else:
            print("Fine-tuning the model is disabled")

            # Use the best fine-tuning parameters without launching the
            # fine-tuning process
            params = {
                "clf__estimator__learning_rate": [0.1],
                "clf__estimator__max_depth": [5],
                "clf__estimator__n_estimators": [200],
            }
        # Building the model
        logging.info("Building the model")
        model = build_model(params=params)
        # Training
        logging.info(
            f"Launching the training pipeline with {X_train.shape[0]} samples "
        )
        model.fit(X_train, y_train)
        logging.info(f"best_params: {model.best_params_}")

        # Evaluation
        logging.info("Evaluating the model")
        evaluate_model(model, X_test, y_test, category_names)
        save_model(model, model_filepath)
        logging.info("Model trained, evaluated and saved successfully!")

        # Saving
        logging.info(f"Saving the model to {model_filepath}")

    else:
        logging.info(
            """Please provide the filepath of the disaster messages database
            as the first argument and the filepath of the pickle file to
            save the model to as the second argument and a boolean specifying
            whether we are fine-tuning our model using GridSearchCV or we use
            the best parameters directly
            .\n\nExample: python train_classifier.py
            ../data/DisasterResponse.db classifier.pkl False"""
        )


if __name__ == "__main__":
    # Execute the Machine Learning pipeline, load, train, evaluate
    # and save the classifier
    main()

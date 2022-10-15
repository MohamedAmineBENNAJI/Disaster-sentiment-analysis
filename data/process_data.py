"""This module includes the code used for the ETL pipeline."""
import logging
import re
import sys

import pandas as pd
from sqlalchemy import create_engine

# set the logger config
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """This utility function is used to load messages and categories dataframes
        and merge them together.

    Args:
        messages_filepath: Path of the messages file.
        categories_filepath: Path of the file containing annotations.

    Returns:
        data: Dataframe containing messages and categories.
    """
    try:

        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
        data = pd.merge(messages, categories, on=("id"), how="inner")
        return data

    except FileNotFoundError:
        logging.error("Check the provided filepaths")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """This utility function is used to clean the data.

    Args:
        df: The merged dataframe.

    Returns:
        data: The cleaned dataframe.
    """
    categories_df = df[["id", "categories"]]
    # create a dataframe of the 36 individual category columns
    categories_df = categories_df["categories"].str.split(";", expand=True)

    # extract the column names from the first row
    row = categories_df.loc[0]
    category_colnames = list(row.apply(lambda x: re.sub("-\d", "", x)))
    # set the new dataframe columns
    categories_df.columns = category_colnames
    # extract the binary values from each column
    for column in categories_df:
        # set each value to be the last character of the string
        categories_df[column] = categories_df[column].astype(str).str[-1]
        # convert column from string to numeric
        categories_df[column] = pd.to_numeric(categories_df[column])
    # drop the original categories column from `df`

    df = df.drop(columns=["categories"])
    # concatenate the original dataframe with the new `categories` dataframe
    data = pd.concat([df, categories_df], axis=1)

    # remove the duplicates
    number_of_duplicates = data.duplicated(keep="first").sum()
    logging.info(f"The number of duplicated elements is {number_of_duplicates}")

    logging.info("Removing duplicates")
    # drop duplicates
    data = data.drop_duplicates()
    dupplicated_elements = data.duplicated(keep="first").sum()
    # checking the number of duplicates
    logging.info(f"The number of duplicated elements is {dupplicated_elements}")

    return data


def save_data(df, database_filename):
    """This utility function is used to save the cleaned data into a database.

    Args:
        df: The cleaned data.
        database_filename: The name of the database to save.
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("encoded_messages", engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        logging.info(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        logging.info("Cleaning data...")
        df = clean_data(df)

        logging.info("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        logging.info("Cleaned data saved to database!")

    else:
        logging.error(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()

import json
import plotly
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine("sqlite:///data/disaster_response.db")
df = pd.read_sql_table("messages", engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    n_messages = df.groupby("genre").count()["message"]
    count_percentage = 100 * n_messages / n_messages.sum()
    genres = list(n_messages.index)
    messages_per_category = df.drop(
        ["id", "message", "original", "genre"], axis=1
    ).sum()
    messages_per_category = messages_per_category.sort_values(ascending=False)
    categories = list(messages_per_category.index)
    df["text_len"] = df.message.str.len()
    n_messages, bins = np.histogram(df.text_len, bins=range(0, 1000, 100))
    bins = bins[:-1] + bins[1:]

    # create visuals
    graphs = [
        {
            "data": [
                {
                    "type": "pie",
                    "uid": "f4de1f",
                    "hole": 0.4,
                    "name": "Genre",
                    "pull": 0,
                    "domain": {"x": count_percentage, "y": genres},
                    "textinfo": "label+value",
                    "hoverinfo": "all",
                    "labels": genres,
                    "values": n_messages,
                }
            ],
            "layout": {"title": "Number of messages by genre"},
        },
        {
            "data": [
                {
                    "type": "bar",
                    "x": bins,
                    "y": n_messages,
                    "marker": {"color": "#7fc97f"},
                }
            ],
            "layout": {
                "title": "Message distribution by length",
                "yaxis": {"title": "Number of messages"},
                "xaxis": {"title": "Message length"},
                "barmode": "group",
            },
        },
        {
            "data": [
                {
                    "type": "bar",
                    "x": categories,
                    "y": messages_per_category,
                    "marker": dict(
                        size=36,
                        # set color equal to a variable
                        color=np.random.randn(256),
                        # one of plotly colorscales
                        colorscale="hot",
                        # enable color scale
                        showscale=False,
                    ),
                }
            ],
            "layout": {
                "title": "Number of messages by category",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
                "barmode": "group",
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()

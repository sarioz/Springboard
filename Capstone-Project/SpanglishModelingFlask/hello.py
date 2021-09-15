from flask import Flask, render_template, request, flash, url_for, redirect
from werkzeug.exceptions import abort
from pos_tagger.pos_tagger import PosTagger
from sentiment_analyzer.sentiment_analyzer import SentimentAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = '8t4haf390h9a4h8r'


pos_tagger = PosTagger()
sentiment_analyzer = SentimentAnalyzer()


@app.route('/hello')
def hello():
    return 'Hello, World!'


def prettify_pos_prediction(tweet_tokens, pos_predictions) -> str:
    output = ""
    for token, tag in zip(tweet_tokens, pos_predictions):
        output += f"{token}:\t{tag}\n<BR>\n"
    return output


def perform_pos_tagging(content) -> str:
    tweet_tokens, pos_predictions = pos_tagger.make_prediction(content)
    return prettify_pos_prediction(tweet_tokens, pos_predictions)


def prettify_sa_prediction(tweet_tokens, sentiment) -> str:
    return f"Content: {' '.join(tweet_tokens)}<BR>\nSentiment: {sentiment}<BR>"


def perform_sentiment_analysis(content) -> str:
    tweet_tokens, sentiment = sentiment_analyzer.make_prediction(content)
    return prettify_sa_prediction(tweet_tokens, sentiment)


@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        content = request.form['content']
        if not content:
            flash('Content is required!')
        else:
            if 'tag_pos' in request.form:
                return perform_pos_tagging(content)
            elif 'analyze_sentiment' in request.form:
                return perform_sentiment_analysis(content)
            else:
                return 'invalid submission - please use the UI'

    return render_template('query.html')

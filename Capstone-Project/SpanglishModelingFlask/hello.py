from flask import Flask, render_template, request, flash, url_for, redirect
from werkzeug.exceptions import abort
from pos_tagger.pos_tagger import PosTagger

app = Flask(__name__)
app.config['SECRET_KEY'] = '8t4haf390h9a4h8r'


pos_tagger = PosTagger()


@app.route('/')
def index():
    return render_template('index.html')


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


@app.route('/query', methods=('GET', 'POST'))
def query():
    if request.method == 'POST':
        content = request.form['content']
        if not content:
            flash('Content is required!')
        else:
            if 'tag_pos' in request.form:
                return perform_pos_tagging(content)
            elif 'analyze_sentiment' in request.form:
                return f"Will perform Sentiment Analysis on {content}"
            else:
                return 'invalid submission - please use the UI'

    return render_template('query.html')

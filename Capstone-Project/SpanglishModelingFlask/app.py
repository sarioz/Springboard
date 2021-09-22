import sys
import os

from flask import Flask, render_template, request, flash, url_for, redirect
from werkzeug.exceptions import abort

from pos_tagger.pos_tagger_base import PosTagger
from sentiment_analyzer.sentiment_analyzer_base import SentimentAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = '8t4haf390h9a4h8r'
app.jinja_env.globals.update(zip=zip)

pos_tagger = PosTagger()
sentiment_analyzer = SentimentAnalyzer()


def perform_pos_tagging(content) -> str:
    tweet_tokens, pos_predictions = pos_tagger.make_prediction(content)
    return render_template('result_pos_tagging.html', tweet_tokens=tweet_tokens, pos_predictions=pos_predictions)


def perform_sentiment_analysis(content) -> str:
    tokens, sentiment = sentiment_analyzer.make_prediction(content)
    return render_template('result_sentiment_analysis.html', tokens=tokens, sentiment=sentiment)


@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        content = request.form['content']
        if not content:
            flash('Content is required to be nonempty')
        else:
            if 'tag_pos' in request.form:
                try:
                    return perform_pos_tagging(content)
                except ValueError as value_error:
                    if value_error.args:
                        flash(' '.join(value_error.args))
                    else:
                        flash("A further unspecified ValueError has occurred.")
            elif 'analyze_sentiment' in request.form:
                try:
                    return perform_sentiment_analysis(content)
                except ValueError as value_error:
                    if value_error.args:
                        flash(' '.join(value_error.args))
                    else:
                        flash("A further unspecified ValueError has occurred.")
            else:
                return 'Invalid submission - please use the web based user interface.'

    return render_template('query.html')

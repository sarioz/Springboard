# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from loader import Loader
from tweet_cleaner import TweetCleaner
from noiser import DisjointNoiser

from random import choice

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

TRAINING_INPUT_PATH = '../data/lid_train_lines.txt'


def main():
    raw_tweets = Loader(TRAINING_INPUT_PATH).load()
    cleaner = TweetCleaner()
    clean_tweets = [cleaner.clean_tweet(t) for t in raw_tweets]

    noiser = DisjointNoiser()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

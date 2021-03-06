import tensorflow as tf
from tensorflow.keras.models import load_model

from data_loader import DataLoader
from inference_runner import InferenceRunner
from nn_input_preparer import NNInputPreparer
from nn_model_creator import NNModelCreator
from noiser import DisjointNoiser
from tweet_cleaner import TweetCleaner
from tweet_selector import TweetSelector


# preprocessing params
MIN_TWEET_LENGTH = 10
MAX_TWEET_LENGTH = 50

# architectural params
LATENT_DIM = 512
DENSE_DIM = 256

# file paths
TRAINING_INPUT_FILENAME = '../data/lid_train_lines.txt'
DEV_INPUT_FILENAME = '../data/lid_dev_lines.txt'
EXPERIMENT_NAME = f'07.sel_{MIN_TWEET_LENGTH}-{MAX_TWEET_LENGTH}_BiLSTMs_2_Dense_2'
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/dim_{LATENT_DIM}/dfepoch_49_end.h5'


def main():
    print('tf:', tf.__version__)
    print('TRAINING_MODEL_FILENAME =', TRAINING_MODEL_FILENAME)

    nn_input_preparer = NNInputPreparer()

    model_creator = NNModelCreator(latent_dim=LATENT_DIM, dense_dim=DENSE_DIM)
    loaded_training_model = load_model(TRAINING_MODEL_FILENAME)

    encoder_model, decoder_model = model_creator.derive_inference_models(loaded_training_model)
    inference_runner = InferenceRunner(encoder_model=encoder_model, decoder_model=decoder_model)

    cleaner = TweetCleaner()
    selector = TweetSelector(min_length=MIN_TWEET_LENGTH, max_length=MAX_TWEET_LENGTH)
    noiser = DisjointNoiser()

    for input_filename in [TRAINING_INPUT_FILENAME, DEV_INPUT_FILENAME]:
        k = 10
        print(f'processing the first {k} selected tweets from {input_filename}')
        raw_tweets = DataLoader(input_filename).load()
        clean_tweets = [cleaner.clean_tweet(t) for t in raw_tweets]
        clean_tweets_as_lists = [list(t) for t in clean_tweets]
        selected_tweets_as_lists = [t for t in clean_tweets_as_lists if selector.select(t)]
        gb_inference = nn_input_preparer.get_batches(selected_tweets_as_lists, noiser, batch_size=1)
        for i in range(k):
            noised_batch, originals_batch, original_delayed_batch = next(gb_inference)
            print('[noised    ]', nn_input_preparer.decode_tweet(noised_batch[0]))
            print('[original  ]', nn_input_preparer.decode_tweet(originals_batch[0]))
            print('[original 2]', ''.join(selected_tweets_as_lists[i]))
            print('[or-delayed]', nn_input_preparer.decode_tweet(original_delayed_batch[0]))
            decoded_tweet = inference_runner.decode_sequence(noised_batch)
            print('[decoded   ]', decoded_tweet)
            print()


if __name__ == '__main__':
    main()

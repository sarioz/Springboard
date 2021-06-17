import tensorflow as tf
from tensorflow.keras.models import load_model

from data_loader import DataLoader
from inference_runner import InferenceRunner
from nn_input_preparer import NNInputPreparer
from nn_model_creator import NNModelCreator
from noiser import DisjointNoiser
from tweet_cleaner import TweetCleaner

TRAINING_INPUT_FILENAME = '../data/lid_train_lines.txt'

EXPERIMENT_NAME = '03.BiLSTMs'

LATENT_DIM = 512

TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/dim_{LATENT_DIM}/dfepoch_1_0.87239.h5'


def main():
    print('tf:', tf.__version__)
    print('TRAINING_MODEL_FILENAME =', TRAINING_MODEL_FILENAME)

    raw_tweets = DataLoader(TRAINING_INPUT_FILENAME).load()
    cleaner = TweetCleaner()
    clean_tweets = [cleaner.clean_tweet(t) for t in raw_tweets]
    noiser = DisjointNoiser()
    clean_tweets_as_lists = [list(t) for t in clean_tweets]

    nn_input_preparer = NNInputPreparer()

    model_creator = NNModelCreator(latent_dim=LATENT_DIM)
    loaded_training_model = load_model(TRAINING_MODEL_FILENAME)

    encoder_model, decoder_model = model_creator.derive_inference_models(loaded_training_model)
    inference_runner = InferenceRunner(encoder_model=encoder_model, decoder_model=decoder_model)

    gb_inference = nn_input_preparer.get_batches(clean_tweets_as_lists, noiser, batch_size=1)
    for i in range(10):
        noised_batch, originals_batch, original_delayed_batch = next(gb_inference)
        print('[noised    ]', nn_input_preparer.decode_tweet(noised_batch[0]))
        print('[original  ]', nn_input_preparer.decode_tweet(originals_batch[0]))
        print('[original 2]', ''.join(clean_tweets_as_lists[i]))
        print('[or-delayed]', nn_input_preparer.decode_tweet(original_delayed_batch[0]))
        decoded_tweet = inference_runner.decode_sequence(noised_batch)
        print('[decoded   ]', decoded_tweet)
        print()


if __name__ == '__main__':
    main()

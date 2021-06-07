from data_loader import DataLoader
from inference_runner import InferenceRunner
from nn_input_preparer import NNInputPreparer
from nn_model_creator import NNModelCreator
from tweet_cleaner import TweetCleaner
from noiser import DisjointNoiser
from vocab_util import NN_VOCAB_TO_INT

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf

import random

TRAINING_INPUT_FILENAME = '../data/lid_train_lines.txt'


def main():
    print('tf:', tf.__version__)

    random.seed(42)

    raw_tweets = DataLoader(TRAINING_INPUT_FILENAME).load()
    num_training_tweets = len(raw_tweets)
    cleaner = TweetCleaner()
    clean_tweets = [cleaner.clean_tweet(t) for t in raw_tweets]

    noiser = DisjointNoiser()
    noisy_tweets_as_lists = [noiser.add_noise(list(t)) for t in clean_tweets]
    noisy_tweets_readable = [''.join(t) for t in noisy_tweets_as_lists]
    clean_tweets_as_lists = [list(t) for t in clean_tweets]

    model_creator = NNModelCreator()
    training_model = model_creator.create_training_model()

    GENERATOR_BATCH_SIZE = 2048

    nn_input_preparer = NNInputPreparer()
    TRAINING_MODEL_FILENAME = 'models/weights-improvement-4-0.87262.h5'

    IS_TRAINING_RUN = False
    if IS_TRAINING_RUN:
        num_generations = 0

        for de_facto_epoch in range(5):
            gb_training = nn_input_preparer.get_batches(
                clean_tweets_as_lists, noiser, GENERATOR_BATCH_SIZE)

            cp_filepath = f'models/weights-improvement-{de_facto_epoch}-' + "{val_accuracy:.5f}.h5"
            print(cp_filepath)

            checkpoint = ModelCheckpoint(cp_filepath, monitor='val_accuracy', verbose=1,
                                         save_best_only=True, mode='max')

            while True:
                try:
                    noised_batch, originals_batch, originals_delayed_batch = next(gb_training)
                    assert (len(noised_batch) == GENERATOR_BATCH_SIZE)
                    print(noised_batch.shape, originals_batch.shape,
                          originals_delayed_batch.shape)
                    VALIDATION_SPLIT = 0.125
                    FIT_BATCH_SIZE = 32
                    # We take care here so as not to manifest the "Your input ran out of data" warning
                    VALIDATION_STEPS = int(GENERATOR_BATCH_SIZE * VALIDATION_SPLIT) // FIT_BATCH_SIZE
                    TRAINING_STEPS = GENERATOR_BATCH_SIZE // FIT_BATCH_SIZE - VALIDATION_STEPS
                    training_model.fit([noised_batch, originals_delayed_batch],
                                       originals_batch,
                                       batch_size=FIT_BATCH_SIZE,
                                       steps_per_epoch=TRAINING_STEPS,
                                       epochs=1,
                                       validation_split=VALIDATION_SPLIT,
                                       validation_steps=VALIDATION_STEPS,
                                       callbacks=[checkpoint])
                    # https://keras.io/api/models/model_training_apis/ says:
                    # "The validation data is selected from the last samples in the ... data provided"
                    # This means the model is never validated on tweets that we train it on.
                except StopIteration:
                    print('StopIteration')
                    break

                num_generations += 1
                print(num_generations)

        training_model.save(TRAINING_MODEL_FILENAME)

    loaded_training_model = load_model(TRAINING_MODEL_FILENAME)

    encoder_model, decoder_model = model_creator.create_inference_models(loaded_training_model)
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

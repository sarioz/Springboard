from data_loader import DataLoader
from inference_runner import InferenceRunner
from nn_input_preparer import NNInputPreparer
from nn_model_creator import NNModelCreator
from tweet_cleaner import TweetCleaner
from noiser import DisjointNoiser

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf

import random

TRAINING_INPUT_FILENAME = '../data/lid_train_lines.txt'

# TRAINING_MODEL_FILENAME = 'models/weights-improvement-4-0.87262.h5'
# TRAINING_MODEL_FILENAME = 'models/weights-improvement-2-0.85544.h5'
# TRAINING_MODEL_FILENAME = 'models/weights-improvement-1-0.84504.h5'
# TRAINING_MODEL_FILENAME = 'models/weights-improvement-0-0.80569.h5'
TRAINING_MODEL_FILENAME = 'models/trained_model.h5'
IS_TRAINING_RUN = False


def main():
    print('tf:', tf.__version__)

    random.seed(42)

    raw_tweets = DataLoader(TRAINING_INPUT_FILENAME).load()
    cleaner = TweetCleaner()
    clean_tweets = [cleaner.clean_tweet(t) for t in raw_tweets]

    noiser = DisjointNoiser()
    clean_tweets_as_lists = [list(t) for t in clean_tweets]

    model_creator = NNModelCreator()
    training_model = model_creator.create_training_model()

    generator_batch_size = 2048

    nn_input_preparer = NNInputPreparer()

    if IS_TRAINING_RUN:
        num_generations = 0

        for de_facto_epoch in range(5):
            gb_training = nn_input_preparer.get_batches(
                clean_tweets_as_lists, noiser, generator_batch_size)

            cp_filepath = f'models/weights-improvement-{de_facto_epoch}-' + "{val_accuracy:.5f}.h5"
            print(cp_filepath)

            checkpoint = ModelCheckpoint(cp_filepath, monitor='val_accuracy', verbose=1,
                                         save_best_only=True, mode='max')

            while True:
                try:
                    noised_batch, originals_batch, originals_delayed_batch = next(gb_training)
                    assert (len(noised_batch) == generator_batch_size)
                    print(noised_batch.shape, originals_batch.shape,
                          originals_delayed_batch.shape)
                    validation_split = 0.125
                    fit_batch_size = 32
                    # We take care here so as not to manifest the "Your input ran out of data" warning
                    validation_steps = int(generator_batch_size * validation_split) // fit_batch_size
                    training_steps = generator_batch_size // fit_batch_size - validation_steps
                    training_model.fit([noised_batch, originals_delayed_batch],
                                       originals_batch,
                                       batch_size=fit_batch_size,
                                       steps_per_epoch=training_steps,
                                       epochs=1,
                                       validation_split=validation_split,
                                       validation_steps=validation_steps,
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

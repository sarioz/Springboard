import random
import time

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from data_loader import DataLoader
from nn_input_preparer import NNInputPreparer
from nn_model_creator import NNModelCreator
from noiser import DisjointNoiser
from tweet_cleaner import TweetCleaner

TRAINING_INPUT_FILENAME = '../data/lid_train_lines.txt'

EXPERIMENT_NAME = "04.BiLSTMs_2_Dense"

LATENT_DIM = 512
NUM_DE_FACTO_EPOCHS = 50

BASE_DIR = f'models/{EXPERIMENT_NAME}/dim_{LATENT_DIM}/'
FINAL_TRAINED_MODEL_FILENAME = BASE_DIR + 'trained_model.h5'

CONTINUE_TRAINING = True
INITIALLY_COMPLETED_DFEPOCH = 11 if CONTINUE_TRAINING else -1

TRAINING_MODEL_FILENAME_TO_CONTINUE = BASE_DIR + f'dfepoch_{INITIALLY_COMPLETED_DFEPOCH}_end.h5'


def main():
    print('tf:', tf.__version__)

    random.seed(42)

    raw_tweets = DataLoader(TRAINING_INPUT_FILENAME).load()
    cleaner = TweetCleaner()
    clean_tweets = [cleaner.clean_tweet(t) for t in raw_tweets]

    noiser = DisjointNoiser()
    clean_tweets_as_lists = [list(t) for t in clean_tweets]

    if CONTINUE_TRAINING:
        training_model = load_model(TRAINING_MODEL_FILENAME_TO_CONTINUE)
    else:
        model_creator = NNModelCreator(latent_dim=LATENT_DIM)
        training_model = model_creator.create_training_model()

    generator_batch_size = 2048

    nn_input_preparer = NNInputPreparer()

    num_generations = 0

    print(time.ctime())

    for de_facto_epoch in range(INITIALLY_COMPLETED_DFEPOCH + 1, NUM_DE_FACTO_EPOCHS):
        gb_training = nn_input_preparer.get_batches(
            clean_tweets_as_lists, noiser, generator_batch_size)

        cp_filepath = BASE_DIR + f'dfepoch_{de_facto_epoch}_' + "{val_accuracy:.5f}.h5"

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
                break

            num_generations += 1
            print(f'num_generations: {num_generations}')

        print(time.ctime())
        print(f'End of de facto epoch {de_facto_epoch} - saving model')

        training_model.save(BASE_DIR + f'dfepoch_{de_facto_epoch}_end.h5')

    training_model.save(FINAL_TRAINED_MODEL_FILENAME)


if __name__ == '__main__':
    main()

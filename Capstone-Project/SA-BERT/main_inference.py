import numpy as np
import tensorflow as tf
from bert import BertModelLayer
from tensorflow.keras.models import load_model
from numpy.random import default_rng

from bert_token_converter import BertTokenConverter
from labeled_data_loader import LabeledDataLoader
from main_training import BERT_PRETRAINED_MODEL_DIR
from nn_input_preparer import NNInputPreparer
from vocab_util import TargetVocabUtil

EXPERIMENT_NAME = f'01_MLBERT_SA_hello'
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/ep_2_valacc_0.56639.h5'

TRAINING_INPUT_FILENAME = '../data/sa/train.conll'
DEV_INPUT_FILENAME = '../data/sa/dev.conll'

MAX_SEQ_LEN = 128


def main_inference():
    print(f'Using TensorFlow version {tf.__version__}')
    print(f'Loading model {TRAINING_MODEL_FILENAME}')
    loaded_training_model = load_model(TRAINING_MODEL_FILENAME,
                                       custom_objects={"BertModelLayer": BertModelLayer})
    loaded_training_model.summary()

    rng = default_rng()

    tvu = TargetVocabUtil()
    btc = BertTokenConverter(model_dir=BERT_PRETRAINED_MODEL_DIR, tvu=tvu)
    nn_input_preparer = NNInputPreparer(tvu=tvu, max_seq_len=MAX_SEQ_LEN)

    for input_filename in [TRAINING_INPUT_FILENAME, DEV_INPUT_FILENAME]:
        k = 500
        print(f'processing the first {k} tweets from {input_filename}')
        loader = LabeledDataLoader(input_filename)
        tweets = loader.parse_tokens_and_labels(loader.load_lines())[:k]
        tweets = btc.convert(tweets)
        tweets = btc.convert_to_ids(tweets)
        tweets = btc.prepend_cls(tweets)
        tweets = nn_input_preparer.filter_out_long_sequences(tweets)

        irregular_inputs = [tweet[0] for tweet in tweets]
        rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_inputs)
        print('rectangular_inputs.shape:', rectangular_inputs.shape)
        rectangular_targets = [tweet[1] for tweet in tweets]

        predicted_probabilities_sequence = loaded_training_model.predict(rectangular_inputs)

        matches_argmax = matches_sampling = 0
        print('predicted_label_argmax, predicted_human_argmax, predicted_human_sampling, target')

        for predicted_probabilities, target in zip(predicted_probabilities_sequence, rectangular_targets):
            predicted_label_argmax = np.argmax(predicted_probabilities)
            # predicted_human_argmax = tvu.raw_sentiment_labels[predicted_label_argmax]

            if predicted_label_argmax == target:
                matches_argmax += 1

            predicted_label_sampling = rng.choice(3, 1, p=predicted_probabilities)[0]
            # predicted_human_sampling = tvu.raw_sentiment_labels[predicted_label_sampling]
            if predicted_label_sampling == target:
                matches_sampling += 1

            # print(f'{predicted_human_argmax}, {predicted_human_sampling}, {tvu.raw_sentiment_labels[target]}')

        print('accuracy_argmax:', matches_argmax / len(rectangular_targets))
        print('accuracy_sampling:', matches_sampling / len(rectangular_targets))


if __name__ == '__main__':
    main_inference()

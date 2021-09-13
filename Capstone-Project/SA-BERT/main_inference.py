import numpy as np
import tensorflow as tf
from bert import BertModelLayer
from tensorflow.keras.models import load_model
from tqdm import tqdm

from bert_token_converter import BertTokenConverter
from labeled_data_loader import LabeledDataLoader
from main_training import BERT_PRETRAINED_MODEL_DIR
from nn_input_preparer import NNInputPreparer
from vocab_util import TargetVocabUtil

EXPERIMENT_NAME = f'04_MLBERT_SA_AdamW_on_dev'
MAX_SEQ_LEN = 128
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/ep_5_valacc_0.55783.h5'

TRAINING_INPUT_FILENAME = '../data/sa/train.conll'
DEV_INPUT_FILENAME = '../data/sa/dev.conll'


def main_inference():
    print(f'Using TensorFlow version {tf.__version__}')
    print(f'Loading model {TRAINING_MODEL_FILENAME}')
    trained_model = load_model(TRAINING_MODEL_FILENAME,
                               custom_objects={"BertModelLayer": BertModelLayer})
    trained_model.summary()

    tvu = TargetVocabUtil()
    btc = BertTokenConverter(model_dir=BERT_PRETRAINED_MODEL_DIR, tvu=tvu)
    nn_input_preparer = NNInputPreparer(tvu=tvu, max_seq_len=MAX_SEQ_LEN)

    for input_filename in [DEV_INPUT_FILENAME]:
        loader = LabeledDataLoader(input_filename)
        tweets = loader.parse_tokens_and_labels(loader.load_lines())
        tweets = btc.convert(tweets)
        tweets = btc.convert_to_ids(tweets)
        tweets = btc.prepend_cls(tweets)
        tweets = nn_input_preparer.filter_out_long_sequences(tweets)
        print(f'Processing all not-too-long {len(tweets)} tweets from {input_filename}')

        irregular_inputs = [tweet[0] for tweet in tweets]
        rectangular_targets = [tweet[1] for tweet in tweets]

        argmax_confusion_matrix = np.zeros((tvu.get_output_vocab_size(), tvu.get_output_vocab_size()), dtype=int)
        expected_sampling_confusion_matrix = np.zeros((tvu.get_output_vocab_size(), tvu.get_output_vocab_size()))

        num_correct_argmax_predictions = 0
        expected_sampling_accuracy_sum = 0.0

        for irregular_input, target_index in tqdm(zip(irregular_inputs, rectangular_targets)):
            rectangular_input_singleton = nn_input_preparer.rectangularize_inputs([irregular_input])
            predicted_probabilities = trained_model(rectangular_input_singleton)[0]
            # the predicted index if we take the class with the largest probability
            argmax_index = np.argmax(predicted_probabilities)
            if argmax_index == target_index:
                num_correct_argmax_predictions += 1
            argmax_confusion_matrix[target_index][argmax_index] += 1

            # rhs is the probability of guessing target if we sample according to predicted probabilities
            expected_sampling_accuracy_sum += tf.keras.backend.get_value(predicted_probabilities[target_index])
            for i in range(tvu.get_output_vocab_size()):
                expected_sampling_confusion_matrix[target_index][i] += predicted_probabilities[i]

        num_tweets_in_dataset = len(rectangular_targets)

        print(f'Argmax accuracy for {input_filename}:',
              num_correct_argmax_predictions / num_tweets_in_dataset)
        print(f'Expected sampling accuracy for {input_filename}:',
              expected_sampling_accuracy_sum / num_tweets_in_dataset)

        print(f"Argmax confusion matrix of targets vs predicted for {input_filename}:\n"
              f"{tvu.raw_sentiment_labels}\n",
              argmax_confusion_matrix)
        print(f"Expected sampling confusion matrix of targets vs predicted for {input_filename}:\n"
              f"{tvu.raw_sentiment_labels}\n",
              expected_sampling_confusion_matrix)


if __name__ == '__main__':
    main_inference()

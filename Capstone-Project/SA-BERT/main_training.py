import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from bert_token_converter import BertTokenConverter
from labeled_data_loader import LabeledDataLoader
from model_creator import BertModelCreator
from nn_input_preparer import NNInputPreparer
from vocab_util import TargetVocabUtil

BERT_PRETRAINED_MODEL_DIR = "../multi_cased_L-12_H-768_A-12/"
MAX_SEQ_LEN = 128
EXPERIMENT_NAME = f'04_MLBERT_SA_AdamW_on_dev'
MAX_EPOCHS = 50
BASE_DIR = f'models/{EXPERIMENT_NAME}/'
FINAL_TRAINED_MODEL_FILENAME = BASE_DIR + 'trained_model.h5'
CONTINUE_TRAINING = False
INITIAL_EPOCH = 10 if CONTINUE_TRAINING else 0
TRAINING_MODEL_FILENAME_TO_CONTINUE = BASE_DIR + 'ep_6_valacc_0.87869.h5'
TR_INPUT_FILENAME = '../data/sa/train.conll'
DEV_INPUT_FILENAME = '../data/sa/dev.conll'


def main_training():
    print(f'Using TensorFlow version {tf.__version__}')

    tvu = TargetVocabUtil()
    btc = BertTokenConverter(model_dir=BERT_PRETRAINED_MODEL_DIR, tvu=tvu)
    nn_input_preparer = NNInputPreparer(tvu=tvu, max_seq_len=MAX_SEQ_LEN)

    tr_loader = LabeledDataLoader(TR_INPUT_FILENAME)
    tr_labeled_tweets = tr_loader.parse_tokens_and_labels(tr_loader.load_lines())
    tr_labeled_tweets = btc.convert(tr_labeled_tweets)
    tr_labeled_tweets = btc.convert_to_ids(tr_labeled_tweets)
    tr_labeled_tweets = btc.prepend_cls(tr_labeled_tweets)
    tr_labeled_tweets = nn_input_preparer.filter_out_long_sequences(tr_labeled_tweets)
    print(f'Processing all not-too-long {len(tr_labeled_tweets)} tweets from {TR_INPUT_FILENAME}')
    tr_irregular_inputs = [tweet[0] for tweet in tr_labeled_tweets]
    tr_rectangular_targets = [tweet[1] for tweet in tr_labeled_tweets]
    tr_rectangular_inputs = nn_input_preparer.rectangularize_inputs(tr_irregular_inputs)
    tr_targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(tr_rectangular_targets)

    if CONTINUE_TRAINING:
        print('Continuing training from', TRAINING_MODEL_FILENAME_TO_CONTINUE)
        model = load_model(TRAINING_MODEL_FILENAME_TO_CONTINUE)
        model.summary()
    else:
        print("Commencing new training run")
        model_creator = BertModelCreator(model_dir=BERT_PRETRAINED_MODEL_DIR,
                                         tvu=tvu,
                                         max_seq_len=MAX_SEQ_LEN)
        model = model_creator.create_model()

    cp_filepath = BASE_DIR + 'ep_{epoch}_valacc_{val_accuracy:.5f}.h5'

    checkpoint = ModelCheckpoint(cp_filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=False)

    dev_loader = LabeledDataLoader(DEV_INPUT_FILENAME)
    dev_labeled_tweets = tr_loader.parse_tokens_and_labels(dev_loader.load_lines())
    dev_labeled_tweets = btc.convert(dev_labeled_tweets)
    dev_labeled_tweets = btc.convert_to_ids(dev_labeled_tweets)
    dev_labeled_tweets = btc.prepend_cls(dev_labeled_tweets)
    dev_labeled_tweets = nn_input_preparer.filter_out_long_sequences(dev_labeled_tweets)
    print(f'Processing all not-too-long {len(dev_labeled_tweets)} tweets from {DEV_INPUT_FILENAME}')
    dev_irregular_inputs = [tweet[0] for tweet in dev_labeled_tweets]
    dev_rectangular_targets = [tweet[1] for tweet in dev_labeled_tweets]
    dev_rectangular_inputs = nn_input_preparer.rectangularize_inputs(dev_irregular_inputs)
    dev_targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(dev_rectangular_targets)

    model.fit(tr_rectangular_inputs, tr_targets_one_hot_encoded, batch_size=32,
              initial_epoch=INITIAL_EPOCH,
              epochs=MAX_EPOCHS,
              validation_data=(dev_rectangular_inputs, dev_targets_one_hot_encoded),
              callbacks=[checkpoint])

    model.save(FINAL_TRAINED_MODEL_FILENAME)


if __name__ == '__main__':
    main_training()

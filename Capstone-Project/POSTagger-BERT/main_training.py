import tensorflow as tf
from bert import BertModelLayer

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from vocab_util import TargetVocabUtil
from labeled_data_loader import LabeledDataLoader
from model_creator import BertModelCreator
from bert_token_converter import BertTokenConverter

from nn_input_preparer import NNInputPreparer

BERT_PRETRAINED_MODEL_DIR = "../multi_cased_L-12_H-768_A-12/"

MAX_SEQ_LEN = 128

EXPERIMENT_NAME = f'05_MLBERT_frozen_on_dev'
MAX_EPOCHS = 50

BASE_DIR = f'models/{EXPERIMENT_NAME}/'
FINAL_TRAINED_MODEL_FILENAME = BASE_DIR + 'trained_model.h5'

CONTINUE_TRAINING = False
INITIAL_EPOCH = 2 if CONTINUE_TRAINING else 0
TRAINING_MODEL_FILENAME_TO_CONTINUE = BASE_DIR + 'ep_2_valacc_0.98046.h5'

TRAINING_INPUT_FILENAME = '../data/pos/train.conll'
DEV_INPUT_FILENAME = '../data/pos/dev.conll'


def main_training():
    print(f'Using TensorFlow version {tf.__version__}')
    loader = LabeledDataLoader(TRAINING_INPUT_FILENAME)
    tweets = loader.parse_raw_tokens_and_labels(loader.load_lines())

    tvu = TargetVocabUtil()

    tokenizer = BertTokenConverter(model_dir=BERT_PRETRAINED_MODEL_DIR, tvu=tvu)
    tweets = tokenizer.convert_to_tokens(tweets)
    tweets = tokenizer.convert_to_ids(tweets)

    nn_input_preparer = NNInputPreparer(tvu, max_seq_len=MAX_SEQ_LEN)
    tweets = nn_input_preparer.filter_out_long_sequences(tweets)

    print(f'Training and validating on {len(tweets)} tweets, each no longer than {MAX_SEQ_LEN} tokens')

    irregular_inputs = [[item[0] for item in tweet] for tweet in tweets]
    irregular_targets = [[item[1] for item in tweet] for tweet in tweets]

    rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_inputs)
    rectangular_targets = nn_input_preparer.rectangularize_targets(irregular_targets)

    targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(rectangular_targets)

    if CONTINUE_TRAINING:
        print('Continuing training from', TRAINING_MODEL_FILENAME_TO_CONTINUE)
        model = load_model(TRAINING_MODEL_FILENAME_TO_CONTINUE,
                           custom_objects={"BertModelLayer": BertModelLayer})
        model.summary()
    else:
        print('Commencing new training run')
        model_creator = BertModelCreator(model_dir=BERT_PRETRAINED_MODEL_DIR,
                                         tvu=tvu,
                                         max_seq_len=MAX_SEQ_LEN,
                                         freeze_bert_layer=True)
        model = model_creator.create_model()

    cp_filepath = BASE_DIR + 'ep_{epoch}_valacc_{val_accuracy:.5f}.h5'

    dev_loader = LabeledDataLoader(DEV_INPUT_FILENAME)
    tweets = loader.parse_raw_tokens_and_labels(dev_loader.load_lines())
    tweets = tokenizer.convert_to_tokens(tweets)
    tweets = tokenizer.convert_to_ids(tweets)
    tweets = nn_input_preparer.filter_out_long_sequences(tweets)
    print(f'processing all {len(tweets)} not-too-long tweets from {DEV_INPUT_FILENAME}')
    dev_irregular_inputs = [[item[0] for item in tweet] for tweet in tweets]
    # print('proportion of non-pad in dev:', sum(len(tweet) for tweet in tweets) / (MAX_SEQ_LEN * len(tweets)))
    dev_irregular_targets = [[item[1] for item in tweet] for tweet in tweets]
    dev_rectangular_inputs = nn_input_preparer.rectangularize_inputs(dev_irregular_inputs)
    dev_rectangular_targets = nn_input_preparer.rectangularize_targets(dev_irregular_targets)
    dev_targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(dev_rectangular_targets)

    checkpoint = ModelCheckpoint(cp_filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=False)

    model.fit(rectangular_inputs, targets_one_hot_encoded, batch_size=32,
              initial_epoch=INITIAL_EPOCH,
              epochs=MAX_EPOCHS,
              validation_data=(dev_rectangular_inputs, dev_targets_one_hot_encoded),
              callbacks=[checkpoint])

    model.save(FINAL_TRAINED_MODEL_FILENAME)


if __name__ == '__main__':
    main_training()

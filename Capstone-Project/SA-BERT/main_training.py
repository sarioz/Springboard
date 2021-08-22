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
EXPERIMENT_NAME = f'03_MLBERT_SA_AdamW'
MAX_EPOCHS = 10
BASE_DIR = f'models/{EXPERIMENT_NAME}/'
FINAL_TRAINED_MODEL_FILENAME = BASE_DIR + 'trained_model.h5'
CONTINUE_TRAINING = False
INITIAL_EPOCH = 10 if CONTINUE_TRAINING else 0
TRAINING_MODEL_FILENAME_TO_CONTINUE = BASE_DIR + 'ep_6_valacc_0.87869.h5'


def main_training():
    print(f'Using TensorFlow version {tf.__version__}')

    loader = LabeledDataLoader('../data/sa/train.conll')
    labeled_tweets = loader.parse_tokens_and_labels(loader.load_lines())

    tvu = TargetVocabUtil()

    btc = BertTokenConverter(model_dir=BERT_PRETRAINED_MODEL_DIR, tvu=tvu)

    converted_tweets = btc.convert(labeled_tweets)
    converted_tweets = btc.convert_to_ids(converted_tweets)
    converted_tweets = btc.prepend_cls(converted_tweets)

    nn_input_preparer = NNInputPreparer(tvu=tvu, max_seq_len=MAX_SEQ_LEN)
    converted_tweets = nn_input_preparer.filter_out_long_sequences(converted_tweets)

    irregular_inputs = [tweet[0] for tweet in converted_tweets]
    rectangular_targets = [tweet[1] for tweet in converted_tweets]

    rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_inputs)
    targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(rectangular_targets)

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

    model.fit(rectangular_inputs, targets_one_hot_encoded, batch_size=32,
              initial_epoch=INITIAL_EPOCH,
              epochs=MAX_EPOCHS,
              validation_split=0.1, callbacks=[checkpoint])

    model.save(FINAL_TRAINED_MODEL_FILENAME)


if __name__ == '__main__':
    main_training()

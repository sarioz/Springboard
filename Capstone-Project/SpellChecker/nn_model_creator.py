from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM

from vocab_util import LEN_NN_VOCAB


class NNModelCreator:
    def __init__(self, latent_dim=256):
        self.latent_dim = latent_dim

    def create_training_model(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, LEN_NN_VOCAB))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, LEN_NN_VOCAB))

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(LEN_NN_VOCAB, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model

    def create_inference_models(self, training_model):
        encoder_inputs = training_model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = training_model.input[1]  # input_2
        decoder_state_input_h = Input(shape=(self.latent_dim,), name="decoder_state_input_h")
        decoder_state_input_c = Input(shape=(self.latent_dim,), name="decoder_state_input_c")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = training_model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = training_model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

        return encoder_model, decoder_model

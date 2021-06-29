from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Bidirectional, Concatenate, Dense, LSTM

from vocab_util import LEN_NN_VOCAB


class NNModelCreator:
    def __init__(self, latent_dim=256):
        self.latent_dim = latent_dim

    def create_chollet_training_model(self) -> Model:
        """Create training model according to Chollet's https://keras.io/examples/nlp/lstm_seq2seq/"""
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

    def derive_chollet_inference_models(self, training_model: Model) -> (Model, Model):
        """Create inference models according to Chollet's https://keras.io/examples/nlp/lstm_seq2seq/"""
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

    def create_4_lstm_training_model(self) -> Model:
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, LEN_NN_VOCAB), name="encoder_inputs")
        encoder_layer_1 = LSTM(self.latent_dim, return_sequences=True, name="encoder_layer_1")
        encoder_layer_1_outputs = encoder_layer_1(encoder_inputs)
        encoder_layer_2 = LSTM(self.latent_dim, return_state=True, name="encoder_layer_2")
        encoder_last_layer_outputs, state_h, state_c = encoder_layer_2(encoder_layer_1_outputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_last_layer_states = [state_h, state_c]

        # Set up the decoder, using `encoder_last_layer_states` as initial state.
        decoder_inputs = Input(shape=(None, LEN_NN_VOCAB), name="decoder_inputs")

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_layer_1 = LSTM(self.latent_dim, return_sequences=True, return_state=True, name="decoder_layer_1")
        decoder_layer_1_outputs, _, _ = decoder_layer_1(decoder_inputs, initial_state=encoder_last_layer_states)
        decoder_layer_2 = LSTM(self.latent_dim, return_sequences=True, return_state=True, name="decoder_layer_2")
        decoder_layer_2_outputs, _, _ = decoder_layer_2(decoder_layer_1_outputs)

        decoder_dense = Dense(LEN_NN_VOCAB, activation="softmax", name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_layer_2_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        print('training_model:', model.summary())
        print()

        return model

    def derive_4_lstm_inference_models(self, training_model: Model) -> (Model, Model):
        encoder_inputs = training_model.input[0]
        _, state_h_enc, state_c_enc = training_model.get_layer(name="encoder_layer_2").output
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = training_model.input[1]
        decoder_state_input_h = Input(shape=(self.latent_dim,), name="decoder_state_input_h")
        decoder_state_input_c = Input(shape=(self.latent_dim,), name="decoder_state_input_c")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm_1 = training_model.get_layer("decoder_layer_1")
        decoder_lstm_1_outputs, _, _ = decoder_lstm_1(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_lstm_2 = training_model.get_layer("decoder_layer_2")
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm_2(
            decoder_lstm_1_outputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = training_model.get_layer("decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

        return encoder_model, decoder_model

    def create_bilstm_training_model(self) -> Model:
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, LEN_NN_VOCAB))
        encoder = Bidirectional(LSTM(self.latent_dim, return_state=True), name="encoder_bilstm")
        _, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, LEN_NN_VOCAB))

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.latent_dim * 2, return_sequences=True, return_state=True, name="decoder_lstm")
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(LEN_NN_VOCAB, activation="softmax", name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model

    def derive_bilstm_inference_models(self, training_model: Model) -> (Model, Model):
        encoder_inputs = training_model.input[0]
        _, forward_h, forward_c, backward_h, backward_c = training_model.get_layer(name="encoder_bilstm").output
        state_h_enc = Concatenate()([forward_h, backward_h])
        state_c_enc = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = training_model.input[1]
        decoder_state_input_h = Input(shape=(self.latent_dim * 2,), name="decoder_state_input_h")
        decoder_state_input_c = Input(shape=(self.latent_dim * 2,), name="decoder_state_input_c")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = training_model.get_layer("decoder_lstm")
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = training_model.get_layer(name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

        return encoder_model, decoder_model

    def create_bilstm_2dense_training_model(self) -> Model:
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, LEN_NN_VOCAB))
        encoder = Bidirectional(LSTM(self.latent_dim, return_state=True), name="encoder_bilstm")
        _, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
        state_h = Concatenate(name="concatenate_1")([forward_h, backward_h])
        state_c = Concatenate(name="concatenate_2")([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, LEN_NN_VOCAB))

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.latent_dim * 2, return_sequences=True, return_state=True, name="decoder_lstm")
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense_1 = Dense(LEN_NN_VOCAB, activation="tanh", name="decoder_dense_1")
        decoder_outputs = decoder_dense_1(decoder_outputs)
        decoder_dense_2 = Dense(LEN_NN_VOCAB, activation="softmax", name="decoder_dense_2")
        decoder_outputs = decoder_dense_2(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model

    def derive_bilstm_2dense_inference_models(self, training_model: Model) -> (Model, Model):
        encoder_inputs = training_model.input[0]
        _, forward_h, forward_c, backward_h, backward_c = training_model.get_layer(name="encoder_bilstm").output
        state_h_enc = Concatenate(name="concatenate_1")([forward_h, backward_h])
        state_c_enc = Concatenate(name="concatenate_2")([forward_c, backward_c])
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = training_model.input[1]
        decoder_state_input_h = Input(shape=(self.latent_dim * 2,), name="decoder_state_input_h")
        decoder_state_input_c = Input(shape=(self.latent_dim * 2,), name="decoder_state_input_c")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = training_model.get_layer("decoder_lstm")
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense_1 = training_model.get_layer(name="decoder_dense_1")
        decoder_outputs = decoder_dense_1(decoder_outputs)
        decoder_dense_2 = training_model.get_layer(name="decoder_dense_2")
        decoder_outputs = decoder_dense_2(decoder_outputs)

        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

        return encoder_model, decoder_model

    def create_training_model(self) -> Model:
        m = self.create_bilstm_2dense_training_model()
        print('training model:')
        m.summary()

        return m

    def derive_inference_models(self, training_model: Model) -> (Model, Model):
        print('loaded training_model:')
        training_model.summary()
        encoder_model, decoder_model = self.derive_bilstm_2dense_inference_models(training_model)
        print('derived encoder_model:')
        encoder_model.summary()
        print('derived decoder_model:')
        decoder_model.summary()

        return encoder_model, decoder_model

from vocab_util import LEN_NN_VOCAB, NN_VOCAB_TO_INT, NN_VOCAB_TUPLE

import numpy as np

class InferenceRunner:

    def __init__(self, encoder_model, decoder_model):
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def decode_sequence(self, input_seq):
        """This function is doing a bit too much. Should break it up."""
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)
        # print('states values:', states_value)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, LEN_NN_VOCAB))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, NN_VOCAB_TO_INT['<GO>']] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_tweet = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = NN_VOCAB_TUPLE[sampled_token_index]
            decoded_tweet += sampled_char

            # Exit condition: either hit max length or find stop character.
            if sampled_char == '<EOT>' or len(decoded_tweet) > 100:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, LEN_NN_VOCAB))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]

        return decoded_tweet
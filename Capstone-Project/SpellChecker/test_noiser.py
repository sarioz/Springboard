import random
import unittest

from noiser import DisjointNoiser


class NoiserTestCase(unittest.TestCase):

    def test_noisings(self):
        random.seed(42)
        noiser = DisjointNoiser()

        clean_tweets = [
            'echa un vistazo a nuestros planes móviles y descubre las opciones .',
            'pues yo solo digo que hoy os enamoráis de mí'
        ]

        noisy_tweets_as_lists = [noiser.add_noise(list(t)) for t in clean_tweets]
        noisy_tweets_readable = [''.join(t) for t in noisy_tweets_as_lists]

        # The resulting noisy tweets depend on the random seed set above.
        self.assertEqual([
                'ch un vistazo a nstrs planes mviles y descubre las opcins .',
                'pues yo solo digo que hoy os enamorais de mi'
            ],
            noisy_tweets_readable)


if __name__ == '__main__':
    unittest.main()

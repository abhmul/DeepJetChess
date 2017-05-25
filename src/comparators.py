from cachetools import LRUCache
import numpy as np
from np_board_utils import create_input

class Comparator(object):

    def compare(self, a, b):
        """
        Returns < 0 if a < b, 0 if a = b, and > 0 if a > b.
        """
        raise NotImplementedError()

    def pcompare(self, a_arr, b_arr):
        raise NotImplementedError()

    def _extrema(self, vals, extrema_cmp):
        """
        Calculates the max in vals using the compare method.
        """
        extrema_val = None
        for val in vals:
            if extrema_val is None:
                extrema_val = val

            elif extrema_cmp(self.compare(extrema_val, val)):
                extrema_val = val

        return extrema_val

    def min(self, vals):
        return self._extrema(vals, lambda x: x > 0)

    def max(self, vals):
        return self._extrema(vals, lambda x: x < 0)

    def greater(self, a, b):
        return self.compare(a, b) > 0

    def equal(self, a, b):
        return self.compare(a, b) == 0

    def less(self, a, b):
        return self.compare(a, b) < 0

    def greater_equal(self, a, b):
        return self.compare(a, b) >= 0

    def less_equal(self, a, b):
        return self.compare(a, b) <= 0

class DeepJetChess(Comparator):

    def __init__(self, embedder, comparer, cache=None, cache_size=None):
        self.embedder = embedder
        self.comparer = comparer
        # This cache only stores zobrist hashes, since with 16 byte
        # hashes, there is 1 in 1.84e19 of a collision
        self._cache = LRUCache(maxsize=cache_size) if cache is None else cache

    def load_weights(self, weights_path):
        self.embedder.load_weights(weights_path, by_name=True)
        self.comparer.load_weights(weights_path, by_name=True)

    def _filter_uncached(self, np_boards):
        return [np_board for np_board in np_boards if hash(np_board) not in self._cache]

    def embed(self, orig_np_boards):
        np_boards = self._filter_uncached(orig_np_boards)
        # Just return if we've already seen all of these boards
        if len(np_boards) == 0:
            return

        batch = np.concatenate([np_board.create_input() for np_board in np_boards], axis=0)
        # Make sure the batch was created properly
        assert(batch.shape == (len(np_boards), 8, 8, 13))

        posvecs = self.embedder.predict_on_batch(batch)
        # Cache the outputs
        for i, np_board in enumerate(np_boards):
            self._cache[hash(np_board)] = posvecs[i]

    def compare(self, a, b):
        """
        Applies Model to figure which position, a or b, is better
        Arguments:
        a -- an 8 x 8 NpBoard (player is white-oriented and positive)
        b -- an 8 x 8 NpBoard

        Returns:
            -- < 0 if a is a worse position than b
            -- == 0 if a is an equivalent position to b
            -- > 0 if a is a better position than b
        """
        # Check if the array is a win or loss
        if a.is_win() or b.is_loss():
            return 1
        if b.is_win() or a.is_loss():
            return -1

        # Make sure we were passed valid inputs
        assert(a.np_board.shape == (8, 8) and b.np_board.shape == (8, 8))

        # No edge case, we need to get the cached embeddings for a and b
        if hash(a) not in self._cache:
            a_embedding = self.embed([a])
        if hash(b) not in self._cache:
            b_embedding = self.embed([b])
        a_embedding = self._cache[hash(a)][np.newaxis]
        b_embedding = self._cache[hash(b)][np.newaxis]
        return self.comparer.predict_on_batch([a_embedding, b_embedding])[0] * 2 - 1

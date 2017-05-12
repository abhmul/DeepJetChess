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

    def __init__(self, model):
        self.model = model

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

        # No edge case, we need to prepare a and b for model
        a_in = a.create_input()
        b_in = b.create_input()
        return self.model.predict_on_batch([a_in, b_in])[0] * 2 - 1

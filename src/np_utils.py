import numpy as np

def frontpad(arr, value, pad=1):
    new_arr = np.empty((len(arr) + pad,), dtype=arr.dtype)
    new_arr[:pad] = value
    new_arr[pad:] = arr
    return new_arr

def backpad(arr, value, pad=1):
    new_arr = np.empty((len(arr) + pad,), dtype=arr.dtype)
    new_arr[-pad:] = value
    new_arr[:-pad] = arr
    return new_arr

def symmetricpad(arr, value1, value2=None, pad=1):
    new_arr = np.empty((len(arr) + 2 * pad,), dtype=arr.dtype)
    if value2 is None:
        value2 = value1
    new_arr[-pad:] = value2
    new_arr[:pad] = value1
    new_arr[pad:-pad] = arr
    return new_arr

import sys
import h5py

from np_board_utils import sb2array, array2b, make_gn_child

def expand_h5(fname_out, fname_in, chunksize=1000):

    # Get the input file
    print("Opening file %s" % arg)
    h5f = h5py.File(arg, 'r')
    # Get the number of boards in the file
    X_in, M_in, W_in, C_in, E_in = [h5f[group] for group in ('X', 'M', 'W', 'C', 'E')])
    num_boards = X_in.shape[0]
    legal_dataset = num_boards * 50

    X = g.create_dataset('X', (legal_dataset, 8, 8), dtype='i8', maxshape=(None, 8, 8))
    L = g.create_dataset('M', (legal_dataset,), dtype='i8', maxshape=(None,))
    M = g.create_dataset('M', (legal_dataset,), data=M_in, dtype='i8', maxshape=(None,))
    W = g.create_dataset('W', (legal_dataset,), data=W_in, dtype='i8', maxshape=(None,))
    C = g.create_dataset('C', (legal_dataset, max_legals, 4), data=C_in, dtype='i8', maxshape=(None, 4))
    E = g.create_dataset('E', (legal_dataset, max_legals, 2), data=E_in, dtype='i8', maxshape=(None, 2))

    L_arr = np.empty((num_boards,), dtype= np.int8)
    for chunk_offset in range(0, num_boards, chunksize):
        # Load the chunk into memory
        X_arr = np.array(X_in[chunk_offset:chunk_offset+chunksize], dtype=np.int8)
        M_arr = np.array(M_in[chunk_offset:chunk_offset+chunksize], dtype=np.int8)
        W_arr = np.array(W_in[chunk_offset:chunk_offset+chunksize], dtype=np.int8)
        C_arr = np.array(C_in[chunk_offset:chunk_offset+chunksize], dtype=np.int8)
        E_arr = np.array(E_in[chunk_offset:chunk_offset+chunksize], dtype=np.int8)
        for i in range(X_arr.shape[0]):
            board = array2b(X_arr[i], M_arr[i], C_arr[i], E_arr[i])
            # Make the legal moves
            gn_children = [make_gn_child(gn, move) for move in board.legal_moves]
            x_legals = np.zeros((len(gn_children), 8, 8))

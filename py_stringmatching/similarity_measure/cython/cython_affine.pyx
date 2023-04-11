
import numpy as np
from py_stringmatching.similarity_measure.cython.cython_utils import float_max_two
from py_stringmatching.similarity_measure.cython.cython_utils import float_max_three



def affine(unicode string1, unicode string2, float main_gap_start, float main_gap_continuation, sim_func ):

    cdef float gap_start = - main_gap_start
    cdef float gap_continuation = - main_gap_continuation
    cdef int len_str1 = len(string1)
    cdef int len_str2 = len(string2)
    cdef int i=0, j=0
    cdef double[:, :] m = np.zeros((len_str1 + 1, len_str2 + 1), dtype=np.double)
    cdef double[:, :] x = np.zeros((len_str1 + 1, len_str2 + 1), dtype=np.double)
    cdef double[:, :] y = np.zeros((len_str1 + 1, len_str2 + 1), dtype=np.double)

    # DP initialization
    for i from 1 <= i < (len_str1+1):
        m[i, 0] = -float(np.inf)
        x[i, 0] = gap_start + (i-1) * gap_continuation
        y[i, 0] = -float(np.inf)
    #
    # # DP initialization
    for j from 1 <= j < (len_str2+1):
        m[0, j] = -float(np.inf)
        x[0, j] = -float(np.inf)
        y[0, j] = gap_start + (j-1) * gap_continuation


    # affine gap calculation using DP
    for i from 1 <= i < (len_str1 + 1):
        for j from 1 <= j < (len_str2 + 1):
            # best score between x_1....x_i and y_1....y_j
                # given that x_i is aligned to y_j
            m[i, j] = (sim_func(string1[i-1], string2[j-1]) + float_max_three(m[i-1][j-1],
                                                                       x[i-1][j-1], y[i-1][j-1]))
            # the best score given that x_i is aligned to a gap
            x[i, j] = float_max_two((gap_start + m[i-1, j]), (gap_continuation+ x[i-1, j]))
            # the best score given that y_j is aligned to a gap
            y[i, j] = float_max_two((gap_start+ m[i, j-1]), (gap_continuation + y[i, j-1]))

    return float_max_three(m[len_str1, len_str2], x[len_str1, len_str2], y[len_str1, len_str2])


from py_stringmatching.similarity_measure.cython.cython_utils import int_max_two
import numpy as np
cimport numpy as np


#Cython functions to compute the Jaro score
def jaro(unicode string1, unicode string2):
    """Computes the Jaro score between two strings.
        Args:
            string1,string2 (str): Input strings.
        Returns:
            Jaro distance score (float).
    """


    cdef int len_str1 = len(string1), len_str2 = len(string2)
    cdef int max_len = int_max_two(len_str1, len_str2)
    cdef int search_range = (max_len // 2) - 1

    if search_range < 0:
        search_range = 0

    # populating numpy arrays of length as each string with zeros
    cdef int[:] flags_s1 = np.zeros(len_str1, dtype=np.int32)
    cdef int[:] flags_s2 = np.zeros(len_str2, dtype=np.int32)

    cdef int common_chars = 0, low = 0, high = 0, i = 0, j = 0

    # Finding the number of common characters in two strings
    for i from 0 <= i < len_str1:
        low = i - search_range if i > search_range else 0
        high = i + search_range if i + search_range < len_str2 else len_str2 - 1
        for j from low <= j < (high + 1):
            if flags_s2[j] == 0 and string2[j] == string1[i]:
                flags_s1[i] = flags_s2[j] = 1
                common_chars += 1
                break

    if common_chars == 0:
        return 0

    cdef int trans_count = 0, k = 0

    # Finding the number of transpositions and Jaro distance
    for i from 0 <= i < len_str1:
        if flags_s1[i] == 1:
            for j from k <= j < len_str2:
                if flags_s2[j] == 1:
                    k = j + 1
                    break
            if string1[i] != string2[j]:
                trans_count += 1
    trans_count /= 2
    cdef float score = (float(common_chars) / len_str1 + float(common_chars) / len_str2 +
                         (float(common_chars) - trans_count) / float(common_chars)) / 3
    return score





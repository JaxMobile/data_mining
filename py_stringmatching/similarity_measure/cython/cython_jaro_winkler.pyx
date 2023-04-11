
from py_stringmatching.similarity_measure.cython.cython_utils import int_min_two
from py_stringmatching.similarity_measure.cython.cython_jaro import jaro


def jaro_winkler(unicode string1, unicode string2, float prefix_weight):
    """Function to find the Jaro Winkler distance between two strings.
    Args:
        string1,string2 (unicode), prefix_weight (float): Input strings and prefix weight.
    Returns:
        Jaro Winkler distance score (float)
    """
    cdef int i = 0
    cdef float jw_score = jaro(string1, string2)
    cdef int min_len = int_min_two(len(string1), len(string2))
    cdef int j = int_min_two(min_len, 4)

    #Finding the Jaro Winkler distance between two strings
    while i < j and string1[i] == string2[i]:
        i += 1
    if i != 0:
        jw_score += i * prefix_weight * (1 - jw_score)

    return jw_score


import cython


def cython_sim_ident(unicode char1, unicode char2):
    return 1 if char1 == char2 else 0


def int_max_two(int a, int b):
    """Finds the maximum integer of the given two integers.
        Args:
            integer1, integer2 (int): Input integers.
        Returns:
            Maximum integer (int).
    """
    if a > b : return a
    else: return b


def int_max_three(int a, int b, int c):
    """Finds the maximum integer of the given three integers.
        Args:
            integer1, integer2, integer3 (int): Input integers.
        Returns:
            Maximum integer (int).
    """
    cdef int max_int = a
    if b > max_int:
        max_int = b
    if c > max_int:
        max_int = c
    return max_int


def float_max_two(float a, float b):
    """Finds the maximum float of the given two floats.
        Args:
            float1, float2 (float): Input floats.
        Returns:
            Maximum float (float).
    """
    if a > b : return a
    else: return b


def float_max_three(float a, float b, float c):
    """Finds the maximum float of the given two float.
        Args:
            float1, float2, float3 (float): Input floats.
        Returns:
            Maximum float (float).
    """
    cdef float max_float = a
    if b > max_float:
        max_float = b
    if c > max_float:
        max_float = c
    return max_float


def int_min_two(int a, int b):
    """Finds the minimum integer of the given two integers.
    Args:
        integer a,integer b (int): Input integers.
    Returns:
        Minimum integer (int).
    """
    if a > b : return b
    else: return a


def int_min_three(int a, int b, int c):
    """Finds the minimum integer of the given two integers.
    Args:
        integer a, integer b, integer c (int): Input integers.
    Returns:
        Minimum integer (int).
    """
    cdef int min_int = a
    if b < min_int:
        min_int = b
    if c < min_int:
        min_int = c
    return min_int

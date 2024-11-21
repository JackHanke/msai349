import time
from functools import wraps


def timeit(func):
    """
    A decorator that times the execution of a function and prints the time taken.

    Args:
        func (callable): The function to be timed.

    Returns:
        callable: The wrapped function with timing logic.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")
        return result
    return wrapper
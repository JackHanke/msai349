import time
from functools import wraps
import os
import sys


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


def suppress_stdout(func):
    """Decorator to suppress all stdout and stderr during function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save original stdout and stderr file descriptors
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()
        
        # Open /dev/null to redirect output
        with open(os.devnull, 'w') as devnull:
            # Duplicate stdout and stderr to /dev/null
            stdout_fd = os.dup(original_stdout_fd)
            stderr_fd = os.dup(original_stderr_fd)
            try:
                os.dup2(devnull.fileno(), original_stdout_fd)
                os.dup2(devnull.fileno(), original_stderr_fd)

                # Execute the function
                return func(*args, **kwargs)
            finally:
                # Restore original stdout and stderr
                os.dup2(stdout_fd, original_stdout_fd)
                os.dup2(stderr_fd, original_stderr_fd)
                os.close(stdout_fd)
                os.close(stderr_fd)
    return wrapper
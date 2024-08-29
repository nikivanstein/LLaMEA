import numpy as np


class NoCodeException(Exception):
    """Could not extract generated code."""

    pass


def handle_timeout(signum, frame):
    raise TimeoutError

"""Common utility functions"""


def listify(obj):
    """Turn an object into a list, but don't split strings"""
    try:
        assert not isinstance(obj, str)
        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]

    return list(obj)

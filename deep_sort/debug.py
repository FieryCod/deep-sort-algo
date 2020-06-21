import os
from time import time


def dprint(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)

        print("\n>> S " + func.__name__ + " <<\n")
        print(ret)
        print("\n>> E " + func.__name__ + " <<")

        return ret

    return wrapper


def bench(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        ret = func(*args, **kwargs)
        elapsed = time() - t1

        print('Elapsed time for function ' + func.__name__ +
              ' is %f seconds.' % elapsed)

        return ret

    return wrapper

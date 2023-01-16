from time import time

def timeit(fn):
    def wraper(*args, **kwargs):
        t1: float = time()
        res = fn(*args, **kwargs)
        t2: float = time()
        print(f"running {fn.__name__} took {t2 - t1} seconds.")
        return res
    return wraper

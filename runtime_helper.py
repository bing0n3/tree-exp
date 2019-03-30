import time

def time_usage(func):
    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        print("run time: %f" % (end_ts - beg_ts))
        return retval
    return wrapper


@time_usage
def test():
    for i in range(0, 10000):
        pass

if __name__ == "__main__":
    test()
import time

import ray


def time_profiler(f):
    def wrap(*args, **kwargs):
        begin = time.perf_counter()
        wrap_returned_value = f(*args, **kwargs)
        end = time.perf_counter()
        print(f"Running time: {round(end - begin, 3)} time.")
        return wrap_returned_value

    return wrap


@ray.remote
def idle_process(caller_id: int) -> int:
    time.sleep(1)
    print(f"{caller_id} was done.")
    return caller_id


@time_profiler
def run_with_ray():
    futures = [idle_process.remote(i) for i in range(100)]
    results = [ray.get(future) for future in futures]
    print(results)


run_with_ray()

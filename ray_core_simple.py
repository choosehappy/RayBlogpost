import time

import ray


@ray.remote
def idle_process(caller_id: int) -> int:
    time.sleep(1)
    print(f"{caller_id} was done.")
    return caller_id


futures = [idle_process.remote(i) for i in range(100)]
results = [ray.get(future) for future in futures]
print(results)

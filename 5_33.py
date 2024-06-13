import time
import numpy as np
import matplotlib.pyplot as plt

num_iterations = 100
time_results_loop = []

for iteration in range(1, num_iterations + 1):

    start_time = time.time()

    data = np.arange(0, 10000 * iteration, 1)

    my_sum = 0
    for i in data:
        my_sum += i

    end_time = time.time()

    print('{} - :{}'.format(iteration, end_time - start_time))
    time_results_loop.append(end_time - start_time)

    num_iterations = 100
    time_results_np = []

    for iteration in range(1, num_iterations + 1):
        start_time = time.time()

        data = np.arange(0, 10000 * iteration, 1)
        my_sum = np.sum(data)

        end_time = time.time()

        print('{} - :{}'.format(iteration, end_time - start_time))
        time_results_np.append(end_time - start_time)


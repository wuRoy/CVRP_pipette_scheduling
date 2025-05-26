
import numpy as np
import matplotlib.pyplot as plt
from ortools_solver import CVRP_solver
from utils import random_choose_candidate_2
from QAP_solver import calculate_D, calculate_S_E, calculate_D_prime,add_depot,calculate_T
from optimization_methods import row_wise_optimization, greedy_scheduling
import pandas as pd

labware_list =[12,24,96,384]
# enumerate all the two combinations of the labware_list
labware_combinations = []
for i in range(len(labware_list)):
    for j in range(len(labware_list)):
        labware_combinations.append([labware_list[i], labware_list[j]])
#labware_combinations.append([1536,1536])

# establish an empty dataframe to store the results
df = pd.DataFrame(columns=['source_labware', 'dest_labware', 'num_samples', 'repeat', 'unoptimized','rowwise', 'greedy','VRP'])

np.random.seed(0)
for labware_combination in labware_combinations: 

    source_dim = labware_combination[0]
    dest_dim = labware_combination[1]
    print(f'source_dim={source_dim}, dest_dim={dest_dim}')
    for r in range(3):
        print('repeat:',r+1)
        stats = []
        for i in range(1, 11):

            num_candidates = dest_dim * i - 5
            num_candidates = int(num_candidates)
            print(f'num_candidates={num_candidates}')
            a = random_choose_candidate_2(source_dim,dest_dim,num_candidates)
            a[a>0] = 1
            jobs = np.argwhere(a)
            D_S = calculate_D(a.shape[0])
            D_D = calculate_D(a.shape[1])
            S, E = calculate_S_E(a)   
            # calculate distance matrix
            D_prime = calculate_D_prime(D_S,D_D, S, E)
            D_prime = add_depot(D_prime)
            # VRP solver
            VRP_distance, VRP_recorder = CVRP_solver(D_prime.astype(np.int64), solving_time =20)
            print(f'VRP_distance: {VRP_distance}')
            # calculate the cost of the non-optimized sequence
            tasks = np.array(range(jobs.shape[0]))
            tasks = tasks+1
            # if tasks.shape[0] %8 != 0, pad with -1
            if tasks.shape[0] %8 != 0:
                tasks = np.pad(tasks, (0, 8-tasks.shape[0]%8), 'constant', constant_values=-1)
            unoptimized_seuqnece = tasks.reshape(-1, 8)
            t = calculate_T(unoptimized_seuqnece)
            d = D_prime[1:, 1:]
            non_optimized_distance = np.trace(np.dot(t.T, d))
            # change non_optimized_distance to integer
            non_optimized_distance = int(non_optimized_distance)
            print(f'non_optimized_distance: {non_optimized_distance}')
            
            # calculate the cost of the row-wise optimized sequence
            index_matrix = np.zeros((source_dim,dest_dim))
            for j in range(jobs.shape[0]):
                index_matrix[jobs[j, 0], jobs[j, 1]] = j+1
            row_wise_optimized_sequence = row_wise_optimization(index_matrix)
            if row_wise_optimized_sequence.shape[0] %8 != 0:
                row_wise_optimized_sequence = np.pad(row_wise_optimized_sequence, (0, 8-row_wise_optimized_sequence.shape[0]%8), 'constant', constant_values=-1)
            row_wise_optimized_sequence = row_wise_optimized_sequence.reshape(-1, 8)
            t = calculate_T(row_wise_optimized_sequence)
            row_wise_optimized_distance = np.trace(np.dot(t.T, d))
            # change non_optimized_distance to integer
            row_wise_optimized_distance = int(row_wise_optimized_distance)
            print(f'row_wise_optimized_distance: {row_wise_optimized_distance}')

            # calculate the cost of the greedy optimized sequence
            greedy_optimized_sequence = greedy_scheduling(jobs, d)
            if greedy_optimized_sequence.shape[0] %8 != 0:
                greedy_optimized_sequence = np.pad(greedy_optimized_sequence, (0, 8-greedy_optimized_sequence.shape[0]%8), 'constant', constant_values=-1)
            greedy_optimized_sequence = greedy_optimized_sequence.reshape(-1, 8)
            t = calculate_T(greedy_optimized_sequence)
            greedy_optimized_distance = np.trace(np.dot(t.T, d))
            print(f'greedy_optimized_distance: {greedy_optimized_distance}')
            # append the results to the df
            stats.append([source_dim, dest_dim, num_candidates, r+1, non_optimized_distance, row_wise_optimized_distance, greedy_optimized_distance, VRP_distance])
        # convert the stats to the dataframe
        stats = pd.DataFrame(stats, columns=['source_labware', 'dest_labware', 'num_samples', 'repeat', 'unoptimized','rowwise', 'greedy','VRP'])
        # append the stats to the df
        df = pd.concat([df, stats], ignore_index=True)
        print(df)

df.to_csv('results_different_labwares_4.csv', index=False)
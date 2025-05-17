
import numpy as np
import matplotlib.pyplot as plt
from ortools_solver import CVRP_solver
from QAP_solver import calculate_S_E, calculate_D_prime, add_depot, calculate_D
from utils import distance_calculator, random_choose_candidate
import pygmtools as pygm

labware_list =[12,24,96,384]
# enumerate all the two combinations of the labware_list
labware_combinations = []
for i in range(len(labware_list)):
    for j in range(len(labware_list)):
        labware_combinations.append([labware_list[i], labware_list[j]])

np.random.seed(0)
for labware_combination in labware_combinations:
        stats = []
        print(f'labware_combination: {labware_combination}')

        for r in range (6):
            print('repeat:',r)
            stat = []
            for i in range(1,10):
                print(f'num_candidates={i}')
                experiments = random_choose_candidate(labware_combination[0],labware_combination[1],i)
                jobs = np.argwhere(experiments)
                D_S = calculate_D(experiments.shape[0])
                D_D = calculate_D(experiments.shape[1])
                S, E = calculate_S_E(experiments)   
                # calculate distance matrix
                D_prime = calculate_D_prime(D_S,D_D, S, E)
                D_prime = add_depot(D_prime)
                optimized_distance = 0
                VRP_distance, _ = CVRP_solver(D_prime.astype(np.int64), solving_time =20)
                non_optimized_distance = distance_calculator(jobs)
                stat.append((i,non_optimized_distance,optimized_distance, VRP_distance))
            stat = np.array(stat)
            np.savetxt(f'different_labwares/stats_{labware_combination[0]}_{labware_combination[1]}_repeat_{r}.csv', stat,fmt="%.2f", delimiter=",", header="num_candidates,non_optimized_distance,optimized_distance,VRP_distance")
            #stats.append(stat)
        #stats = np.array(stats)
        # save the stats
        #np.savetxt(f'stats_{labware_combination[0]}_{labware_combination[1]}.csv', stats)
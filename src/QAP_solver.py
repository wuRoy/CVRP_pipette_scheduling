import numpy as np
import pygmtools as pygm
from ortools_solver import CVRP_solver
from utils import calculate_num_rows

pygm.set_backend('numpy') # set default backend for pygmtools
np.random.seed(42) # fix random seed

def calculate_S_E(a, P=None):
    # a the jobs matrix, row is the source, column is the destination
    if P is not None:
        a = np.dot(a, P)
    a[a>0] = 1
    jobs = np.argwhere(a)
    starting_point = np.zeros((jobs.shape[0], a.shape[0]))
    ending_point = np.zeros((jobs.shape[0], a.shape[1]))
    ending_map = dict(np.argwhere(P if P is not None else np.eye(a.shape[1])).tolist())
    new_jobs = []
    for i in range(jobs.shape[0]):
        starting_point[i, jobs[i, 0]] = 1
        ending_point[i, ending_map[jobs[i, 1]]] = 1
        new_jobs.append([jobs[i, 0], ending_map[jobs[i, 1]]])
    if P is None:
        return starting_point, ending_point
    else:
        return starting_point, ending_point, np.array(new_jobs)

def calculate_D_prime(D_S, D_D, S, E, weight_S=1, weight_D=1):
    # calculate D' = S(D_S)S^T + E(D_D)E^T
    D_prime = np.dot(np.dot(S, D_S), S.T)*weight_S + np.dot(np.dot(E, D_D), E.T)*weight_D
    return D_prime

def calculate_D(labware:int):
    num_rows = calculate_num_rows(labware)
    D = np.ones((labware,labware))
    for i in range(labware):
        for j in range(labware):
            if i//num_rows == j//num_rows:
                if i-j == -1:
                    D[i, j] = 0

    if labware == 384:
        D = np.ones((labware,labware))
        for i in range(labware):
            for j in range(labware):
                if i//num_rows == j//num_rows:
                    if i-j == -2:
                        D[i, j] = 0
    elif labware == 1536:
        D = np.ones((labware,labware))
        for i in range(labware):
            for j in range(labware):
                if i//num_rows == j//num_rows:
                    if i-j == -4:
                        D[i, j] = 0
    return D

def add_depot(D_prime):
    # add a depot to D'
    D_prime = np.vstack((np.zeros(D_prime.shape[0]), D_prime))
    D_prime = np.hstack((np.zeros((D_prime.shape[0], 1)), D_prime))
    return D_prime

def get_optimized_sequence(recorder):
    # get the optimized sequences from the VCRP solver, pad with -1 and sort
    for i in range(len(recorder)):
        recorder[i] = np.array(recorder[i])
        recorder[i] = np.pad(recorder[i], (0, 8-recorder[i].shape[0]), 'constant', constant_values=-1)
    # move the elements containing -1 to the end
    optimized_seuqnece = np.array(recorder)
    optimized_seuqnece = np.array(sorted(optimized_seuqnece, key=lambda x: np.sum(x!=-1)))
    optimized_seuqneces = np.array(optimized_seuqnece[::-1])
    return optimized_seuqneces

def calculate_T(sequences):
    # the matrix should be paddled with -1, return a n*n matrix
    # sequences is a n*8 matrix

    sequences_flat = sequences.flatten()
    sequences_flat = sequences_flat[sequences_flat != -1]
    zeros = np.zeros((sequences_flat.shape[0],sequences_flat.shape[0]))
    for sequence in sequences:
        for i in range(sequence.shape[0]-1):
            if sequence[i] != -1 and sequence[i+1] != -1:
                zeros[sequence[i]-1,sequence[i+1]-1] = 1
            else:
                break
    return zeros

def CVRP_QAP(task_matrix, iteration=5, inner_cvrp_timewall=2, final_cvrp_timewall=10, ipfp_maxiter=50):
    task_matrix[task_matrix>0] = 1
    jobs = np.argwhere(task_matrix)
    D_S = calculate_D(task_matrix.shape[0])
    D_D = calculate_D(task_matrix.shape[1])   

    output_P = np.eye(task_matrix.shape[1])
    S, E = calculate_S_E(task_matrix)
    D_prime = calculate_D_prime(D_S,D_D, S, E)
    best_cost = float('inf')
    for i in range(iteration):
        # construct & update CVRP
        D_prime = add_depot(D_prime)

        # solve CVRP
        optimized_distance, recorder = CVRP_solver(D_prime.astype(np.int64), solving_time=inner_cvrp_timewall)

        optimized_seuqnecess = get_optimized_sequence(recorder)
        t = calculate_T(optimized_seuqnecess)
        D_prime = D_prime[1:, 1:]
        # cost = trace (T^T * D_prime)
        cost = np.trace(np.dot(t.T, D_prime))
        print(f'iter={i}, cost={cost} after CVRP')
        if best_cost > cost:
            best_cost = cost
            best_output_P = output_P

        # construct QAP
        A = np.dot(np.dot(E.T, t.T), E)
        B = D_D
        K = np.kron(1-B, A.T) # transform minimization into maximization

        # solve QAP
        P = pygm.ipfp((K + K.T), n1=task_matrix.shape[1], n2=task_matrix.shape[1], x0=np.eye(task_matrix.shape[1])[None,:,:], max_iter=ipfp_maxiter)

        # new_E = E * P
        new_E = np.dot(E, P)
        new_D_prime = calculate_D_prime(D_S,D_D, S, new_E)
        cost = np.trace(np.dot(t.T, new_D_prime))
        output_P = np.dot(output_P, P)
        print(f'iter={i}, cost={cost} after QAP')
        if best_cost > cost:
            best_cost = cost
            best_output_P = output_P

        # update params
        D_prime = new_D_prime
        E = new_E

    # calculate best CVRP result with more solving time
    S, E, new_jobs = calculate_S_E(task_matrix, best_output_P)
    D_prime = calculate_D_prime(D_S,D_D, S, E)
    D_prime = np.vstack((np.zeros(D_prime.shape[0]), D_prime))
    D_prime = np.hstack((np.zeros((D_prime.shape[0], 1)), D_prime))
    best_cost, best_recorder = CVRP_solver(D_prime.astype(np.int64), solving_time=final_cvrp_timewall)
    print(f'solution cost={best_cost}')

    # transform to job id sequence
    best_recorder = get_optimized_sequence(best_recorder)

    return best_cost, best_output_P, new_jobs, best_recorder

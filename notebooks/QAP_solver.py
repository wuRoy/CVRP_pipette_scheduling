import numpy as np
import pygmtools as pygm
from ortools_solver import CVRP_solver

pygm.set_backend('numpy') # set default backend for pygmtools
np.random.seed(42) # fix random seed

def calculate_S_E(jobs, num_dest, num_src):
    starting_point = np.zeros((jobs.shape[0], num_src))
    ending_point = np.zeros((jobs.shape[0], num_dest))
    for i in range(jobs.shape[0]):
        starting_point[i, jobs[i, 0]] = 1
        ending_point[i, jobs[i, 1]] = 1
    return starting_point, ending_point

def calculate_D_prime(D_s, D_e, S, E):
    # calculate D' = SDS^T + EDE^T
    D_prime = np.dot(np.dot(S, D_s), S.T) + np.dot(np.dot(E, D_e), E.T)
    return D_prime

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

def CVRP_QAP(jobs, num_dest=96, num_src=96, wells_per_row=8, iteration=5, eye_init_qap_solver=True):
    assert num_dest % wells_per_row == 0, f'number of destinations ({num_dest}) must be a multiplier of wells per row ({wells_per_row})'
    assert num_src % wells_per_row == 0, f'number of sources ({num_src}) must be a multiplier of wells per row ({wells_per_row})'

    def get_D(dim):
        D = np.ones((dim, dim))
        for i in range(dim):
            for j in range(dim):
                if i // wells_per_row == j // wells_per_row:
                    if i - j == -1:
                        D[i, j] = 0
        return D
    D_src = get_D(num_src)
    D_dest = get_D(num_dest)

    output_P = np.eye(num_dest)
    S, E = calculate_S_E(jobs, num_dest, num_src)
    D_prime = calculate_D_prime(D_src, D_dest, S, E)
    best_cost = float('inf')
    for i in range(iteration):
        # construct & update CVRP
        D_prime = np.vstack((np.zeros(D_prime.shape[0]), D_prime))
        D_prime = np.hstack((np.zeros((D_prime.shape[0], 1)), D_prime))
        
        # solve CVRP
        optimized_distance, recorder = CVRP_solver(D_prime.astype(np.int64), solving_time=2)
        
        optimized_seuqnecess = get_optimized_sequence(recorder)
        t = calculate_T(optimized_seuqnecess)
        D_prime = D_prime[1:,1:]
        # cost = trace (T^T * D_prime)
        cost = np.trace(np.dot(t.T, D_prime))
        print(f'iter={i}, cost={cost} after CVRP')
        if best_cost > cost:
            best_cost = cost
            best_output_P = output_P
        
        # construct QAP
        A = np.dot(np.dot(E.T, t.T), E)
        B = D_dest
        K = np.kron(1 - B, A.T) # transform minimization into maximization
        
        # solve QAP
        P = pygm.ipfp((K + K.T), n1=num_dest, n2=num_dest, 
                      x0=np.eye(num_dest)[None,:,:] if eye_init_qap_solver else None)

        # new_E = E * P
        new_E = np.dot(E, P)
        #new D_prime = S * D * S^T + new_E * D * new_E^T
        new_D_prime = np.dot(np.dot(S, D_src), S.T) + np.dot(np.dot(new_E, D_dest), new_E.T)
        cost = np.trace(np.dot(t.T, new_D_prime))
        output_P = np.dot(output_P, P)
        print(f'iter={i}, cost={cost} after QAP')
        if best_cost > cost:
            best_cost = cost
            best_output_P = output_P

        # update params
        D_prime = new_D_prime
        E = new_E
    return best_cost, best_output_P
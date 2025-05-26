import numpy as np

from ortools_solver import CVRP_solver
from utils import calculate_num_rows

np.random.seed(42) # fix random seed


def calculate_T(sequences):

    """Calculate matrix T which defines adjacency of two liquid transfers in the sequence.
    
    This function creates a transition matrix that represents which jobs follow each other
    in the optimized sequences. The depot (index 0) is connected to the first job in each
    sequence and the last job in each sequence is connected back to the depot.
    
    Args:
        sequences (np.ndarray): A n*8 matrix where each row represents a sequence of jobs.
                               Jobs are represented by integers, and -1 indicates padding.
    
    Returns:
        np.ndarray: A (num_jobs+1)*(num_jobs+1) binary matrix where T[i,j]=1 indicates
                   job i is immediately followed by job j in some sequence. Index 0 
                   represents the depot.
    """
    # Get all valid jobs (non-padding values)
    sequences_flat = sequences.flatten()
    sequences_flat = sequences_flat[sequences_flat != -1]
    
    # Initialize adjacency matrix (include depot at index 0)
    zeros = np.zeros((sequences_flat.shape[0]+1,sequences_flat.shape[0]+1))
    for sequence in sequences:
        # link the depot with the first job in a cycle
        zeros[0,sequence[0]] = 1
    
        # Link consecutive jobs within each sequence
        for i in range(sequence.shape[0]-1):
            if sequence[i] != -1 and sequence[i+1] != -1:
                zeros[sequence[i],sequence[i+1]] = 1
            else:
                # link the last job with the depot in a cycle
                zeros[sequence[i],0] = 1
                break
    return zeros

def calculate_S_E(a):

    """Calculate starting point matrix S, ending point matrix E, and volume vector.
    
    This function extracts the liquid transfer information from the task matrix and
    creates binary matrices indicating source wells (S) and destination wells (E)
    for each transfer job.
    
    Args:
        a (np.ndarray): Task matrix of size n_src * n_dst where non-zero elements
                       indicate volume transfers between wells.
    
    Returns:
        tuple: A tuple containing:
            - starting_point (np.ndarray): Binary matrix S of size num_jobs * n_src
                                         where S[i,j]=1 if job i starts from well j
            - ending_point (np.ndarray): Binary matrix E of size num_jobs * n_dst  
                                       where E[i,j]=1 if job i ends at well j
            - volumes (np.ndarray): Vector of volumes for each transfer job
    """

    jobs = np.argwhere(a>0)
    starting_point = np.zeros((jobs.shape[0], a.shape[0]))
    ending_point = np.zeros((jobs.shape[0], a.shape[1]))
    # Create mapping for destination wells
    ending_map = dict(np.argwhere(np.eye(a.shape[1])).tolist())
    
    # Set binary indicators for each job's source and destination
    for i in range(jobs.shape[0]):
        starting_point[i, jobs[i, 0]] = 1
        ending_point[i, ending_map[jobs[i, 1]]] = 1

    # Extract volumes for each transfer
    volumes = a[(jobs[:, 0], jobs[:, 1])]

    return starting_point, ending_point, volumes


def calculate_D_prime(D_s, D_d, S, E, volumes, time_s, speed_s, time_d, speed_d):
    """Calculate the distance matrix D_prime for the CVRP solver.
        D_prime = S(D_s)S^T + E(D_d)E^T
    Args:
        D_s (np.ndarray): Distance matrix for source plate wells
        D_d (np.ndarray): Distance matrix for destination plate wells  
        S (np.ndarray): Starting point binary matrix
        E (np.ndarray): Ending point binary matrix
        volumes (np.ndarray): Volume vector for each transfer
        time_s (float): Fixed time cost for aspiration operations (seconds)
        speed_s (float): Aspiration speed (uL/s)
        time_d (float): Fixed time cost for dispensing operations  (seconds)
        speed_d (float): Dispensing speed (uL/s)
    
    Returns:
        np.ndarray: Modified distance matrix D' including depot (index 0) with
                   time costs for transitions between all pairs of jobs.
    """

    # Calculate aspiration cost matrix
    binary_s = np.dot(np.dot(S, D_s), S.T)
    D_s_prime = binary_s * (time_s + volumes[None, :] / speed_s) \
        + (1 - binary_s) * np.maximum(0, (volumes[None, :] - volumes[:, None]) / speed_s)
    # add depot
    D_s_prime = np.vstack((time_s + volumes / speed_s, D_s_prime))
    D_s_prime = np.hstack((np.zeros((D_s_prime.shape[0], 1)), D_s_prime))
    # Calculate dispensing cost matrix
    binary_d = np.dot(np.dot(E, D_d), E.T)
    D_d_prime = binary_d * (time_d + volumes[None, :] / speed_d) \
        + (1 - binary_d) * np.maximum(0, (volumes[None, :] - volumes[:, None]) / speed_d)
    # add depot
    D_d_prime = np.vstack((time_d + volumes / speed_d, D_d_prime))
    D_d_prime = np.hstack((np.zeros((D_d_prime.shape[0], 1)), D_d_prime))

    D_prime = D_s_prime + D_d_prime

    return D_prime


def calculate_D(labware:int):
    """Calculate pairwise distance matrix for a given labware type.
    
    This function generates a pairwise distance matrix representing the spatial relationships
    between wells in standard laboratory plates. Adjacent wells in the same row
    have distance 0, while all other pairs have distance 1.
    
    Args:
        labware (int): Number of wells in the labware (12, 24, 96, 384, or 1536)
    
    Returns:
        np.ndarray: Pairwise distance matrix of size labware * labware where D[i,j] represents
                   the distance between well i and well j.
    
    Raises:
        ValueError: If labware type is not supported.
    """


    if labware not in [12, 24, 96, 384, 1536]:
        raise ValueError(f"Labware {labware} is not supported.")

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



def CVRP_pipette_scheduling(
    task_matrix, 
    aspirate_t=1, 
    aspirate_speed=10,
    dispense_t=1,
    dispense_speed=10,
    cvrp_timewall=10,
    decimal_points=2):
    r"""
    This is the main function that orchestrates the entire pipette scheduling optimization
    process. 

    Args:
        task_matrix: matrix of size $n^{src} \times n^{dst}$, where a non-zero element at (a, b) means the volume 
            (in uL) to move from well a in source plate to well b in destination plate. The dimensions of 
            task_matrix are used to infer the labware used (supported labwares: 12, 24, 96, 384, 1536)
        aspirate_t: time needed (in seconds) to move tips down, move tips up and move arm in one aspiration 
            operation. It is $t_1 + t_3 + t_4$ as described in the paper.
        aspirate_speed: the speed (in uL/s) for aspiration. ``volume / aspirate_speed`` is the $t_2$ as described 
            in the paper.
        dispense_t: time needed (in seconds) to move tips down, move tips up and move arm in one dispensing 
            operation. It is $t_1 + t_3 + t_4$ as described in the paper.
        dispense_speed: the speed (in uL/s) for dispensing. ``volume / aspirate_speed`` is the $t_2$ as described 
            in the paper.
        cvrp_timewall: max solving time (in seconds) for the CVRP solver. Allowing a longer solving time will
            increase the chance of having a better solution. Set this to the maximal time you can accept, or tune
            this parameter to the minimal solving time that gives you solutions with acceptalbe quality.
        decimal_points: when the time costs are not integers, the number of decimal points you want to keep for
            numerical precision.
    Returns:
        optimized_distance, optimized_seuqnecess
    """
    nwells_source, nwells_destination = task_matrix.shape


    D_S = calculate_D(nwells_source)
    D_D = calculate_D(nwells_destination)

    S, E, volumes = calculate_S_E(task_matrix)

    D_prime = calculate_D_prime(D_S, D_D, S, E, volumes, aspirate_t, aspirate_speed, dispense_t, dispense_speed)

    # solve CVRP
    optimized_distance, recorder = CVRP_solver(np.round(D_prime * 10 ** decimal_points).astype(np.int64), solving_time=cvrp_timewall)
    optimized_distance /= 10 ** decimal_points

    optimized_seuqnecess = get_optimized_sequence(recorder)

    return optimized_distance, optimized_seuqnecess


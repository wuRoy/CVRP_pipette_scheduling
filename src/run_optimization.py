import numpy as np
from QAP_solver import CVRP_QAP 
from ortools_solver import CVRP_solver
from utils import calculate_distance_matrix, get_optimized_sequence, print_command

def pipette_scheduling_optimization(task_matrix:np.ndarray,file_path:str='command_line.csv',total_volume:int=100):
    # task_matrix is a 96*96 matrix
    # row is the source, column is the destination
    
    binary_task_matrix = task_matrix.copy()
    binary_task_matrix[binary_task_matrix>0] = 1
    jobs = np.argwhere(binary_task_matrix)
    # compute the permutation matrix and the cost
    cost, P = CVRP_QAP(jobs,iteration=10)
    # permute the task_matrix
    update_task = np.dot(task_matrix,P)
    updated_jobs = np.argwhere(update_task)
    fraction = [update_task[i,j] for i,j in updated_jobs]
    distance_matrix = calculate_distance_matrix(jobs)
    VRP_distance, VRP_recorder = CVRP_solver(distance_matrix)
    # turn the recorder into a sequence
    recorder = get_optimized_sequence(VRP_recorder)
    sequence = recorder.flatten()
    sequence = sequence[sequence!=-1] -1
    # print the command line
    command_line = print_command(sequence,updated_jobs,fraction, total_volume)
    np.savetxt(file_path,command_line.round(2),fmt='%s',delimiter=',')
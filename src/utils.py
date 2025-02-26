import numpy as np
import matplotlib.pyplot as plt

def show_matrix(matrix):
    plt.figure(figsize=(4,4))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.show()

def random_choice(total_elements, chosen_elements):
    '''
    total_elements: total number of elements in the array
    chosen_elements: number of elements to be chosen
    '''
    a = np.zeros(total_elements)
    random_vector = np.random.rand(chosen_elements)
    random_vector = random_vector.round(2)
    random_vector = random_vector / random_vector.sum(axis=0, keepdims=1)
    a[:chosen_elements] = random_vector
    np.random.shuffle(a)
    return a

def random_choose_candidate(num_candidate,total_dim,non_zeros_dim): 
    '''
    num_candidate: number of candidate to be chosen
    total_candidate: total number of candidates
    chosen_elements: number of elements to be chosen
    '''
    # repeat the random_choice function for num_candidate times
    lower_bound = np.zeros(total_dim)
    upper_bound = np.ones(total_dim)
    a = np.zeros((num_candidate,total_dim))
    for i in range(num_candidate):
        while True:
            candidate = random_choice(total_dim,non_zeros_dim)
            if np.all((candidate >= lower_bound) & (candidate <= upper_bound)):
                a[i,:] = candidate
                break
    return a.round(2)

def calculate_distance_matrix(pos_matrix):
    # given a non-zero position matrix, calculate the distance between each pair
    distance_matrix = np.zeros((pos_matrix.shape[0], pos_matrix.shape[0]))
    for i in range(pos_matrix.shape[0]):
        for j in range(pos_matrix.shape[0]):
            is_neighbor_dest = ((pos_matrix[i, 0]//8) == (pos_matrix[j, 0]//8)) and ((abs(pos_matrix[i, 0] - pos_matrix[j, 0]) <=1) and (abs(pos_matrix[i, 0] - pos_matrix[j, 0])>0))
            is_neighbor_source = ((pos_matrix[i, 1]//8) == (pos_matrix[j, 1]//8)) and ((abs(pos_matrix[i, 1] - pos_matrix[j, 1]) <=1) and (abs(pos_matrix[i, 1] - pos_matrix[j, 1])>0))
            
            if is_neighbor_dest and is_neighbor_source:
                if (pos_matrix[i, 0]-pos_matrix[j, 0]<0) and (pos_matrix[i, 1]-pos_matrix[j, 1]<0):
                    distance_matrix[i, j] = 0
                elif (pos_matrix[i, 0]-pos_matrix[j, 0]<0) and (pos_matrix[i, 1]-pos_matrix[j, 1]>0):
                    distance_matrix[i, j] = 1
                elif (pos_matrix[i, 0]-pos_matrix[j, 0]>0) and (pos_matrix[i, 1]-pos_matrix[j, 1]<0):
                    distance_matrix[i, j] = 1
                else:
                    distance_matrix[i, j] = 2
                    
            elif is_neighbor_dest or is_neighbor_source:
                if is_neighbor_dest:
                    if pos_matrix[i, 0]-pos_matrix[j, 0]<0:
                        distance_matrix[i, j] = 1
                    else:
                        distance_matrix[i, j] = 2
                        
                if is_neighbor_source:
                    if pos_matrix[i, 1]-pos_matrix[j, 1]<0:
                        distance_matrix[i, j] = 1
                    else:
                        distance_matrix[i, j] = 2
            else:
                distance_matrix[i, j] = 2
    distance_matrix = np.vstack((np.zeros(distance_matrix.shape[0]), distance_matrix))
    distance_matrix = np.hstack((np.zeros((distance_matrix.shape[0], 1)), distance_matrix))
    return distance_matrix

def pair_distance_calculator(pos1,pos2):
    # given two positions, calculate the distance between them
    is_neighbor_dest = ((pos1[0]//8) == (pos2[0]//8)) and ((abs(pos1[0] - pos2[0]) <=1) and (abs(pos1[0] - pos2[0])>0))
    is_neighbor_source = ((pos1[1]//8) == (pos2[1]//8)) and ((abs(pos1[1] - pos2[1]) <=1) and (abs(pos1[1] - pos2[1])>0))
    if is_neighbor_dest and is_neighbor_source:
        if (pos1[0]-pos2[0]<0) and (pos1[1]-pos2[1]<0):
            return 0
        elif (pos1[0]-pos2[0]<0) and (pos1[1]-pos2[1]>0):
            return 1
        elif (pos1[0]-pos2[0]>0) and (pos1[1]-pos2[1]<0):
            return 1
        else:
            return 2
    elif is_neighbor_dest or is_neighbor_source:
        if is_neighbor_dest:
            if pos1[0]-pos2[0]<0:
                return 1
            else:
                return 2
        if is_neighbor_source:
            if pos1[1]-pos2[1]<0:
                return 1
            else:
                return 2
    else:
        return 2

def distance_calculator(jobs):
    # calculate the total distance given a list of instructions
    distance_sum = 0
    for i in range(jobs.shape[0]//8):
        # take the first 8 elements
        temp = jobs[i*8:(i+1)*8]
        temp_sum = 0
        for j in range(7):
            #print(temp[j],temp[j+1])
            temp_sum += pair_distance_calculator(temp[j],temp[j+1])
        #print(temp_sum)
        distance_sum += temp_sum
    
    return distance_sum

def get_optimized_sequence(recorder):
    # get the optimized sequences from the CVRP solver, pad with -1 and sort
    for i in range(len(recorder)):
        recorder[i] = np.array(recorder[i])
        recorder[i] = np.pad(recorder[i], (0, 8-recorder[i].shape[0]), 'constant', constant_values=-1)
    # move the elements containing -1 to the end
    optimized_seuqnece = np.array(recorder)
    optimized_seuqnece = np.array(sorted(optimized_seuqnece, key=lambda x: np.sum(x!=-1)))
    optimized_seuqneces = np.array(optimized_seuqnece[::-1])
    return optimized_seuqneces

def print_command(flatten_sequence, jobs, total_volume=20):
    '''
        flatten_sequence: the optimized sequence
        jobs: the job pair
        total_volume: the total volume of the system
    
    '''
    command_line = []
    for i in range(flatten_sequence.shape[0]):
    # add the command line base on the index, set the volume as 20ul by default
        command_line.append(
            [
                'source',
                jobs[flatten_sequence[i],0]+1,
                'dest',
                jobs[flatten_sequence[i],1]+1,
                20
            ]
        )
    command_line = np.array(command_line) 
    return command_line
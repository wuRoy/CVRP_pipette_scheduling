import numpy as np
import matplotlib.pyplot as plt

def show_matrix(matrix):
    plt.figure(figsize=(4,4))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.show()

def random_choose_candidate_2(source_dim,dest_dim,non_zeros_num,if_random_volume=True):
    '''
    num_candidate: number of candidate to be chosen
    total_candidate: total number of candidates
    chosen_elements: number of elements to be chosen
    '''
    non_zeros =0
    # repeat the random_choice function for num_candidate times
    a = np.zeros((source_dim,dest_dim))
    # randomly choose non_zeros_dim elements in the matrix as 1
    while non_zeros < non_zeros_num:
        i = np.random.randint(0, source_dim)
        j = np.random.randint(0, dest_dim)
        if a[i,j] == 0:
            if if_random_volume:
                # randomly choose the volume between 1 and 100
                a[i,j] = np.random.randint(1, 100)
            else:
                a[i,j] = 1
            non_zeros += 1
    return a

def random_choose_candidate_3(source_dim,dest_dim,non_zeros_num,if_random_volume=True):
    non_zeros =0
    # repeat the random_choice function for num_candidate times
    a = np.zeros((source_dim,dest_dim))
    # randomly choose non_zeros_dim elements in the matrix as 1 
    for i in range(source_dim):
        for j in range(dest_dim):
            if np.random.rand() >(1-non_zeros_num/(source_dim*dest_dim)):
                if if_random_volume:
                    # randomly choose the volume between 1 and 100
                    a[i,j] = np.random.randint(1, 100)
                else:
                    a[i,j] = 1
                non_zeros += 1
            if non_zeros >= non_zeros_num:
                break
    return a

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

def random_choose_candidate(source_dim,dest_dim,non_zeros_dim): 
    '''
    num_candidate: number of candidate to be chosen
    total_candidate: total number of candidates
    chosen_elements: number of elements to be chosen
    '''
    # repeat the random_choice function for num_candidate times
    a = np.zeros((source_dim,dest_dim))
    for i in range(source_dim):
        candidate = random_choice(dest_dim,non_zeros_dim)
        a[i,:] = candidate
    return a.round(2)



def random_choice_backup(total_elements, chosen_elements):
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
# to be deleted
def random_choose_candidate_backup(source_dim,dest_dim,non_zeros_dim): 
    '''
    num_candidate: number of candidate to be chosen
    total_candidate: total number of candidates
    chosen_elements: number of elements to be chosen
    '''
    # repeat the random_choice function for num_candidate times
    lower_bound = np.zeros(dest_dim)
    upper_bound = np.ones(dest_dim)
    a = np.zeros((source_dim,dest_dim))
    for i in range(source_dim):
        while True:
            candidate = random_choice(total_dim,non_zeros_dim)
            if np.all((candidate >= lower_bound) & (candidate <= upper_bound)):
                a[i,:] = candidate
                break
    return a.round(2)

def calculate_num_rows(labware):
    '''
    define the neighborhood of the source and destination labware
    labware: the labware, could be 12, 24, 96, 384
    return the criteria of the neighborhood
    '''
    if labware not in [12, 24, 96, 384, 1536]:
        raise ValueError("labware should be one of [12, 24, 96, 384, 1536]")
    if labware == 12:
        return 3
    elif labware == 24:
        return 4
    elif labware == 96:
        return 8
    elif labware == 384:
        return 16
    elif labware == 1536:
        return 64 
    

# to be deleted
def calculate_distance_matrix(pos_matrix, source_labware, dest_labware):
    '''
    Given a job list, calculate the distance matrix
    pos_matrix: the job list, a n*2 matrix
    source_labware: the source labware, could be 12, 24, 96, 384
    dest_labware: the destination labware, could be 12, 24, 96, 384
    return the distance matrix
    '''
    # given a non-zero position matrix, calculate the distance between each pair
    num_rows_source = calculate_num_rows(source_labware)
    num_rows_dest = calculate_num_rows(dest_labware)
    distance_matrix = np.zeros((pos_matrix.shape[0], pos_matrix.shape[0]))
    for i in range(pos_matrix.shape[0]):
        for j in range(pos_matrix.shape[0]):
            is_neighbor_source = ((pos_matrix[i, 0]//num_rows_source) == (pos_matrix[j, 0]//num_rows_source)) and ((abs(pos_matrix[i, 0] - pos_matrix[j, 0]) <=1) and (abs(pos_matrix[i, 0] - pos_matrix[j, 0])>0))
            is_neighbor_dest = ((pos_matrix[i, 1]//num_rows_dest) == (pos_matrix[j, 1]//num_rows_dest)) and ((abs(pos_matrix[i, 1] - pos_matrix[j, 1]) <=1) and (abs(pos_matrix[i, 1] - pos_matrix[j, 1])>0))
            
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
                    if pos_matrix[i, 1]-pos_matrix[j, 1]<0:
                        distance_matrix[i, j] = 1
                    else:
                        distance_matrix[i, j] = 2
                        
                if is_neighbor_source:
                    if pos_matrix[i, 0]-pos_matrix[j, 0]<0:
                        distance_matrix[i, j] = 1
                    else:
                        distance_matrix[i, j] = 2
            else:
                distance_matrix[i, j] = 2
    distance_matrix = np.vstack((np.zeros(distance_matrix.shape[0]), distance_matrix))
    distance_matrix = np.hstack((np.zeros((distance_matrix.shape[0], 1)), distance_matrix))
    return distance_matrix
# to be deleted
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
# to be deleted
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

def print_command(flatten_sequence, jobs, source_name='source',dest_name='dest', volume=None):
    '''
        flatten_sequence: the optimized sequence, start from 0
        jobs: the job pair
        total_volume: the total volume of the system
    
    '''
    command_line = []
    for i in range(flatten_sequence.shape[0]):
    # add the command line base on the index, set the volume as 20ul by default
        if volume is not None:
            command_line.append(
                [
                    source_name,
                    jobs[flatten_sequence[i],0]+1,
                    dest_name,
                    jobs[flatten_sequence[i],1]+1,
                    volume[flatten_sequence[i]]
                ]
            )
        else:
            command_line.append(
                [
                    source_name,
                    jobs[flatten_sequence[i],0]+1,
                    dest_name,
                    jobs[flatten_sequence[i],1]+1,
                    20
                ]
            )
    command_line = np.array(command_line) 
    return command_line
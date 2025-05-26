import numpy as np
import matplotlib.pyplot as plt

def show_matrix(matrix):
    """Display a matrix as a heatmap visualization.
    
    Args:
        matrix (np.ndarray): 2D numpy array to be visualized as a heatmap.
                           Values are color-coded with higher values appearing 'hotter'.
    
    Returns:
        None: Displays the plot directly using matplotlib.

    """
    plt.figure(figsize=(4,4))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.show()

def random_task_generation(source_dim,dest_dim,non_zeros_num,if_random_volume=True):
    """Generate a random task matrix for liquid transfer simulation.
    
    This function creates a matrix representing liquid transfer tasks between
    source and destination wells. The matrix contains a specified number of non-zero
    elements placed randomly throughout the matrix.
    
    Args:
        source_dim (int): Number of wells in the source plate (rows in matrix).
        dest_dim (int): Number of wells in the destination plate (columns in matrix).
        non_zeros_num (int): Number of non-zero transfer tasks to generate.
                            Must be ≤ source_dim * dest_dim.
        if_random_volume (bool, optional): If True, assigns random volumes (1-100 μL)
                                         to each transfer. If False, assigns volume 1
                                         to all transfers.
    
    Returns:
        np.ndarray: A source_dim * dest_dim matrix where non-zero elements represent
                   liquid transfer volumes from source well i to destination well j.

    """
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

def calculate_num_rows(labware):
    """Calculate the number of rows for standard laboratory labware configurations.
    
    This function returns the number of rows for common laboratory
    plate formats, which is used for determining pairwise distance matrix.
    
    Args:
        labware (int): Number of wells in the labware. Must be one of:
                      - 12: 12-well plate (3 rows * 4 columns)
                      - 24: 24-well plate (4 rows * 6 columns)  
                      - 96: 96-well plate (8 rows * 12 columns)
                      - 384: 384-well plate (16 rows * 24 columns)
                      - 1536: 1536-well plate (32 rows * 48 columns)
    
    Returns:
        int: Number of rows in the specified labware format.
    

    """
    if labware not in [12, 24, 96, 384, 1536]:
        raise ValueError("labware should be one of [12, 24, 96, 384, 1536]")
    
    # Standard labware row configurations
    labware_rows = {
        12: 3,    # 3 rows × 4 columns
        24: 4,    # 4 rows × 6 columns
        96: 8,    # 8 rows × 12 columns
        384: 16,  # 16 rows × 24 columns (high-density)
        1536: 32  # 32 rows × 48 columns (ultra-high-density)
    }
    return labware_rows[labware]

def get_optimized_sequence(recorder):

    """Process and format the optimized sequences from the CVRP solver.
    
    This function takes the raw output from the CVRP solver and formats it into
    a standardized matrix format with padding and sorting.
    
    Args:
        recorder (list): List of sequences from CVRP solver, where each sequence
                        is a list of job indices.
    
    Returns:
        np.ndarray: Formatted sequence matrix where each row is a sequence padded
                   to length 8 with -1, sorted by sequence length (longest first).
    """
    # Convert to numpy arrays and pad each sequence to length 8
    for i in range(len(recorder)):
        recorder[i] = np.array(recorder[i])
        recorder[i] = np.pad(recorder[i], (0, 8-recorder[i].shape[0]), 'constant', constant_values=-1)
    # move the elements containing -1 to the end
    optimized_seuqnece = np.array(recorder)
    optimized_seuqnece = np.array(sorted(optimized_seuqnece, key=lambda x: np.sum(x!=-1)))
    optimized_seuqneces = np.array(optimized_seuqnece[::-1])
    return optimized_seuqneces

    

def print_command(flatten_sequence, jobs, source_name='source',dest_name='dest', volume=None):
    """Generate pipetting commands from optimized job sequences for Tecan Evo 200 and Janus G3.
    
    This function converts the numerical job sequence output into a formatted command
    list that can be used on the automated liquid handling system. 
    Each command specifies source well, destination well, and volume.
    
    Args:
        flatten_sequence (np.ndarray): 1D array of job indices representing the
                                      optimized order of liquid transfer operations.
        jobs (np.ndarray): 2D array where each row [i, j] represents a transfer
                          from source well i to destination well j.
        source_name (str, optional): Name identifier for the source plate.
                                   Defaults to 'source'.
        dest_name (str, optional): Name identifier for the destination plate.
                                 Defaults to 'dest'.
        volume (np.ndarray, optional): Array of volumes (μL) for each job.
                                     If None, defaults to 20 μL for all transfers.
    
    Returns:
        np.ndarray: Command matrix where each row contains:
                   [source_name, source_well_number, dest_name, dest_well_number, volume]
                   Well numbers are 1-indexed for laboratory convention.
    
    Note:
        - Well numbering is converted from 0-indexed (programming) to 1-indexed (lab standard)
        - Default volume of 20 μL is used when volume array is not provided
        - Commands are ordered according to the optimized sequence for efficiency

    """
    command_line = []
    for i in range(flatten_sequence.shape[0]):
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
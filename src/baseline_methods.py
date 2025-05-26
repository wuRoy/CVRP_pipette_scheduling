import numpy as np


def LAP(index_matrix):
    """
    Long-axis Prioritize (LAP) method for pipette scheduling optimization.
    It prioritizes adjacent wells within the same pipette tip spacing to 
    minimize physical movement and improve efficiency.
    
    The algorithm adapts to different labware formats by using appropriate tip spacing:
    - Standard plates (12, 24, 96-well): 1-well spacing
    - High-density plates (384-well): 2-well spacing (every other row)
    - Ultra-high-density plates (1536-well): 4-well spacing (every fourth row)
    
    Args:
        index_matrix (np.ndarray): 2D array representing the task matrix where:
                                  - Rows correspond to source wells
                                  - Columns correspond to destination wells
                                  - Non-zero entries contain job IDs for transfers
                                  - Zero entries indicate no transfer required
    
    Returns:
        np.ndarray: 1D array of job IDs ordered according to the LAP algorithm.
                   This sequence minimizes pipette movement by maintaining spatial
                   adjacency when possible.
    
    Algorithm Details:
        1. Transpose matrix if more columns than rows (optimize on longer dimension)
        2. Process rows in round-robin fashion
        3. Within each row, prefer wells adjacent to the last selected column
        4. Use appropriate spacing based on plate density:
           - <384 wells: Standard 1-well spacing
           - 384 wells: 2-well spacing for every other row
           - 1536 wells: 4-well spacing for every fourth row
        5. Break ties randomly to avoid systematic bias
    
    Spatial Optimization Rules:
        - 12-well plates: Prefer wells in same 3-well row blocks
        - 24-well plates: Prefer wells in same 4-well row blocks  
        - 96-well plates: Prefer wells in same 8-well row blocks
        - 384-well plates: Prefer wells in same 16-well row blocks with 2-well spacing
        - 1536-well plates: Prefer wells in same 64-well row blocks with 4-well spacing

    """
    
    mat = index_matrix.copy()
    seq = []
    last_col = 0    # Track last selected column for adjacency preference
    count = 0
    # Transpose matrix if more columns than rows (optimize on longer dimension)
    if mat.shape[0] < mat.shape[1]:
        mat = mat.T
    n_rows = mat.shape[0]


    if n_rows < 384:
        # Standard plates: 12, 24, 96-well with 1-well spacing

        while mat.sum() != 0:
            row_num = count % n_rows
            row = mat[row_num]
            non_zeros = np.argwhere(row).flatten()

            if non_zeros.size > 0:
                # Default: pick one at random
                col = np.random.choice(non_zeros)
                # Prefer adjacent wells within same block for spatial efficiency
                for j in non_zeros:
                    # 96-well: prefer adjacent wells in same 8-well block
                    if mat.shape[1] == 96:
                        if (j - last_col) == 1 and (j // 8) == (last_col // 8):
                            col = j
                            break
                    # 12-well: prefer adjacent wells in same 3-well block
                    elif mat.shape[1] == 12:
                        if (j - last_col) == 1 and (j // 3) == (last_col // 3):
                            col = j
                            break
                    # 24-well: prefer adjacent wells in same 4-well block
                    elif mat.shape[1] == 24:
                        if (j - last_col) == 1 and (j // 4) == (last_col // 4):
                            col = j
                            break

                # record and remove it
                seq.append(mat[row_num, col])
                mat[row_num, col] = 0
                last_col = col
                count += 1
            else:
                # Skip empty rows
                count += 1
                continue
    elif n_rows == 384:
        # 384-well plates: use 2-well spacing (every other row)
        while mat.sum() != 0:
            # Prioritize even rows first, then odd rows
            if mat[::2].sum() > 0:
                row_num = count % n_rows
            else:
                row_num = (count + 1) % n_rows
            row = mat[row_num]
            non_zeros = np.argwhere(row).flatten()

            if non_zeros.size > 0:
                # Default: pick one at random
                col = np.random.choice(non_zeros)
                # Prefer adjacent wells with appropriate spacing
                for j in non_zeros:
                    # 384-well: 2-well spacing in same 16-well block
                    if mat.shape[1] == 384:
                        if (j - last_col) == 2 and (j // 16) == (last_col // 16):
                            col = j
                            break
                    else:
                        # Handle mixed plate scenarios
                        if mat.shape[1] == 96:
                            if (j - last_col) == 1 and (j // 8) == (last_col // 8):
                                col = j
                                break
                        elif mat.shape[1] == 12:
                            if (j - last_col) == 1 and (j // 3) == (last_col // 3):
                                col = j
                                break
                        elif mat.shape[1] == 24:
                            if (j - last_col) == 1 and (j // 4) == (last_col // 4):
                                col = j
                                break

                # record and remove it
                seq.append(mat[row_num, col])
                mat[row_num, col] = 0
                last_col = col
                count += 2  # Increment by 2 for 384-well spacing
            else:
                count += 2
                continue

    elif n_rows == 1536:
        # 1536-well plates: use 4-well spacing (every fourth row)
        while mat.sum() != 0:
            # Process rows in groups of 4 with priority order
            if mat[::4].sum() > 0:
                row_num = count % n_rows
            elif mat[1::4].sum() > 0:
                row_num = (count + 1) % n_rows
            elif mat[2::4].sum() > 0:
                row_num = (count + 2) % n_rows
            elif mat[3::4].sum() > 0:
                row_num = (count + 3) % n_rows

            row = mat[row_num]
            non_zeros = np.argwhere(row).flatten()

            if non_zeros.size > 0:
                # Default: pick one at random
                col = np.random.choice(non_zeros)

                # Prefer adjacent wells with 4-well spacing
                for j in non_zeros:
                    # 1536-well: 4-well spacing in same 64-well block
                    if mat.shape[1] == 1536:
                        if (j - last_col) == 2 and (j // 64) == (last_col // 64):
                            col = j
                            break

                # record and remove it
                seq.append(mat[row_num, col])
                mat[row_num, col] = 0
                last_col = col
                count += 4  # Increment by 4 for 1536-well spacing
            else:
                count += 4
                continue

    return np.array(seq, dtype=int)

def greedy(jobs, d):
    """
    Greedy nearest-neighbor scheduling method for liquid transfer optimization.
    
    This function implements a greedy approach that builds a job sequence by always
    moving to the closest unvisited job according to the provided distance matrix.
    It includes randomization to break ties and prevent getting stuck in local optima.
    
    Args:
        jobs (array-like): Array or list of jobs. Only the length is used to determine
                          the number of jobs to schedule. Shape should be (n_jobs, ...).
        d (np.ndarray): Square distance matrix of shape (n_jobs, n_jobs) where
                       d[i, j] represents the cost/time to move from job i to job j.
                       Depots should be removed.
    
    Returns:
        np.ndarray: 1D array of job indices representing the order in which jobs
                   should be executed to minimize total distance/time.
    
    """
    jobs_idx = np.array(range(jobs.shape[0]))
    # randomly choose an idx as the start point
    start_idx = np.random.choice(jobs_idx)
    # remove the start_idx from the jobs_idx
    jobs_idx = np.delete(jobs_idx, np.where(jobs_idx == start_idx))
    greedy_sequence = [start_idx]
    count = 0
    while jobs_idx.shape[0] != 0:
        count += 1
        # find the minimum distance in the distance matrix of row start_idx
        min_idx = np.where(d[start_idx] == d[start_idx].min())[0]
        # randomly choose one of the minimum distance
        # check if the min_idx is in the jobs_idx
        min_idx = np.intersect1d(min_idx, jobs_idx)
        if min_idx.shape[0] == 0:
            rand_min_idx = np.random.choice(jobs_idx)
        else:
            rand_min_idx = np.random.choice(min_idx)
        start_idx = rand_min_idx
        if count%8 == 7:
            start_idx = np.random.choice(jobs_idx)
        # remove the rand_min_idx from the jobs_idx
        jobs_idx = np.delete(jobs_idx, np.where(jobs_idx == start_idx))

        greedy_sequence.append(start_idx)
    greedy_sequence = np.array(greedy_sequence)
    return greedy_sequence
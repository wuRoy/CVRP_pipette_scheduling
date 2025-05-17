import numpy as np


def row_wise_optimization(index_matrix):
    """
    Perform row‐wise, round‐robin picking of non‐zero entries in `index_matrix`
    with a slight preference for items adjacent (in the same 8‐wide block)
    to the last picked column.

    Args:
        index_matrix (np.ndarray): 2D array where non‐zero entries are job IDs.

    Returns:
        np.ndarray: 1D array of job IDs in the pick order.
    """

    
    # Work on a copy so we don't destroy the caller's matrix
    mat = index_matrix.copy()
    seq = []
    last_col = 0
    count = 0
    if mat.shape[0] < mat.shape[1]:
        # transpose the matrix if it is not square, optimize on the longer side
        mat = mat.T
    n_rows = mat.shape[0]

    # Continue until we've zeroed out every job
    if n_rows < 384:
        while mat.sum() != 0:
            row_num = count % n_rows
            row = mat[row_num]
            non_zeros = np.argwhere(row).flatten()

            if non_zeros.size > 0:
                # pick one at random
                col = np.random.choice(non_zeros)
                # but if any non-zero is immediately to the right of last_col
                # within the same block of width 8, prefer that one
                for j in non_zeros:
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
                count += 1
            else:
                count += 1
                continue
    elif mat.shape[1] == 384:
        # if the matrix is larger than 384, use a different strategy
        while mat.sum() != 0:
            # if the sum of even rows is larger than 0
            if mat[::2].sum() > 0:
                row_num = count % n_rows
            else:
                row_num = (count + 1) % n_rows
            row = mat[row_num]
            non_zeros = np.argwhere(row).flatten()

            if non_zeros.size > 0:
                # pick one at random
                col = np.random.choice(non_zeros)
                # but if any non-zero is immediately to the right of last_col
                # within the same block of width 8, prefer that one
                for j in non_zeros:
                    if mat.shape[1] == 384:
                        if (j - last_col) == 2 and (j // 16) == (last_col // 16):
                            col = j
                            break
                    else:
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
                count += 2
            else:
                count += 2
                continue

    elif mat.shape[1] == 1536:
        # if the matrix is larger than 384, use a different strategy
        while mat.sum() != 0:
            # if the sum of even rows is larger than 0
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
                # pick one at random
                col = np.random.choice(non_zeros)
                # but if any non-zero is immediately to the right of last_col
                # within the same block of width 8, prefer that one
                for j in non_zeros:
                    if mat.shape[1] == 1536:
                        if (j - last_col) == 2 and (j // 64) == (last_col // 64):
                            col = j
                            break

                # record and remove it
                seq.append(mat[row_num, col])
                mat[row_num, col] = 0
                last_col = col
                count += 4
            else:
                count += 4
                continue

    return np.array(seq, dtype=int)

def greedy_scheduling(jobs, d):
    """
    Build a greedy sequence of job indices by always moving
    to the (closest) unvisited job according to the distance matrix.

    Args:
        jobs (array-like): list/array of jobs (only used for its length).
        d (2D array-like): square distance matrix of shape (n_jobs, n_jobs),
                           where d[i, j] is the “distance” from job i to job j.
                            no depot!!

    Returns:
        np.ndarray: 1D array of job indices in the order they were visited.
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
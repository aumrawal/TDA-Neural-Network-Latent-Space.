import numpy as np
from numpy import argmax
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from tensorflow import python
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense

from Data import load_data
from Building_DNN import build_model

(x_train, y_train), (x_test, y_test) = load_data()

model = build_model()
#model.fit(x_train, y_train, epochs=10, batch_size=600)

point_cloud = x_train[:50]

diff = point_cloud[:,np.newaxis,:] - point_cloud[np.newaxis,:,:]
dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))


# 1. Get all indices (i, j) where i < j
rows, cols = np.triu_indices(len(dist_matrix), k=1)

# 2. Extract the distances for just those pairs
upper_distances = dist_matrix[rows, cols]

# 3. Find which of those distances are within our epsilon

# edge_mask = upper_distances <= epsilon !!!

# 4. Filter the indices to get our 1-simplices
# edges = list(zip(rows[edge_mask], cols[edge_mask]))

# sorrting the edges in increasing order of their  (distances)
# 1. Get the indices that sort the distances from smallest to largest
sort_indices = np.argsort(upper_distances)

# 2. Reorder everything using these indices!
sorted_distances = upper_distances[sort_indices]
sorted_rows = rows[sort_indices]
sorted_cols = cols[sort_indices]

# Now we have our perfectly ordered list of 1-simplices (edges)
edges_sorted = list(zip(sorted_rows, sorted_cols))


#Finding 2-simplices (triangles) from the edges :
# A triangle exists if all the three edges between the three vertices exist. So we can iterate through all pairs of edges and check if they form a triangle. We also need to ensure that we only count each triangle once, so we can enforce an ordering on the vertices (e.g., i < j < k).
#Therefore for every edge ij, we look for a vertex k which has edge with i and j.
triangle = []

# for edge in edges:
#     if edge[0] < edge[1]:
#         i = edge[0]
#         j = edge[1]
    
#         for edge_2 in edges:
#             if edge_2[0] == i and edge_2[1] != j:
#                 k = edge_2[1]
#                 if (j, k) in edges or (k, j) in edges and j < k:
#                     triangle.append([i, j, k])


for edge in edges_sorted:
    if edge[0] < edge[1]:
        i = edge[0]
        j = edge[1]
    
        for edge_2 in edges_sorted:
            if edge_2[0] == i and edge_2[1] != j:
                k = edge_2[1]
                if (j, k) in edges_sorted or (k, j) in edges_sorted and j < k:
                    triangle.append([i, j, k])


#Creating boundary matrix :
M = len(edges_sorted) # number of edges
N = len(triangle) # number of triangles

boundary_matrix = np.zeros((M, N), dtype=int)

edge_to_idx = {edge: idx for idx, edge in enumerate(edges_sorted)}

for col_idx, tri in enumerate (triangle):
    e_1 = (tri[0], tri[1])
    e_2 = (tri[1], tri[2])
    e_3 = (tri[0], tri[2])

    for edge in [e_1, e_2, e_3]:
        row_idx = edge_to_idx[edge] # get the row index for this edge
        boundary_matrix[row_idx, col_idx] = 1 # set the corresponding entry to 1

# this sets all the edges that are part of a triangle to 1 in the boundary matrix, which is what we need for computing homology. 

def get_lowest_one(col):
    indices = np.where(col == 1)[0]
    if len(indices) == 0:
        return -1
    else:
        return indices[-1] # return the largest index where col is 1

lowest_ones = {} # {} creates a dcitionayry that has keys and values.


# Dictionary to keep track of claimed rows
lowest_ones = {}
num_columns = boundary_matrix.shape[1]

# Loop through every column (triangle)
for j in range(num_columns):
    # Find the initial lowest 1 for this column
    pivot_row = get_lowest_one(boundary_matrix[:, j])
    
    # Keep reducing as long as the pivot is already claimed
    while pivot_row != -1 and pivot_row in lowest_ones:
        # 1. Find who already claimed this row
        claiming_col_idx = lowest_ones[pivot_row]
        
        # 2. XOR the older column into our current column
        boundary_matrix[:, j] = boundary_matrix[:, j] ^ boundary_matrix[:, claiming_col_idx]
        
        # 3. Recalculate the lowest 1 for the updated column
        pivot_row = get_lowest_one(boundary_matrix[:, j])
    
    # If the column didn't completely zero out, it claims this new pivot!
    if pivot_row != -1:
        lowest_ones[pivot_row] = j



#boundary matrix for H_0

M_2 = len (point_cloud)

N_2 = len (edges_sorted)

# 1. Convert the Keras Tensor back into a standard NumPy array
boundary_matrix_2 = np.zeros((M_2, N_2), dtype=int)



for col_idx, edge in enumerate(edges_sorted):
    point_a_idx = edge[0]
    point_b_idx = edge[1]
    
    # An edge connects two points, so we put a 1 in their respective rows
    boundary_matrix_2[point_a_idx, col_idx] = 1
    boundary_matrix_2[point_b_idx, col_idx] = 1

lowest_edges = {}

for j in range(N_2):
    pivot_row = get_lowest_one(boundary_matrix_2[:, j])

    while pivot_row != -1 and pivot_row in lowest_edges:
        claiming_col = lowest_edges[pivot_row] #vertex that claims the pivot row
        boundary_matrix_2[:, j] = (boundary_matrix_2[:, j] ^ boundary_matrix_2[:, claiming_col]) #XOR operation
        pivot_row = get_lowest_one(boundary_matrix_2[:, j])

    if pivot_row != -1:
        lowest_edges[pivot_row] = j


        #lowest edges has row_index with the lowest one as key and the column index of the edge that claims it as value.

holes = []

for row_idx, col_idx in lowest_edges.items():
    birth_time = 0
    death_time = dist_matrix[edges_sorted[col_idx][0], edges_sorted[col_idx][1]]
    holes.append((birth_time, death_time))
    print(f"H0 Loop - Birth: {birth_time:.3f}, Death: {death_time:.3f}")
    


h1_intervals = []

# Extract the Birth and Death times
for row_idx, col_idx in lowest_ones.items():
    # 1. Look up the edge that created the hole and its birth time
    edge = edges_sorted[row_idx]
    birth_time = dist_matrix[edge[0], edge[1]]
    
    # 2. Look up the triangle that filled the hole
    tri = triangle[col_idx]
    
    # 3. The death time is the longest edge of that triangle
    death_time = max(
        dist_matrix[tri[0], tri[1]],
        dist_matrix[tri[1], tri[2]],
        dist_matrix[tri[0], tri[2]]
    )
    
    h1_intervals.append((birth_time, death_time))
    print(f"H1 Loop - Birth: {birth_time:.3f}, Death: {death_time:.3f}")

# --- Plotting the Persistence Diagram ---
# Separate the coordinates for plotting
births = [interval[0] for interval in h1_intervals]
deaths = [interval[1] for interval in h1_intervals]

births_0 = [interval[0] for interval in holes]
deaths_0 = [interval[1] for interval in holes]

plt.figure(figsize=(6, 6))
# Plot our H1 features
plt.scatter(births, deaths, color='orange', label='H1 (Loops)', s=100, zorder=5)

#plotting H0 features
plt.scatter(births_0, deaths_0, color='blue', label='H0 (Connected Components)', s=100, zorder=5)   

# Add the diagonal y=x line
# Everything must appear above this line because a feature cannot die before it is born!
max_val = max(deaths) + 0.5 if deaths else 2.0
plt.plot([0, max_val], [0, max_val], linestyle='--', color='gray', label='y=x')

plt.xlabel('Birth (Epsilon)')
plt.ylabel('Death (Epsilon)')
plt.title('Persistence Diagram (Raw data)')
plt.legend()
plt.grid(True)
plt.show()





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
model.fit(x_train, y_train, epochs=100, batch_size=600)

import tensorflow as tf

# Assuming your original model is saved in a variable called 'model'
# and your 256-neuron layer is named 'dense_1' (you can check model.summary() for the name)

# 1. Create a new model that stops at your target layer
feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer('dense_2').output # Replace 'dense_1' with your layer's actual name
)

# 2. Pass your 20 images through this extractor
# This is your NEW point cloud!
point_cloud_256D = feature_extractor.predict(x_train[:50]) # Extracts the output of 2nd dense layer

# 3. Convert to a standard numpy array (just to be safe)
point_cloud = point_cloud_256D.numpy() if hasattr(point_cloud_256D, 'numpy') else point_cloud_256D


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
plt.title('Persistence Diagram (final Dense layer output)')
plt.legend()
plt.grid(True)
plt.show()


# 1. Find the column index of the last edge that merged an H_0 component
# The highest column index corresponds to the longest edge in your sorted list
max_col_idx = max(lowest_ones.keys())

# 2. Retrieve the exact pair of points that this edge connected
max_edge = edges_sorted[max_col_idx]
point_a_idx = max_edge[0]
point_b_idx = max_edge[1]

# 3. Look up the real, human-readable labels from your raw data
label_a = np.argmax(y_train[point_a_idx])
label_b = np.argmax(y_train[point_b_idx])

# 4. Print the big reveal!
print(f"The most persistent gap was bridged by the edge connecting Image {point_a_idx} and Image {point_b_idx}.")
print(f"Image {point_a_idx} is the digit: '{label_a}'")
print(f"Image {point_b_idx} is the digit: '{label_b}'")


img_1 = x_train[point_a_idx].numpy().reshape(28, 28)
img_2 = x_train[point_b_idx].numpy().reshape(28, 28)

plt.figure(figsize=(8, 6))
plt.suptitle("Farthest points", fontsize=16, fontweight='bold')

# Top Vertex of the Triangle
plt.subplot(1, 2, 1)
plt.imshow(img_1, cmap='gray')
plt.title(f"Image {label_a}\n(Digit: {label_a})")
plt.axis('off')

# Bottom-Left Vertex
plt.subplot(1, 2, 2)
plt.imshow(img_2, cmap='gray')
plt.title(f"Image {label_b}\n(Digit: {label_b})")
plt.axis('off')

# Display the plot cleanly
plt.tight_layout()
plt.show()


# 1. Find the triangle with the highest index (This is the highest Death time)
# Since your columns (triangles) are sorted by size, the max index is the last one to form
max_death_col_idx = max(lowest_ones.keys())

# 2. Find the specific edge that originally closed this loop (The Birth time)
birthed_row_idx = lowest_ones[max_death_col_idx]



# 3. Retrieve the actual sets of points (image indices) from your sorted lists
killer_triangle = triangle[max_death_col_idx]
creator_edge = edges_sorted[birthed_row_idx]

# 4. Print out the raw image indices involved in this massive loop
print(f"The H_1 hole with the highest death was born by Edge: {creator_edge}")
print(f"It was finally filled in (killed) by Triangle: {killer_triangle}")

# 5. Look up the real labels of the 3 images that formed that final, massive triangle
print("\nThe 3 images that finally formed the triangle to fill this void are:")
for point_idx in killer_triangle:
    # Assuming y_train is one-hot encoded like your previous test
    label = np.argmax(y_train[point_idx])
    print(f"Image {point_idx} is the digit: '{label}'")

import matplotlib.pyplot as plt
import numpy as np

# 1. Unpack your 3 indices from the killer_triangle variable
# (Assuming killer_triangle looks something like (12, 45, 8))
idx_1, idx_2, idx_3 = killer_triangle

# 2. Extract the raw images, convert from Tensor to NumPy, and reshape to 28x28
img_1 = x_train[idx_1].numpy().reshape(28, 28)
img_2 = x_train[idx_2].numpy().reshape(28, 28)
img_3 = x_train[idx_3].numpy().reshape(28, 28)

# 3. Extract their actual human-readable labels
label_1 = np.argmax(y_train[idx_1])
label_2 = np.argmax(y_train[idx_2])
label_3 = np.argmax(y_train[idx_3])

# 4. Set up the plotting canvas
plt.figure(figsize=(8, 6))
plt.suptitle("The 3 Images Forming the H_1 Killer Triangle", fontsize=16, fontweight='bold')

# Top Vertex of the Triangle
plt.subplot(2, 3, 2)
plt.imshow(img_1, cmap='gray')
plt.title(f"Image {idx_1}\n(Digit: {label_1})")
plt.axis('off')

# Bottom-Left Vertex
plt.subplot(2, 3, 4)
plt.imshow(img_2, cmap='gray')
plt.title(f"Image {idx_2}\n(Digit: {label_2})")
plt.axis('off')

# Bottom-Right Vertex
plt.subplot(2, 3, 6)
plt.imshow(img_3, cmap='gray')
plt.title(f"Image {idx_3}\n(Digit: {label_3})")
plt.axis('off')

# Display the plot cleanly
plt.tight_layout()
plt.show()
# Topological Interpretability of Neural Network Latent Spaces via Persistent Homology

## Overview
Deep neural networks compress high-dimensional, rigid pixel data into complex, lower-dimensional latent manifolds. However, the geometric rules governing these "decision spaces" are notoriously opaque. This project applies **Topological Data Analysis (TDA)**—specifically persistent homology—to reverse-engineer and interpret the hidden layers of a trained neural network.

Rather than relying on pre-built TDA libraries, this project features a **custom, from-scratch implementation** of a Vietoris-Rips complex and $\mathbb{Z}_2$ boundary matrix reduction. By tracking the birth and death of $H_0$ connected components and $H_1$ topological voids, we can mathematically map how the network structurally isolates distinct data classes.

## Key Features
* **Latent Space Extraction:** Bypasses the final softmax probability simplex to extract raw continuous geometry from the 256-dimensional hidden layer of a Keras DNN trained on MNIST.
* **Custom Simplicial Complex:** Computes exact pairwise Euclidean distance matrices and dynamically builds $1$-simplices (edges) and $2$-simplices (triangles) based on continuous filtration limits ($\epsilon$).
* **From-Scratch Matrix Reduction:** Implements the core algebraic topology of persistent homology. Constructs boundary matrices ($\partial_1$ and $\partial_2$) and performs modulo-2 arithmetic (XOR) column reduction to pair topological births and deaths using the "lowest-1" pivot algorithm.
* **Visual Interpretability:** Connects abstract topological features directly back to the raw image data. The script isolates and visualizes the specific images that act as the structural anchors for the most persistent geometric voids in the network.

## Dependencies
* Python 3.x
* TensorFlow / Keras (for model training and latent space extraction)
* NumPy (for tensor manipulation and matrix algebra)
* Matplotlib (for plotting persistence diagrams and raw image anchors)
* *Note: Requires custom local modules `Data.py` (for dataset loading) and `Building_DNN.py` (for the Keras sequential architecture).*

## Usage
Run the main script to train the model, extract the 256D point cloud, and execute the TDA pipeline:

```bash
python main.py
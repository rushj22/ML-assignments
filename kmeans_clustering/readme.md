We had to write a routine/class called ‘my kMeans’ that implements k-means clustering. The code is written in python and Scikit Learn is used. The clustering code was run on the following datasets (uploaded here) from the UCI Machine Learning Repository:
1. Seeds
2. Vertebral Column Data Set - (use the column 3C.dat)
3. Image Segmentation - Use only the training data to cluster.
4. Iris dataset

Visualization and qualitative evaluation:
t-Distributed Stochastic Neighbor Embedding (t-SNE) is a very useful tool for visualization of high-dimensional
data. For each of the datasets, we visualized the points using t-SNE.

Quantitative Evaluation:
The following metrics were used for quantitative comparison of the clustering outputs with the ground truth
labels:
1. Normalized Mutual Information (MI)
2. Adjusted Mutual Information (AMI)
3. Rand Index (RI)
4. Adjusted Rand Index (ARI)
For different values of k, an appropriate table for metric was calculated (drawn in the report attached)

The following images for each of the four datasets was plotted.
1. 2D scatter after applying t-SNE. The data should be color coded according to the ground truth labels
(available from UCI data).
2. 2D scatter after applying t-SNE. The data should be color coded according to the clustering labels
(from your code) from any one of your runs.
3. Plot of objective function vs iteration no. for true value of k.


To run the code :

imports needed. please install the following through the terminal
numpy
operator
random
sklearn.manifold
matplotlib.pyplot
sklearn

Run the python file main.py
It will ask for a input : Enter a number for from 0-3
0 seeds_dataset.txt
1 iris.data
2 segmentation.data
3 column_3C.dat

The second input asks for the value of K (number of clusters)
Click enter

The function my_kMeans (X1, initial_centroids, max_iters) will be called from the file my_kmeans.py
which in turn calls other functions to implement Kmeans and objective function

Two plots will be shown, one for the original data and one for the implemented Kmeans.
Click enter on the terminal again

Another plot of Objective loss plot will be shown

A list will be displayed in the terminal in the end which consists of 4 comma separated values in the order MI, AMI, RI, ARI for the given plot as compared to the original plot

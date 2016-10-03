import numpy as np
from operator import itemgetter
from random import randint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import metrics
# X:- Data set matrix where each row of X is represents a single training example.
# initial_centroids:- Matrix storing initial centroid position.
# max_iters:- Maximum number of iterations that K means should run for.

# newCentroid:- Matrix storing final cluster centroids.
# evaluationMatrix:- Array that returns MI, AMI, RI, ARI.

#x = np.array([1,2,3,4,5])
#np.linalg.norm(x)

# a = [[40, 10], [50, 11]]
# arr = np.mean(a, axis=1)
# print arr

truth_label = []

def RI_calc(truth_label,mini_dist):
	a=0
	b=0
	n=len(mini_dist)
	for i in range (0,n):
		for j in range (0,n):
			if(i<j):
				if(mini_dist[i][1]==mini_dist[j][1] and truth_label[i]==truth_label[j]):
					a=a+1
				elif(mini_dist[i][1]!=mini_dist[j][1] and truth_label[i]!=truth_label[j]):
					b=b+1
	RI=float(a+b)/(n*(n-1)/2)
	return RI

def ARI_calc(truth_label,mini_dist):
	calc=[]
	for i in range (0,len(mini_dist)):
		calc.append(mini_dist[i][1])
	return metrics.adjusted_rand_score(truth_label,calc)

def AMI_calc(truth_label,mini_dist):
	calc=[]
	for i in range (0,len(mini_dist)):
		calc.append(mini_dist[i][1])
	return metrics.adjusted_mutual_info_score(truth_label,calc)

def NMI_calc(truth_label,mini_dist):
	calc=[]
	for i in range (0,len(mini_dist)):
		calc.append(mini_dist[i][1])
	return metrics.normalized_mutual_info_score(truth_label,calc)

def plotgraph(X1,mini_dist,truth_label):
	x_data = []
	y_data = []
	z = []
	X2= np.array(X1)
	model = TSNE(learning_rate = 200.0)
	np.set_printoptions(suppress=True)
	X = model.fit_transform(X2) 
	for i in range (0,len(X)):
		#print mini_dist[i][1]
		x_data.append(X[i][0])
		y_data.append(X[i][1])
		z.append(mini_dist[i][1])
	#print x_data
	#print y_data
	#print z
	original = plt.figure(1)
	plt.scatter(x_data, y_data, s=80, c=truth_label, marker="+")
	plt.title("original")
	original.show()
	original.savefig('original')

	new = plt.figure(2)
	plt.scatter(x_data, y_data, s=80, c=z, marker="+")
	plt.title("modified")
	new.show()
	original.savefig('modified')

def plotobj(obj):
	x_data = []
	y_data = []
	for i in range (0,len(obj)):
		#print mini_dist[i][1]
		x_data.append(i+1)
		y_data.append(obj[i])

	#print x_data
	#print y_data
	#print z
	#print y_data
	obj_p = plt.figure(3)
	plt.scatter(x_data, y_data)
	plt.title("objfunc")
	for ctr in xrange(1, len(obj)):
		plt.plot([x_data[ctr-1], x_data[ctr]], [y_data[ctr-1], y_data[ctr]], linestyle="-", c='b')
	obj_p.show()
	obj_p.savefig('objfunc')

	raw_input()
def objective_function(X,mini_dist,K,newCentroids):
	obj_loss = 0
	# print len(X)
	# print len(mini_dist)
	for i in range (0,len(X)):
		for j in range (0,K):
			arr=[]
			for k in range (0,len(X[0])):
				if(mini_dist[i]==j):
					arr.append((X[i][k]-newCentroids[j][k]))
			for ele in range (0,len(arr)):
				obj_loss = obj_loss + (arr[ele]*arr[ele])
	return obj_loss

def centroid_compute(X,mini_dist,K,newCentroids):
	new_centroids = []
	sum = 0
	for i in range (0,K):
		arr = [];
		for j in range (0,len(mini_dist)):
			if(mini_dist[j][1]==i):
				arr.append(X[j])
		if(len(arr)!=0):
			new_centroids.append(np.mean(arr,axis=0))
		else:
			new_centroids.append(newCentroids[i]);
	return new_centroids

def dist_from_centroids(X,centroids,K):
	mini =[]
	for i in range (0,len(X)):
		mini.append([])
	for i in range (0,len(X)):
		for j in range (0,K):
			arr=[]
			for k in range (0,len(X[0])):
				arr.append((X[i][k]-centroids[j][k]))
			x=np.linalg.norm(arr)
			if(len(mini[i])==0):
				mini[i].append(x)
				mini[i].append(j)
			else:
				if(mini[i][0]>x):
					mini[i][0]=x
					mini[i][1]=j
	return mini

def my_kMeans (X1, initial_centroids, max_iters):
	# your code here
	X=X1[0]
	# print X
	# return 
	truth_label=X1[1]
	K=len(initial_centroids)
	newCentroids = initial_centroids
	evaluationMatrix = 0
	obj_val=[]
	x=objective_function(X,truth_label,K,newCentroids)
	# print x
	obj_val.append(x+1000)
	for i in range (0,max_iters):
		mini_dist=dist_from_centroids(X,newCentroids,K)
		newCentroids=centroid_compute(X,mini_dist,K,newCentroids)
		new_mini_dist = []
		for j in range (0,len(mini_dist)):
			new_mini_dist.append(mini_dist[j][1])
		x=objective_function(X,new_mini_dist,K,newCentroids)
		# print x
		obj_val.append(x)
	plotgraph(X,mini_dist,truth_label)
	plotobj(obj_val)
	# print(mini_dist)
	evaluationMatrix = []
	evaluationMatrix.append(NMI_calc(truth_label,mini_dist))
	evaluationMatrix.append(AMI_calc(truth_label,mini_dist))
	evaluationMatrix.append(RI_calc(truth_label,mini_dist))
	evaluationMatrix.append(ARI_calc(truth_label,mini_dist))
	return newCentroids, evaluationMatrix;

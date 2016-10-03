import numpy as np
from operator import itemgetter
from random import randint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import metrics
from my_kmeans import my_kMeans

X1=[]
X1.append([])
X1.append([])
filename = input('Enter filenumber: ')
#0 seeds_dataset.txt
#1 iris.data
#2 segmentation.data
#3 column_3C.dat
truth_label=[]
if(filename == 0):
	with open('seeds_dataset.txt') as f:
		lines = f.readlines()
	labeling = {}
	labeling[1]=0
	labeling[2]=1
	labeling[3]=2
	for line in lines:
		line = line.split()
		X1[1].append(labeling[int(line[-1])])
		del line[-1]
		for i in range(0,len(line)):
			line[i]=float(line[i])
		X1[0].append(line)

elif(filename == 1):
	with open('iris.data') as f:
		lines = f.readlines()
	labeling = {}
	labeling['Iris-setosa']=0
	labeling['Iris-versicolor']=1
	labeling['Iris-virginica']=2
	for line in lines:
		line = line.split(',')
		# print line[-1].split('\n')
		# print (labeling[line[-1].split('\n')[0]])
		X1[1].append((labeling[line[-1].split('\n')[0]]))
		del line[-1]
		for i in range(0,len(line)):
			line[i]=float(line[i])
		X1[0].append(line)

elif(filename == 2):
	with open('segmentation.data') as f:
		lines = f.readlines()
	labeling = {}
	labeling['BRICKFACE']=0
	labeling['SKY']=1
	labeling['FOLIAGE']=2
	labeling['CEMENT']=3
	labeling['WINDOW']=4
	labeling['PATH']=5
	labeling['GRASS']=6
	for line in lines:
		line = line.split(',')
		X1[1].append((labeling[line[0]]))
		del line[0]
		for i in range(0,len(line)):
			line[i]=float(line[i])
		X1[0].append(line)

elif(filename == 3):
	with open('column_3C.dat') as f:
		lines = f.readlines()
	labeling = {}
	labeling['DH']=0
	labeling['SL']=1
	labeling['NO']=2
	for line in lines:
		line = line.split()
		X1[1].append(labeling[line[-1]])
		del line[-1]
		for i in range(0,len(line)):
			line[i]=float(line[i])
		X1[0].append(line)



#X1=[[1,2],[2,3],[3,4],[3,5]]
# X2= np.array(X1)
# model = TSNE(learning_rate = 200.0)
# np.set_printoptions(suppress=True)
# X = model.fit_transform(X2) 
max_iters= 10
K1= input("Enter K: ")
C=[]
for i in range (0,K1):
	C.append(X1[0][randint(0,len(X1)-1)])
initial_centroids = C
#print X1[0]
newCentroids, evaluationMatrix = my_kMeans(X1,initial_centroids,max_iters)
print "Evaluation metrics NMI, AMI, RI, ARI"
print evaluationMatrix

#print evaluationMatrix
#print newCentroids

# C=[]
# for i in range (0,K2):
# 	C.append(X[randint(0,len(X)-1)])
# initial_centroids = C
# newCentroids, evaluationMatrix = my_kMeans(X,initial_centroids,max_iters)
# print evaluationMatrix

# C=[]
# for i in range (0,K3):
# 	C.append(X[randint(0,len(X)-1)])
# initial_centroids = C
# newCentroids, evaluationMatrix = my_kMeans(X,initial_centroids,max_iters)
# print evaluationMatrix
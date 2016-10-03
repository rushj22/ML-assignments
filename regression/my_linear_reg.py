import numpy as np
from operator import itemgetter
from random import randint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import metrics

def mse(X,data_perc_val,hx):
	squared_sum = 0
	#print hx
	for i in range (0,data_perc_val):
		squared_sum += ((hx[i]-X[1][i])*(hx[i]-X[1][i]))
	return (squared_sum/float(data_perc_val))

def lin_reg(X,phi,max_iter,delta,alpha,threshold):
	theta = []
	hx = []
	for i in range (0,len(X[0])+1):
		theta.append(0)
	data_perc_val = len(X[1])
	#data_perc_val = 10
	for i in range (0,data_perc_val):
		temp_hx = 0
		for j in range (0,len(X[0])+1):
			if j==len(X[0]):
				temp_hx += theta[j]*1
			else:
				temp_hx += theta[j]*X[0][j][i]
		hx.append(temp_hx)
	squared_sum_error = mse(X,data_perc_val,hx)
	iterate = 0
	#print 'yo'
	num = 0
	while(True):
		# num+=1
		# print num
		iterate = iterate + 1 
		for j in range (0,len(X[0])+1):
			temp_sum = 0
			for i in range (0,data_perc_val):
				if j==len(X[0]):
					temp_sum +=((X[1][i]-hx[i])*1)
				else:
					temp_sum += ((X[1][i]-hx[i])*X[0][j][i])

			theta[j] = theta[j] + (alpha*temp_sum/data_perc_val) - (alpha*delta*theta[j]/float(data_perc_val))
		for i in range (0,data_perc_val):
			temp_hx = 0
			for j in range (0,len(X[0])+1):
				if j==len(X[0]):
					temp_hx += theta[j]*1
				else:
					temp_hx += theta[j]*X[0][j][i]
			hx[i]=temp_hx
		pres_squared_sum_error = mse(X,data_perc_val,hx)	
		if(abs(pres_squared_sum_error-squared_sum_error)<=threshold or iterate>=max_iter):
			#print theta
			#print squared_sum_error
			return theta
			break
		else:
			# print theta
			# print squared_sum_error
			squared_sum_error = pres_squared_sum_error
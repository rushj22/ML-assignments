import numpy as np
from operator import itemgetter
from random import randint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import metrics
from my_linear_reg import lin_reg
from sklearn import preprocessing
import math

def scatter_plot(x_data,y_data,hx,filename,phi,data_per,delta,alpha):
	original = plt.figure(data_per+50*phi)
	plt.scatter(x_data,y_data,c='red', marker="+")
	if(phi==0):
		for ctr in xrange(1, len(x_data)):
	 		plt.plot([x_data[ctr-1], x_data[ctr]], [hx[ctr-1], hx[ctr]], linestyle="-", c='b')
	else:
		plt.scatter(x_data,hx,c='blue', marker="+")
	save_name = filename +"  phi= " + str(phi) +" delta="+str(delta) +" alpha="+str(alpha)+" "+str(data_per)+"%\ data predicted and actual.png"
	plt.title(save_name)
	#original.show()
	path_name = "graph_results/"
	original.savefig(path_name + save_name)

def plot_graph(x_data,y_data,filename,phi,delta,alpha):
	obj_p = plt.figure(phi)
	plt.scatter(x_data, y_data)
	save_name = filename + ": phi=" + str(phi) +" delta="+str(delta)+" alpha="+str(alpha)+ ".png"
	plt.title(save_name)
	for ctr in xrange(1, len(x_data)):
		plt.plot([x_data[ctr-1], x_data[ctr]], [y_data[ctr-1], y_data[ctr]], linestyle="-", c='b')
	#obj_p.show()
	path_name = "graph_results/"
	obj_p.savefig(path_name+save_name)

def plot_graph_new(x_data,lin,poly,gaus,filename,phi,delta,alpha):
	obj_p = plt.figure(phi+3)
	plt.scatter(x_data, lin)
	# blue : linear_regression
	# polygon : green
	# gaussian : red
	for ctr in xrange(1, len(x_data)):
		plt.plot([x_data[ctr-1], x_data[ctr]], [lin[ctr-1], lin[ctr]], linestyle="-", c='b')
	plt.scatter(x_data, poly)
	for ctr in xrange(1, len(x_data)):
		plt.plot([x_data[ctr-1], x_data[ctr]], [poly[ctr-1], poly[ctr]], linestyle="-", c='g')
	plt.scatter(x_data, gaus)
	for ctr in xrange(1, len(x_data)):
		plt.plot([x_data[ctr-1], x_data[ctr]], [gaus[ctr-1], gaus[ctr]], linestyle="-", c='r')
	save_name = filename + "combined_plot,delta="+str(delta)+".png"
	plt.title(save_name)
	#obj_p.show()
	path_name = "graph_results/"
	obj_p.savefig(path_name+save_name)

def mse(X,data_perc_val,hx):
	squared_sum = 0
	for i in range (0,data_perc_val):
		squared_sum += ((hx[i]-X[1][i])*(hx[i]-X[1][i]))
	return (squared_sum/float(data_perc_val))
def cross_validation(phi,max_iter,alpha,threshold,filename):
	mse_val = []
	fileValues=[]
	if(filename == 2):
		with open('seeds_dataset.txt') as f:
			lines = f.readlines()
		count = 0
		labeling = {}
		labeling[1]=0
		labeling[2]=1
		labeling[3]=2
		sum_val = []
		temp_line = lines[0].split('\n')
		temp_line = temp_line[0].split()
		for i in range (0,len(temp_line)-1):
			sum_val.append(0)
		for line in lines:
			line = line.split('\n')
			line = line[0].split()
			fileValues.append([])
			if(phi==2):
				for i in range(0,len(line)-1):
					sum_val[i] += float(line[i])	
				for i in range(0,len(line)-1):		
					fileValues[count].append(float(line[i]))
				fileValues[count].append(labeling[int(line[-1])])							
				count+=1
			else:
				for i in range(0,len(line)-1):
					for j in range(1,phi+2):		
						fileValues[count].append(float(line[i])**j)
				fileValues[count].append(labeling[float(line[-1])])
				count+=1
		if(phi==2):
			for i in range(0,len(fileValues[0])-1):
				sum_val[i] = sum_val[i]/float(len(fileValues))
			for i in range(0,len(fileValues[0])-1):
				temp_sum = 0
				for j in range (0,len(fileValues)):
					temp_sum = temp_sum + ((fileValues[j][i]-sum_val[i])**2)
				temp_var = temp_sum / len(fileValues)
				for j in range (0,len(fileValues)):
					temp = math.exp(-(float(fileValues[j][i]-sum_val[i])**2/(2.0*temp_var)))
					fileValues[j][i] = temp
	elif(filename == 3):
		with open('iris.data') as f:
			lines = f.readlines()
		count = 0
		labeling = {}
		labeling['Iris-setosa']=0
		labeling['Iris-versicolor']=1
		labeling['Iris-virginica']=2
		sum_val = []
		temp_line = lines[0].split('\n')
		temp_line = temp_line[0].split(',')
		for i in range (0,len(temp_line)-1):
			sum_val.append(0)
		for line in lines:
			line = line.split('\n')
			line = line[0].split(',')
			fileValues.append([])
			if(phi==2):
				for i in range(0,len(line)-1):
					sum_val[i] += float(line[i])	
				for i in range(0,len(line)-1):		
					fileValues[count].append(float(line[i]))
				fileValues[count].append(labeling[line[-1].split('\n')[0]])							
				count+=1
			else:
				for i in range(0,len(line)-1):
					for j in range(1,phi+2):		
						fileValues[count].append(float(line[i])**j)
				fileValues[count].append(labeling[line[-1].split('\n')[0]])
				count+=1
		if(phi==2):
			for i in range(0,len(fileValues[0])-1):
				sum_val[i] = sum_val[i]/float(len(fileValues))
			for i in range(0,len(fileValues[0])-1):
				temp_sum = 0
				for j in range (0,len(fileValues)):
					temp_sum = temp_sum + ((fileValues[j][i]-sum_val[i])**2)
				temp_var = temp_sum / len(fileValues)
				for j in range (0,len(fileValues)):
					temp = math.exp(-(float(fileValues[j][i]-sum_val[i])**2/(2.0*temp_var)))
					fileValues[j][i] = temp
	elif(filename == 4):
		with open('AirQualityUCI.csv') as f:
			lines = f.readlines()
		lines = lines[0:200]
		count = 0
		sum_val = []
		temp_line = lines[0].split('\n')
		temp_line = temp_line[0].split(',')
		for i in range (0,len(temp_line)-1):
			sum_val.append(0)
		for line in lines:
			line = line.split('\n')
			line = line[0].split(',')
			fileValues.append([])
			if(phi==2):
				for i in range(0,len(line)-1):
					sum_val[i] += float(line[i])	
				for i in range(0,len(line)-1):		
					fileValues[count].append(float(line[i]))
				fileValues[count].append(float(line[-1]))							
				count+=1
			else:
				for i in range(0,len(line)-1):
					for j in range(1,phi+2):		
						fileValues[count].append(float(line[i])**j)
				fileValues[count].append(float(line[-1]))
				count+=1
		if(phi==2):
			for i in range(0,len(fileValues[0])-1):
				sum_val[i] = sum_val[i]/float(len(fileValues))
			for i in range(0,len(fileValues[0])-1):
				temp_sum = 0
				for j in range (0,len(fileValues)):
					temp_sum = temp_sum + ((fileValues[j][i]-sum_val[i])**2)
				temp_var = temp_sum / len(fileValues)
				for j in range (0,len(fileValues)):
					temp = math.exp(-(float(fileValues[j][i]-sum_val[i])**2/(2.0*temp_var)))
					fileValues[j][i] = temp

	fileValues = preprocessing.scale(np.array(fileValues))
	X_new = []
	X_new.append([])
	X_new.append([])
	for i in range (0,len(fileValues[0])-1):
		X_new[0].append([])
	for i in range (0,len(fileValues)):
		for j in range (0,len(fileValues[0])):
			if j == len(fileValues[0])-1:
				X_new[1].append(fileValues[i][j])
			else:
				X_new[0][j].append(fileValues[i][j])
	mse_val = []
	sample_size_perc = []
	sumi = 0
	delta = len(fileValues)-10
	for i in range(0,len(fileValues),10):
		X = []
		X.append([])
		X.append([])
		sample_size = min(10,len(fileValues)-i+1)
		start = i
		end = min(i+10, len(fileValues))
		for j in range (0,len(X_new[0])):
			X[0].append([])
		for j in range (0,len(X_new[0])):
			for k in range (0,start):
				X[0][j].append(X_new[0][j][k])
			for k in range (end,len(fileValues)):
				X[0][j].append(X_new[0][j][k])
		for j in range (0,start):
			X[1].append(X_new[1][j])
		for j in range (end,len(fileValues)):
			X[1].append(X_new[1][j])
		theta = []
		theta = lin_reg(X,phi,max_iter,delta,alpha,threshold)
		#print theta
		hx = []
		X_small = []
		X_small.append([])
		X_small.append([])
		for k in range (0,len(fileValues)):
			temp_hx = 0
			for j in range (0,len(X_new[0])+1):
				if j==len(X_new[0]):
					temp_hx += theta[j]*1
				else:
					temp_hx += theta[j]*X_new[0][j][k]
			hx.append(temp_hx)
			X_small[1].append(X_new[1][k])
		pres_squared_sum_error = mse(X_small,len(X_small[1]),hx)
		sumi = sumi + pres_squared_sum_error

		mse_val.append(pres_squared_sum_error)
		sample_size_perc.append(i)
	mean = sumi/((len(fileValues)/10)+(len(fileValues)%10))
	print 'phi= ',phi
	print 'mean', mean
	std_dev = 0
	for i in range (0,len(mse_val)):
		std_dev+=(mse_val[i]-mean)**2
	std_dev = math.sqrt(std_dev/((len(fileValues)/10)+(len(fileValues)%10)))
	print 'std_dev ',std_dev

def extracting_data(filename,phi,alpha,max_iter,delta,fileValues,originalValues):
	if(filename == 0):
		lines = []
		with open('lin.txt') as f:
			lines = f.readlines()
		filename = 'lin.txt'
		count = 0
		sum_val = []
		temp_line = lines[0].split('\n')
		temp_line = temp_line[0].split(',')
		for i in range (0,len(temp_line)-1):
			sum_val.append(0)
			originalValues.append([])
		for line in lines:
			line = line.split('\n')
			line = line[0].split(',')
			fileValues.append([])
			for i in range (0,len(line)-1):
				originalValues[i].append(float(line[i]))
			if(phi==2):
				for i in range(0,len(line)-1):
					sum_val[i] += float(line[i])	
				for i in range(0,len(line)-1):		
					fileValues[count].append(float(line[i]))
				fileValues[count].append(float(line[len(line)-1]))							
				count+=1
			else:
				for i in range(0,len(line)-1):
					for j in range(1,phi+2):		
						fileValues[count].append(float(line[i])**j)
				fileValues[count].append(float(line[len(line)-1]))
				count+=1
		if(phi==2):
			for i in range(0,len(fileValues[0])-1):
				sum_val[i] = sum_val[i]/float(len(fileValues))
			for i in range(0,len(fileValues[0])-1):
				temp_sum = 0
				for j in range (0,len(fileValues)):
					temp_sum = temp_sum + ((fileValues[j][i]-sum_val[i])**2)
				temp_var = temp_sum / len(fileValues)
				for j in range (0,len(fileValues)):
					temp = math.exp(-(float(fileValues[j][i]-sum_val[i])**2/(2.0*temp_var)))
					fileValues[j][i] = temp

	elif(filename == 1):
		with open('sph.txt') as f:
			lines = f.readlines()
		filename = 'sph.txt'
		count = 0
		sum_val = []
		temp_line = lines[0].split('\n')
		temp_line = temp_line[0].split(',')
		for i in range (0,len(temp_line)-1):
			sum_val.append(0)
			originalValues.append([])
		for line in lines:
			line = line.split('\n')
			line = line[0].split(',')
			fileValues.append([])
			for i in range (0,len(line)-1):
				originalValues[i].append(float(line[i]))
			if(phi==2):
				for i in range(0,len(line)-1):
					sum_val[i] += sum_val[i] + float(line[i])	
				for i in range(0,len(line)-1):		
					fileValues[count].append(float(line[i]))
				fileValues[count].append(float(line[len(line)-1]))							
				count+=1
			else:
				for i in range(0,len(line)-1):
					for j in range(1,phi+2):		
						fileValues[count].append(float(line[i])**j)
				fileValues[count].append(float(line[len(line)-1]))
				count+=1
		if(phi==2):
			for i in range(0,len(fileValues[0])-1):
				sum_val[i] = sum_val[i]/float(len(fileValues))
			for i in range(0,len(fileValues[0])-1):
				temp_sum = 0
				for j in range (0,len(fileValues)):
					temp_sum = temp_sum + ((fileValues[j][i]-sum_val[i])**2)
				temp_var = temp_sum / len(fileValues)
				for j in range (0,len(fileValues)):
					temp = math.exp(-(fileValues[j][i]-sum_val[i])/float(2*temp_var))
					fileValues[j][i] = temp
	elif(filename == 2):
		with open('seeds_dataset.txt') as f:
			lines = f.readlines()
		filename = 'seeds_dataset.txt'
		count = 0
		labeling = {}
		labeling[1]=0
		labeling[2]=1
		labeling[3]=2
		sum_val = []
		temp_line = lines[0].split('\n')
		temp_line = temp_line[0].split()
		for i in range (0,len(temp_line)-1):
			sum_val.append(0)
		for line in lines:
			line = line.split('\n')
			line = line[0].split()
			fileValues.append([])
			if(phi==2):
				for i in range(0,len(line)-1):
					sum_val[i] += float(line[i])	
				for i in range(0,len(line)-1):		
					fileValues[count].append(float(line[i]))
				fileValues[count].append(labeling[int(line[-1])])							
				count+=1
			else:
				for i in range(0,len(line)-1):
					for j in range(1,phi+2):		
						fileValues[count].append(float(line[i])**j)
				fileValues[count].append(labeling[float(line[-1])])
				count+=1
		if(phi==2):
			for i in range(0,len(fileValues[0])-1):
				sum_val[i] = sum_val[i]/float(len(fileValues))
			for i in range(0,len(fileValues[0])-1):
				temp_sum = 0
				for j in range (0,len(fileValues)):
					temp_sum = temp_sum + ((fileValues[j][i]-sum_val[i])**2)
				temp_var = temp_sum / len(fileValues)
				for j in range (0,len(fileValues)):
					temp = math.exp(-(float(fileValues[j][i]-sum_val[i])**2/(2.0*temp_var)))
					fileValues[j][i] = temp
		delta = len(fileValues)-10
	elif(filename == 3):
		with open('iris.data') as f:
			lines = f.readlines()
		filename = 'iris.data'
		count = 0
		labeling = {}
		labeling['Iris-setosa']=0
		labeling['Iris-versicolor']=1
		labeling['Iris-virginica']=2
		sum_val = []
		temp_line = lines[0].split('\n')
		temp_line = temp_line[0].split(',')
		for i in range (0,len(temp_line)-1):
			sum_val.append(0)
		for line in lines:
			line = line.split('\n')
			line = line[0].split(',')
			fileValues.append([])
			if(phi==2):
				for i in range(0,len(line)-1):
					sum_val[i] += float(line[i])	
				for i in range(0,len(line)-1):		
					fileValues[count].append(float(line[i]))
				fileValues[count].append(labeling[line[-1].split('\n')[0]])							
				count+=1
			else:
				for i in range(0,len(line)-1):
					for j in range(1,phi+2):		
						fileValues[count].append(float(line[i])**j)
				fileValues[count].append(labeling[line[-1].split('\n')[0]])
				count+=1
		if(phi==2):
			for i in range(0,len(fileValues[0])-1):
				sum_val[i] = sum_val[i]/float(len(fileValues))
			for i in range(0,len(fileValues[0])-1):
				temp_sum = 0
				for j in range (0,len(fileValues)):
					temp_sum = temp_sum + ((fileValues[j][i]-sum_val[i])**2)
				temp_var = temp_sum / len(fileValues)
				for j in range (0,len(fileValues)):
					temp = math.exp(-(float(fileValues[j][i]-sum_val[i])**2/(2.0*temp_var)))
					fileValues[j][i] = temp
		delta = len(fileValues)-10
	elif(filename == 4):
		with open('AirQualityUCI.csv') as f:
			lines = f.readlines()
		filename = 'AirQualityUCI.csv'
		count = 0
		sum_val = []
		temp_line = lines[0].split('\n')
		temp_line = temp_line[0].split(',')
		for i in range (0,len(temp_line)-1):
			sum_val.append(0)
		lines = lines[0:400]
		for line in lines:
			line = line.split('\n')
			line = line[0].split(',')
			fileValues.append([])
			if(phi==2):
				for i in range(0,len(line)-1):
					sum_val[i] += float(line[i])	
				for i in range(0,len(line)-1):		
					fileValues[count].append(float(line[i]))
				fileValues[count].append(float(line[-1]))							
				count+=1
			else:
				for i in range(0,len(line)-1):
					for j in range(1,phi+2):		
						fileValues[count].append(float(line[i])**j)
				fileValues[count].append(float(line[-1]))
				count+=1
		if(phi==2):
			for i in range(0,len(fileValues[0])-1):
				sum_val[i] = sum_val[i]/float(len(fileValues))
			for i in range(0,len(fileValues[0])-1):
				temp_sum = 0
				for j in range (0,len(fileValues)):
					temp_sum = temp_sum + ((fileValues[j][i]-sum_val[i])**2)
				temp_var = temp_sum / len(fileValues)
				for j in range (0,len(fileValues)):
					temp = math.exp(-(float(fileValues[j][i]-sum_val[i])**2/(2.0*temp_var)))
					fileValues[j][i] = temp
		fileValues = preprocessing.scale(np.array(fileValues))
		delta = len(fileValues)-10
	return delta,fileValues,filename,originalValues

def calc_plot_values(filename,phi,alpha,max_iter,delta,fileValues,threshold,originalValues):
	X_new = []
	X_new.append([])
	X_new.append([])
	for i in range (0,len(fileValues[0])-1):
		X_new[0].append([])
	for i in range (0,len(fileValues)):
		for j in range (0,len(fileValues[0])):
			if j == len(fileValues[0])-1:
				X_new[1].append(fileValues[i][j])
			else:
				X_new[0][j].append(fileValues[i][j])
	mse_val = []
	sample_size_perc = []
	for i in range(50,100,10):
		X = []
		X.append([])
		X.append([])
		sample_size = int((len(X_new[1])*(i))/100.0)
		for j in range (0,len(X_new[0])):
			X[0].append([])
		for j in range (0,len(X_new[0])):
			for k in range (0,sample_size):
				X[0][j].append(X_new[0][j][k])
		for j in range (0,sample_size):
			X[1].append(X_new[1][j])
		theta = lin_reg(X,phi,max_iter,delta,alpha,threshold)
		print theta
		hx = []
		for k in range (0,len(X_new[1])):
			temp_hx = 0
			for j in range (0,len(X_new[0])+1):
				if j==len(X_new[0]):
					temp_hx += theta[j]*1
				else:
					temp_hx += theta[j]*X_new[0][j][k]
			hx.append(temp_hx)
		pres_squared_sum_error = mse(X_new,len(X_new[1]),hx)
		mse_val.append(pres_squared_sum_error)
		sample_size_perc.append(i)
		if (filename == 'lin.txt' or filename == 'sph.txt'):
			scatter_plot(originalValues[0],X_new[1],hx,filename,phi,i,delta,alpha)
	return mse_val

filename = input('Enter filenumber 0.lin.txt 1.sph.txt 2.seeds_dataset.txt 3.iris.data 4.AirQualityUCI.csv: ')
#phi = input('Phi value: ')
option =  input('Enter Option 1: regression 2: cross_validation: ')

if(option==1):
	plot_values = []

	fileValues = []
	fileValues1 = []
	fileValues2 = []
	originalData = []
	originalData1 = []
	originalData2 = []
	mse_val = []
	mse_val1 = []
	mse_val2 = []
	delta = 0
	phi = 0
	if filename == 0:
		alpha = 2.5*(10**(-4))
		max_iter = 10000
		threshold = 0.0001
		delta = 0
	elif filename ==1 :
		alpha = 2.0*(10**(-5))
		max_iter = 3000
		threshold = 0.007
	elif filename == 2:
		threshold = 0.001
		max_iter = 8000
		alpha = 2.0*(10**(-4))
	elif filename == 3:
		threshold = 0.002
		max_iter = 7000
		alpha = 2.0*(10**(-3))
	else:
		threshold = 0.5
		max_iter = 100
		alpha = 0.4

	delta,fileValues,filename1,originalData = extracting_data(filename,phi,alpha,max_iter,delta,fileValues,originalData)
	mse_val = calc_plot_values(filename1,phi,alpha,max_iter,delta,fileValues,threshold,originalData)
	plot_values.append(mse_val)

	phi = 1
	if filename == 0:
		alpha = 2.0*(10**(-5))
		max_iter = 3000
		threshold = 0.007
	elif filename == 1:
		alpha = 2.5*(10**(-8))
		max_iter = 10000
		threshold = 0.0002
	elif filename == 2:
		threshold = 0.004
		max_iter = 7000
		alpha = 2.2*(10**(-7))
	elif filename == 3:
		threshold = 0.001
		max_itr = 7000
		alpha = 2.0*(10**(-5))
	else:
		threshold = 1.2
		max_iter = 80
		alpha = 0.0001
	delta,fileValues1,filename1,originalData1 = extracting_data(filename,phi,alpha,max_iter,delta,fileValues1,originalData1)
	mse_val1 = calc_plot_values(filename1,phi,alpha,max_iter,delta,fileValues1,threshold,originalData1)
	plot_values.append(mse_val1)
	phi = 2
	if filename == 0:
		alpha = 3.2*(10**(-3))
		max_iter = 18000
		threshold = 0.0002
	elif filename == 1:
		alpha = 2.4*(10**(-4))
		max_iter = 18000
		threshold = 0.001
	elif filename == 2:
		threshold = 0.0001
		max_iter = 5000
		alpha = 2.2*(10**(-2))
	elif filename == 3:
		threshold = 0.005
		max_iter = 10000
		alpha = 2.2*(10**(-2))
	else:
		threshold = 1.2
		max_iter = 80
		alpha = 0.003

	delta,fileValues2,filename1,originalData2 = extracting_data(filename,phi,alpha,max_iter,delta,fileValues2,originalData2)
	mse_val2 = calc_plot_values(filename1,phi,alpha,max_iter,delta,fileValues2,threshold,originalData2)
	plot_values.append(mse_val2)
	sample_size_perc = [50,60,70,80,90]
	plot_graph_new(sample_size_perc,plot_values[0],plot_values[1],plot_values[2],filename1,phi,delta,alpha)

else:
	phi = 0
	if filename == 2:
		threshold = 0.001
		max_iter = 8000
		alpha = 2.0*(10**(-4))
	elif filename == 3:
		threshold = 0.002
		max_iter = 7000
		alpha = 2.0*(10**(-3))
	else:
		threshold = 0.5
		max_iter = 100
		alpha = 0.4
	cross_validation(phi,max_iter,alpha,threshold,filename)
	phi = 1
	if filename == 2:
		threshold = 0.004
		max_iter = 7000
		alpha = 2.2*(10**(-7))
	elif filename == 3:
		threshold = 0.001
		max_itr = 7000
		alpha = 2.0*(10**(-5))
	else:
		threshold = 1.2
		max_iter = 80
		alpha = 0.0001
	cross_validation(phi,max_iter,alpha,threshold,filename)
	phi = 2
	if filename == 2:
		threshold = 0.0001
		max_iter = 5000
		alpha = 2.2*(10**(-2))
	elif filename == 3:
		threshold = 0.005
		max_iter = 10000
		alpha = 2.2*(10**(-2))
	else:
		threshold = 1.2
		max_iter = 80
		alpha = 0.003
	cross_validation(phi,max_iter,alpha,threshold,filename)
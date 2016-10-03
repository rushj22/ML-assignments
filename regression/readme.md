We had to write function for linear regression. We could not use any library for regression
We used lin.dat and sph.dat files (attached) for training and testing purposes. We also used following data sets (attached):
1. Iris data set
2. Air Quality
3. Seeds

We had to show following output: (code, output_plots folder, along with the Report (attached))

a) Implement a simple linear regression function without regularization using gradient descent on first
half of the data contained in lin.dat. Compute MSE (Mean Square Error) on the whole data. Gradually
increase your data size to 90% with a quantum of 10. Plot MSE versus amount of data plot. Repeat
same procedure for sph.dat.
Note: In lin.dat and sph.dat, use first column as the variable and second column as output. Follow same
in all other parts of this question.

b) Using the same template of part a, use polynomial and gaussian kernel function on the first half
of the data to compute MSE. Show MSE versus amount of data in the same plot with different color Machine Learning
of the curve. Show your analysis in the report. Remember that same function needs to be used for
implementing kernel in your linear regression function.

c) Visualize both the dataset along with the fit produced from part a and part b of this question. Write
your inference in the report drawn on the above data sets with different models.

d) As discussed in class about regularization, implement ridge regression. Show MSE versus amount of data plot in a separate figure.

e) Upto now you have implemented linear regression for a single variable. Now use data sets provided
from UCI repository to test it for multiple variables. Perform 10-fold cross validation for part d (reg-
ularized linear regression) and specify mean and standard deviation of the error on the test set. Use
features as variables and labels as output for your regression training and testing.

run python main.py
it calls lin_reg(X,phi,max_iter,delta,alpha,threshold) function from my_linear_reg file

 enter a number from 0-4 to choose files:
 0 - lin.txt
 1 - sph.txt
 2 - seeds_dataset
 3 - iris
 4 - airquality

 then enter 1 to run linear, polynomial(upto square power) kernel, gaussian kernel and 2 to run cross validation

 note that cross validation is implemented for multiple variable files only, that is, cross validation will work for files 2,3,4

 appropriate plots are automatically created and saved in graph_results folder. appropriate naming of graphs indicates the various specifications. In MSE plots, linear regression: blue, polynomial: green, gaussian : red
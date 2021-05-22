import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.odr import ODR,Model,RealData
import os

def csv_accumulation():
    """Returns a list with all the csv files placed in the directory of the current python file."""
    csv_files=[]
    for i in os.listdir():
        if ".csv" in i:
            csv_files.append(i)
    return csv_files

csv_files=csv_accumulation()

def getting_data_csv(f_name):
    """ Reads a csv file created by the ultrachiral_gui.py and returns time and data"""
    data=np.genfromtxt(f_name,delimiter=",") # shape:--> (21,8003)
    data = data[:,2:] # Drop the first two columns because they are nan, shape:-> (21,8001)
    data = data[:,:-1] # Drop the final column because is empty(idk why), shape:-> (21,8000)
    data = data[:,360:] # Drop the firts 360 columns, shape:--> (21,7640)
    time=data[0,:]
    data = data[1:,:] # We drop time from our data, shape:--> (20,7640)
    return time,data

time,data=getting_data_csv("empty_data.csv")

def fit_func(B,t):
    """ Fitting function"""
    return B[0]*np.exp(-t/B[1]) + B[2]

def fit(f,x,y,sx=None,sy=None,save_table=True,save_plot=True):
    """
    f: function with the parameters
    x: 1D array, 'Independent' data variables
    y: 1D array, 'Dependent' data variables
    sx: 1D array, Standard deviation of x
    sy: 1D array, Standard deviation of y
    save_table: Boolean, If True we save the parameter table in a txt file
    plot: Boolean, If True we export a graph of our model
    """
    x_plot=np.linspace(x[0],x[-1],1000) # Values in order to plot a smooth line
    mymodel=Model(f) # Our model
    mydata=RealData(x, y, sx=sx, sy=sy)
    myodr=ODR(mydata, mymodel, beta0=[100.,400*10**(-9),1.,]) #Instantiate ODR with out data, model and initial parameter estimate.
    myoutput=myodr.run()
    
    myoutput.pprint() # Print the 'Parameter Table'
    
    now=datetime.datetime.now()
    f_name = now.strftime("%Y-%m-%d_%H-%M-%S")

    if save_table==True:
        with open(f_name + ".txt",'w',encoding="utf-8") as file:
            for i,j in zip(myoutput.beta,myoutput.sd_beta): file.write("{}\t{}\n".format(i,j))
            # 1st Columnn Parameters, 2nd Column Their Standard deviation
    
    if save_plot==True:
        plt.figure(figsize=(10,7))
        plt.errorbar(x,y, xerr=sx, yerr=sy,fmt="bo",ecolor="red",markersize=1)
        # In order to avoid the line created automatically by plt.plot(), we choose fmt="bo" instead of fmt="bo-"
        plt.plot(x_plot,f(myoutput.beta,x_plot),"r-",label="fit",markersize=5)
        plt.title("ODR Fit")
        plt.legend()
        plt.grid()
        plt.savefig(f_name+"_ODR"+".jpg",dpi=400)
        plt.show()

fit(fit_func,time,data[19,:])
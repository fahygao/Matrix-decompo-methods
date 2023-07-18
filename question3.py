#  Demonstration Python module for Week 3
#  Scientific Computing, Fall 2021, goodman@cims.nyu.edu
#  Read a source and target points and target points from files.

import numpy   as np  # general numpy
import scipy.linalg
import time

SourcePointsFile = "SourcePoints"
TargetPointsFile = "TargetPoints"

#   Source points
inFile = open("SourcePoints2.txt","r")  # open the source points file
firstLine = inFile.readline()  # the first line has the number of points
nPoints = int(firstLine)  # convert the number from string to int
sourcePoints = np.zeros([nPoints, 3])  # the source points array
for p in range(nPoints):
    dataLine = inFile.readline()  # there is one point per line
    words = dataLine.split()  # each word is a number
    x = np.float64(words[0])  # x, y, and z coordinates
    y = np.float64(words[1])  # convert from string to float
    z = np.float64(words[2])
    sourcePoints[p, 0] = x  # save the numbers in the numpy array
    sourcePoints[p, 1] = y
    sourcePoints[p, 2] = z
inFile.close()
#   target points

inFile = open("TargetPoints2.txt","r")  # open the source points file
firstLine = inFile.readline()  # the first line has the number of points
nPoints = int(firstLine)  # convert the number from string to int
targetPoints = np.zeros([nPoints, 3])  # the source points array
for p in range(nPoints):
    dataLine = inFile.readline()  # there is one point per line
    words = dataLine.split()  # each word is a number
    x = np.float64(words[0])  # x, y, and z coordinates
    y = np.float64(words[1])  # convert from string to float
    z = np.float64(words[2])
    targetPoints[p, 0] = x  # save the numbers in the numpy array
    targetPoints[p, 1] = y
    targetPoints[p, 2] = z
inFile.close()


def calculate_squre_vector(a,b):
    answer = (a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2
    return answer

def output_time(time_init, time_over):
    return time_over - time_init

def low_algorithm(U,s,Vh,tol):
    A_est = 0
    Sigma  = np.diag(s)
    n = 1
    i = 0
    while(s[i+1]>=tol):
        i+=1
    U_k = U[:,:i+1]
    sig_v = Sigma[:i+1,:i+1]@Vh[:i+1,:]
    return i+1,sig_v, U_k

def calculate_wj(x_j,r):
    return 1/calculate_squre_vector(x_j,r)

def find_the_bk(r,SourcePoints,targetPoints,U_k,sig_v):
    b_k = np.ones([len(targetPoints),1])
    A_est = U_k@sig_v
    for j in range(len(targetPoints)):
        for i in range(len(SourcePoints)):
            b_k[j] += A_est[j,i]@calculate_wj(sourcePoints[i],r)
    return b_k

def find_b(r,sourcePoints,targetPoints,U,s,Vh):
    b = np.ones([len(targetPoints), 1])
    Sigma = np.diag(s)
    A = U@Sigma@Vh
    for j in range(len(targetPoints)):
        for i in range(len(sourcePoints)):
            b[j] += A[j,i]@calculate_wj(sourcePoints[i], r)
    return b


def predict_accuracy(w,tol):
    pre_acc = tol*np.linalg.norm(w,ord=2)
    return pre_acc


n_t = len(targetPoints)
n_s = len(sourcePoints)
A = np.zeros([n_t, n_s])  # the source points array
time_0= time.time()
for j in range(n_t):
    for i in range(n_s):
        A[j,i]= 1/calculate_squre_vector(sourcePoints[i],targetPoints[j])
time_1=time.time()
print("Time need to take for generating A:"+str(output_time(time_0,time_1)))
U, s, Vh = scipy.linalg.svd(A)
tol = 0.01
time_2=time.time()
time_past = time_2-time_1
print("Time need to take for generating SVD:"+str(output_time(time_1,time_2)))

i, sig_v,U_k= low_algorithm(U,s,Vh,tol)
print(U_k.shape)
# print("A_est:"+str(A_est.shape))
time_3=time.time()
time_past = time_3-time_2
print("Time need to take for generating Sigma*V:"+str(output_time(time_2,time_3)))
r = np.random.rand(1, 3)
w  = 1. / np.sum((sourcePoints[:, :] - r[:])**2, axis=1)
A_est = U_k @ sig_v
b_k = A_est@w
time_4=time.time()
print("Time need to take for finding b_k:"+str(output_time(time_3,time_4)))
b = A@w
time_5=time.time()
print("Time need to take for generating b:"+str(output_time(time_4,time_5)))
pred_acc = predict_accuracy(w,tol)
Actul_acc = np.linalg.norm(b_k-b,ord=2)
print("The Predicted Accuracy is:"+str(pred_acc))
print("The Actual Accuracy: is "+str(Actul_acc))

num_r = [1,100,500,1000,2000,3000,4000,5000]

r = 5 * np.random.rand(1, 3)
w_i = 1. / np.sum((sourcePoints[:, :] - r[:]) ** 2, axis=1)
for i in num_r:
    time_call_r = time.time()
    print("When r is:",i)
    for j in range(i):
        b_i = A@w_i
    time_finish_full_rank=time.time()
    # l_i =
    print("The time we need to find the illumination by full rank is: ",output_time(time_call_r,time_finish_full_rank))


for i in num_r:
    print("When r is:", i)
    time_start_low_rank = time.time()
    for j in range(i):
        U, s, Vh = scipy.linalg.svd(A)
        i, sig_v, U_k = low_algorithm(U, s, Vh, tol)
        b_i = U_k@(sig_v@w_i)
    time_finish_low_rank=time.time()
    print("The time we need to find the illumination by low rank is: ",output_time(time_start_low_rank,time_finish_low_rank))




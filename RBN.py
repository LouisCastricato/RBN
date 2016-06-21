import numpy as np
import numpy.linalg as la
import scipy.spatial.distance as dist
mindomain = 0
maxdomain = 2 * np.pi
step = 0.001

def inputfunc(x):
    return np.sin(x)
def outputfunc(x):
    return np.cos(x)


x = np.arange(mindomain, maxdomain, step)
y = inputfunc(x)
v_xy = np.transpose([x,y])
sd = np.std(v_xy)
numfeaturevectors = 20
featurevectors = [None] * (numfeaturevectors+1)
for i in range(0, len(featurevectors)):
    featurevectors[i] = [float(maxdomain - mindomain) * (float(i)/float(numfeaturevectors)), inputfunc(float(maxdomain - mindomain) * (float(i)/float(numfeaturevectors)))]

coefficients = [1/(2*sd**2)] * len(featurevectors)

def comp_distance(xy, features, coefficients):
    lst = [None] * len(xy)
    for i in range(0,len(xy)):
        lst[i] = [None] * len(features)
        for j in range(0, len(features)):
            lst[i][j] = np.exp(-coefficients[j] * dist.euclidean(xy[i], features[j]))
    return lst
def distance_single(xy, features, coefficients):
    lst = [None] * len(features)
    for j in range(0, len(features)):
        lst[j] = np.exp(-coefficients[j] * dist.euclidean(xy, features[j]))
    return lst


iter_count = 1
W = None
for i in range(0,iter_count):
    curStep = comp_distance(v_xy, featurevectors, coefficients)
    W = np.dot(la.pinv(curStep), outputfunc(x))


error = 0
for testval in x:
    check = distance_single([testval, inputfunc(testval)], featurevectors, coefficients)
    error += np.abs(np.dot(check, W) - outputfunc(testval))
print error/len(x)

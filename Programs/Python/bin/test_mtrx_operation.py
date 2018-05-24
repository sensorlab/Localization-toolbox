"""
    Script show how matrix opperation is working in Python using numpy array

"""
import numpy as np

vec = np.array([1, 2, 3])
vec_m = np.array([[1, 2, 3]])

mtrx = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]])

print "Shapes: vec, vec_m, mtrx:", vec.shape, vec_m.shape, mtrx.shape

m = mtrx.T
v = vec_m.T

print m
print v
print m.dot(v)

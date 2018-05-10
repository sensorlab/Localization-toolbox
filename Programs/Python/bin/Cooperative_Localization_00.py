def exp_func(x, a, b, c):
    return a * np.exp(-b/1000.0 * x) + c

def lin_func(x, a, b):
    return -a * x/1000.0 * x + b

def test_convergence(error):
    error_len = len(error)
    x = np.linspace(1, error_len, error_len)
    out = x * 0.0
    conv_fail = 1
    try:            # try exp fit function
        popt, pcov = curve_fit(exp_func, x, error)
        print popt
        out = exp_func(x, *popt)
        if popt[1] > 0.0:
            conv_fail = 0
        return conv_fail, out
    except RuntimeError:
        try:
            popt, pcov = curve_fit(lin_func, x, error)
            print popt
            out = lin_func(x, *popt)
            if popt[0] > 100:
                conv_fail = 0
            return conv_fail, out
        except RuntimeError:
            pass
    return conv_fail, out


def plot_MSE_error(error, label):
    for e, l in zip(error, label):
        pyplot.plot(e, label=l)
    pyplot.xlabel('Iterations')
    pyplot.ylabel('MSE')
    pyplot.legend()
    pyplot.show()


def pad_with_value(array, ref_shape, value):
    out = np.ones(ref_shape) * value
    ix = range(0, len(array))
    out[ix] = array
    return out

"""
    Cooperative_Localization_00:
    ********************
    is an example how to use a toolbox to test different
    cooperative localization algorithms

    Experiment objective:
    =====================
    test several localization algorithms

    Building experiment consists of following steps:
    - setup of radio environment
    - reading or generation of the measurements
    - running experiments
        - LS localization
        - finger printing localization

    Version 0.0: Tomaz Javornik (May 2017)

"""
print "\n************************************************************"
print "*                                                          *"
print "*   Experiment:                                            *"
print "*     Estimate agents location  using cooperative          *"
print "*     localization algorithm                               *"
print "*        Networks:                                         *"
print "*             Anchors: Anchor nodes                        *"
print "*             Agents: Nodes with unknown location          *"
print "*             refAgents: nodes with exact agent location   *"
print "*        Raster_Maps: Region, RSSI maps                    *"
print "*        Measurements: RSSI                                *"
print "*                                                          *"
print "************************************************************\n"

import os
import RE.Radio_Env as REM_Env
import copy
import RE.Raster_Map as REM_Maps
import RE.Radio_Net as REM_Network
import RE.Measurements as REM_Measure
import misc.iniProject as iniProject
import misc.GIS as GIS
import matplotlib.pyplot as pyplot
import numpy as np
from scipy.optimize import curve_fit

"""
    Parameter section
"""
### Radio map or experimental area
Netw_size = 20                                                  # Number of nodes
low_left = [0.0, 0.0]                                           # low left corner x and y
top_right = [1000.0, 1000.0]                                    # to right corner
pixel = 1.0                                                     # map pixel width/height

### Nodes network configuration area
dim = 2                                                         # Dimensions 2 = 2D, 3 = 3D
Conn_per_Node = 9                                               # Number of connections per node
set_anchors_in_corners = True                                   # set anchor in conrer area
n_anchors = 4                                                   # number of anchors
err_thresh = 1.0                                                # mean error location threshold
step_iter = 0.0001                                              # iteration step
n_iter = 1000                                                   # maximum number of iterations
dist_error_pdf = "Normal"                                       # probability density function of distance error
dist_par_1 = 0.0                                                # mean value of normal pdf
dist_par_2 = 100.0                                              # standard deviation of normal pdf
N_experiment = 10                                               # number of experiments

n_x = (top_right[0] - low_left[0])/pixel + 1                    # number of pixels in x
n_y = (top_right[1] - low_left[1])/pixel + 1                    # number of pixels in x
n_x = int(n_x)
n_y = int(n_y)
Area = [low_left[0], low_left[1], pixel, pixel, n_x, n_y]       # Experimental area
LS_fails = 0

# Setup project dir, results dir, plot files
projDir = iniProject.iniDir(iniProject.get_proj_dir(3))

# Setup radio environment
Rect_Plane = REM_Env.RadioEnvironment("Rectangular plane area")
Rect_Plane.set_Region(Area)
Region = Rect_Plane.get_Region()

## Experiment loop

i_experiment = 0
iters = np.array(range(0, n_iter))
ref_shape = iters.shape

mean_error = np.zeros((5, n_iter))
max_error = np.zeros((5, n_iter))
Converg_fails = np.zeros(5, dtype=np.int)


while i_experiment < N_experiment:
    i_experiment = i_experiment + 1
    print "\nStart experiment no: ", i_experiment
    # Generate reference network
    Ref_Net = REM_Network.RadioNetwork("Reference network")
    Ref_Net.add_rnd_Loc(Region, Netw_size)
    if set_anchors_in_corners:
        Ids = Ref_Net.get_RadioNode_Ids()
        Ref_Net.set_RadioNode_Loc(Ids[0], [low_left[0], low_left[1], 0])
        Ref_Net.set_RadioNode_Loc(Ids[1], [low_left[0], top_right[1], 0])
        Ref_Net.set_RadioNode_Loc(Ids[2], [top_right[0], low_left[1], 0])
        Ref_Net.set_RadioNode_Loc(Ids[3], [top_right[0], top_right[1], 0])
    Ref_Net.est_connects("num_conn", Conn_per_Node, dim)
    Ref_Net.add_rnd_dist(dist_error_pdf, dist_par_1, dist_par_2)

    Ref_Net.coop_loc_BP(Region, n_anchors, mean_error, step_iter, dim, n_iter)
    quit()

    ## test LS algorithm
    try:
        Net = Ref_Net.copy_netw("Test net")
        Out = Net.coop_loc_LS(n_anchors, err_thresh, step_iter, dim, n_iter)
        Out = np.transpose(Out)
        conv_fail, fit_error = test_convergence(Out[3])
        if conv_fail == 0:
            mean_error[0] = mean_error[0] + pad_with_value(Out[0], ref_shape, err_thresh)
            max_error[0] = max_error[0] + pad_with_value(Out[3], ref_shape, err_thresh)

        # plot_MSE_error([mean_error, max_error, fit_error], ["Mean", "Max", "Fit"])
        Converg_fails[0] = Converg_fails[0] + conv_fail

    except OverflowError:
        Converg_fails[0] = Converg_fails[0] + 1

print Converg_fails[0]
## Uncomment for ploting  reference network
#Ref_Net.plot_connects('o', 1, False, 0)

## Uncomment for comparing estimated and actual location
#Ref_Net.plot_compare(Net, 'o', 2, True, 0)

## Uncomment for ploting results
labels = ["LS algorithm"]
errors = []
for e, fail in zip(mean_error, Converg_fails):
    n = N_experiment - fail
    if n > 0:
        errors.append(e/n)
    else:
        errors.append(np.zeros(ref_shape))

plot_MSE_error(errors, labels)





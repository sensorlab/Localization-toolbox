"""
    Cooperative_Localization.py
    ===========================
    Contains python classes for building cooperative localization
"""


import numpy as np
import copy
import scipy.stats
import scipy.interpolate
import matplotlib.pyplot as pyplot


ErrorStr = "--- ERROR: "
WarningStr = "--- Warning: "


def im_show(fig_num, vals):
    """
    Plots image as pcolormesh
    :param fig_num: figure number
    :param vals: values to plot
    :return:
    """
    pyplot.figure(fig_num)
    CS = pyplot.pcolormesh(vals)
    pyplot.colorbar(CS)
    pyplot.show()
    return


def generate_xy_mesh(West, South, d_West, d_South, Cols, Rows):
    """
    Generates the mesh grid of west-east and south north values.

    :returns: X -> west-east, Y -> south - north  array of raster points (x, y)
    """
    East = West + d_West * Cols
    North = South + d_South * Rows
    x_tmp = np.linspace(West, East, Cols + 1)
    y_tmp = np.linspace(South, North, Rows + 1)
    X, Y = np.meshgrid(x_tmp, y_tmp)
    return X, Y


def calc_pdf(*arg):
    """
    Calculates the probability density function according to arguments:
    :param arg[0]: pdf name ("Normal", "Uniform", "Rayleigh", "LogNormal")
    :param arg[1]: mean value
    :param arg[2]: standard deviation
    :return:
    """
    if arg[0] == "Normal":
        mean = arg[1]
        std = arg[2]
        d = arg[3]
        return scipy.stats.norm(mean, std).pdf(d)
    elif arg[0] == "Uniform":
        low = arg[1]
        high = arg[2]
        d = arg[3]
        return scipy.stats.uniform(low, high).pdf(d)
    elif arg[0] == "Rayleigh":
        val = abs(arg[1])
        d = arg[2]
        return scipy.stats.rayleigh(val).pdf(d)
    elif arg[0] == "LogNormal":
        mean = arg[1]
        std = arg[2]
        d = arg[3]
        return scipy.stats.lognormal(mean, std).pdf(d)
    else:
        print ErrorStr + "The distribution of noise is not correct!"
        print ErrorStr + "PDF is not calculated!"
        return
    return


class BP_message(object):
    """
    Defines a believes propagation message for belief propagation algorithm
    """
    def __init__(self, origin, destination, distance, shape):
        self.org = origin                               # origin node of message
        self.dest = destination                         # destination node of message
        self.dist = distance                            # measured distance between node i and j
        self.value = np.ones(shape, dtype=float)       # value of message, np.array
        return

    def get_org_dest(self):
        """
        :return: destination
        """
        return self.org, self.dest

    def get_dist(self):
        """
        :return: distance
        """
        return self.dist

    def get_value(self):
        """
        :return: Values
        """
        return self.value

    def set_dist(self, distance):
        """
        Sets distance
        :param distance: distance
        :return:
        """
        self.dist = distance
        return

    def set_org_dest(self, origin, destination):
        """
        Sets origin and destination
        :param origin: origin
        :param destination: destination
        :return:
        """
        self.org = origin
        self.dest = destination
        return


class BP_message_list(object):
    """
    Specifies the believes propagation message lists
    """

    def __init__(self):
        self.messages = []

    def append(self, message):
        """
        Appends a message to the list
        :param message: message
        :return:
        """
        if self.index(message.org, message.dest) == -1:
            self.messages.append(message)
        else:
            print " Message already exists!"
        return

    def index(self, org, dest):
        """
        Returns the index of message specified by origin and destination
        :param org: origin
        :param dest: destination
        :return: index of message in the list
        """
        i = -1
        if not self.messages:
            return -1
        for m in self.messages:
            i = i + 1
            if m.dest == dest and m.org == org:
                return i
        return -1

    def get(self, index):
        """
        Returns message which index is index
        :param index: message index
        :return: message which index is index
        """
        return self.messages[index]

    def size(self):
        """

        :return: number of messages in the list
        """
        return len(self.messages)

    def build(self, distance_matrix, shape):
        """
        Builds a believe propagation message list from a distance matrix.
        Distance matrix is a matrix of distances between node i na node j.
        If there is no connection between nodes, the distance is 0.
        :param distance_matrix: distance matrix
        :param shape: size of the observation area rows, columns
        :return:
        """
        i = -1
        for line in distance_matrix:
            i = i + 1
            j = -1
            for d in line:
                j = j + 1
                if d > 0:
                    m = BP_message(i, j, d, shape)
                    self.append(m)
        return

    def get_value(self, org, dest):
        """
        Returns the message value for message with org origin and
        dest destination.
        :param org: BP message origin
        :param dest: BP message destination
        :return: BP message value
        """
        idx = self.index(org, dest)
        if idx > -1:
            m = self.get(idx)
            return m.value
        else:
            return np.NaN

    def set_value(self, org, dest, val):
        """
        Sets value of the believe propagation message.
        :param org: BP message origin
        :param dest: BP message destination
        :param val: value of message
        :return:
        """
        idx = self.index(org, dest)
        if idx > -1:
            self.messages[idx].value = val
            return True
        else:
            return False

    def compute_marginals(self, beliefs):
        """
        Method computes magrinals from the believes.
        :param beliefs: believes
        :return:
        """
        out = []
        j = -1
        for b in beliefs.swapaxes(2, 0).swapaxes(1, 2):
            j = j + 1
            #print "----------------------"
            #im_show(b)
            prod = b
            for m in self.messages:
#                print j, m.dest
                if j == int(m.dest):
                    #print "----->"
                    #im_show(m.value)
                    prod = np.multiply(prod, m.value)
            norm = np.sum(prod)
            out.append(prod/norm)
            #im_show(prod/norm)
        out = np.array(out)
        return out.swapaxes(2, 0).swapaxes(0, 1)

    def compute_messages(self, beliefs, Area, Pdf):
        """
        Method compute the BP messages
        :param beliefs: beliefs
        :param Area: region
        :param Pdf: probability density function
        :return:
        """
        #print "Compute messages"
        X, Y = generate_xy_mesh(*Area)
        Xj = np.array([X, Y]).swapaxes(2, 0).swapaxes(1, 0)
        h_sampled = np.arange(0.0, 2000.0, 1.0)
        q_sampled = scipy.stats.norm(0, 1).pdf(h_sampled)
        pdf_look_up = scipy.interpolate.interp1d(h_sampled, q_sampled, kind='nearest')

        msgs_old = copy.deepcopy(self)

        for m in self.messages:                             # loop over messages
            m_val = np.zeros(m.value.shape)
            i = -1
            for line in Xj:                                 # loop over observation area
                i += 1
                j = -1
                for xj in line:
                    j += 1
                    p_i =  beliefs[:, :, m.org]
                    # calculate distance form xj
                    m_ji = msgs_old.get_value(m.dest, m.org)
                    dx =  np.square( np.subtract(X, xj[0]) )
                    dy = np.square( np.subtract(Y, xj[1]) )
                    d = np.sqrt( np.add(dx, dy) )
                    d = np.subtract(d, m.dist)
                    # calculate Phi 1
                    # Psi = pdf_look_up(np.abs(d))
                    # calculate Phi 2
                    Pdf = ('Normal', 0.0, 5.0)
                    Psi = calc_pdf(*(Pdf + (d,)))
                    # integration
                    tmp = np.divide(p_i, m_ji)
                    tmp = np.multiply(tmp, Psi)
                    m_val[i, j] = np.sum(tmp)/tmp.size
            self.set_value(m.org, m.dest, m_val)
        return


class BP_beliefs(object):
    """
    Defines a belief for belief propagation algorithm
    """
    def __init__(self, region, n_nodes):
        b_shape = (region[5] + 1, region[4] + 1)
        self.region = region    # region (min_x, min_y, delta_x, delta_y, n_x, n_y)
        self.beliefs = np.zeros((b_shape[0], b_shape[1], n_nodes), dtype=float)
        return

    def get_beliefs(self):
        """
        :return: beliefs
        """
        return self.beliefs

    def set_beliefs(self, vals):
        """
        Sets beliefs
        :param vals: value of beleives
        :return:
        """
        self.beliefs = vals
        return

    def get_shape(self):
        """
        Returns shape of believes.
        :return: beliefs shape
        """
        return self.beliefs.shape

    def get_region(self):
        """
        :return: region
        """
        return self.region

    def set_pixel(self, x, y, node_num, value):
        """
        Sets pixel value
        :param x: x coordinate
        :param y: y coordinate
        :param node_num: node number
        :param value: value
        :return:
        """
        ix = (x - self.region[0])/self.region[2] +0.5
        iy = (y - self.region[1])/self.region[3] +0.5
        if ix < 0:
            ix = 0
        if iy < 0:
            iy = 0
        if ix > self.region[4] - 1:
            ix = self.region[4] - 1
        if iy > self.region[5] - 1:
            iy = self.region[5] - 1
        self.beliefs[ix, iy, node_num] = value
        return

    def set_pixels(self, node_num, values):
        """
        Sets all pixels in the region
        :param node_num: node number
        :param values: value of pixel
        :return:
        """
        b_shape = self.get_shape()
        for i in range(0, b_shape[0]):
            for j in range(0, b_shape[1]):
                self.beliefs[i, j, node_num] = values
        return

    def show_image(self, fig_num, node_number):
        """
        Plots a raster image.
        :param fig_num: figure number
        :param node_number: node number
        :return:
        """
        vals = self.beliefs[:, :, 0]
        vals = vals * 0.0
        for i in node_number:
            vals = vals + self.beliefs[:, :, i]

        min = np.amin(vals)
        max = np.amax(vals)
        vals = vals - min
        if (max - min) != 0:
            vals = vals/(max -  min)
        im_show(fig_num, vals)
        return

    def get_location(self):
        """
        Based on believes returns the location of the node.
        :return:location of the node
        """
        b_shape = self.beliefs.shape
        out = []
        for i in range(0, b_shape[2]):
            ar = self.beliefs[:, :, i]
            idx = np.unravel_index(ar.argmax(), ar.shape)
            x = self.region[0] + idx[0] * self.region[2]
            y = self.region[1] + idx[1] * self.region[3]
            out.append([x, y])
        return out


def BP_localization(Anchors, Distances, Region, N_iter, show_results):
    """
    Returns agents location Agents_xy
    :param Anchors:  array of Anchor locations
    :param Distances:   matrix of measured distances between nodes 0 = no connection, [Anchors, Agents]
    :param Region: Region of operation
    :param N_iter: number of iterations
    :return: Agents locations
    """

    # initialize beliefs
    dist_shape = Distances.shape
    n_nodes = dist_shape[0]
    beliefs = BP_beliefs(Region, n_nodes)

    i = 0
    # Anchors
    for node in Anchors:
        x = node[0]
        y = node[1]
        beliefs.set_pixel(x, y, i, 1.0)
        i = i + 1
    n_Anchors = i
    n_Agents = n_nodes - n_Anchors
    b_shape = (Region[5] + 1, Region[4] + 1)
    # Agents
    val = 1.0 / ( (Region[4] + 1.0) * (Region[5] + 1.0) )
    for i in range(n_Anchors, n_nodes):
        beliefs.set_pixels(i, val)

    # beliefs.show_image(1, "all")

    m_list = BP_message_list()
    m = BP_message(0, 0, 0, b_shape)
    m_list.build(Distances, b_shape)

    bb = beliefs.get_beliefs()
    std = (Region[2] + Region[3]) * (Region[4] + Region[5]) * 0.10

    for i in range(0, N_iter):
        print "Iteration: ", i
        m_list.compute_messages(bb, Region, ('Normal', 0.0, std))
        bb = m_list.compute_marginals(bb)
        beliefs.set_beliefs(bb)
        if show_results:
            print "------- Iteration: ", i, " ---------"
            beliefs.show_image(1, range(n_Anchors, n_nodes))

    out = beliefs.get_location()
    return out


def CR_Lipschitz_const(d, n_Anchors):
    """
    Calculates Lipschitz constant according to Soares 2015 eq. 16
    :param d: numpy array of distance matrix
    :param n_Anchors: number of anchors
    :return: Lipschitz constant
    """

    d = np.array(d)
    adj = d
    adj[np.nonzero(d > 0.0)] = 1.0
    nd = np.sum(adj[n_Anchors:, n_Anchors:], axis=1)
    dmax = max(nd)
    maxAi = max(np.sum(adj[:n_Anchors, n_Anchors:], axis=0))
    return 2 * dmax + maxAi

def CR_proj(x, d):
    """
    This function calculate projection of point d on a set x
        d * x
    p = ------- x
        x * x
    :param x: projection plane
    :param d: vector
    :return:
    """
    d = np.array(d)
    x = np.array(x)
    len_d = d.size
    dim = x.size/len_d
    one = np.ones((dim, 1))
    z = np.reshape(x, (len_d, dim))
    # calculate norm vector of x
    Z = np.dot(z, z.T)
    Z = np.diag(Z)
    normx = np.sqrt(Z)
    out = normx > d.flatten()
    out = np.array(out, dtype=float)
    out = np.multiply(one, out.T).flatten()
    tmp = np.divide(d.flatten(), normx)
    d_normx = np.multiply(one, tmp.T).flatten()

    return_val = np.multiply(out, np.multiply(x.flatten(), d_normx)) + np.multiply((1-out), x.flatten())
    return return_val.reshape(return_val.size, 1)


def CR_to_arc_node_matrix(mtrx, dim):
    """
    Function generates arc node matrix (number of edges, number of nodes)
    :param mtrx: input distance matrix
    :param dim: matrix dimension
    :return: arc node matrix [number of edges, number of nodes]
    :return: distance vector
    """
    import networkx as nx
    G = nx.from_numpy_matrix(mtrx)
    #nx.draw(G)
    #nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))
    #pyplot.show()

    dist = []
    vs = []
    for (u, v) in G.edges():
        vs.append(v)
        dist.append(G.get_edge_data(u, v, 'weight')['weight'])
    dist = np.array(dist)
    vs = np.array(vs)
    dist = dist[np.argsort(vs)]

    mtrx = nx.incidence_matrix(G).todense().T

    mtrx_shape = mtrx.shape
    for i in range(0,mtrx_shape[0]):
        sign = 1.0
        for j in range(0, mtrx_shape[1]):
            mtrx[i, j] = sign * mtrx[i, j]
            if mtrx[i, j] > 0.0:
                sign = -1.0
            if mtrx[i, j] == 0:
                mtrx[i, j] = 0

    return np.kron(mtrx, np.eye(dim)), dist.reshape(dist.size,1), G.number_of_edges()


def CR_gradh(Agents, Anchors, Distances):

    Agents = np.array(Agents)
    Anchors = np.array(Anchors)
    Distances = np.array(Distances)
    n_Anchors = Anchors.shape[0]
    n_Agents = Agents.shape[0]
    dim = Agents.shape[1]

    out = np.zeros((n_Agents, dim))
    for n in range(0, n_Agents):
        vect = Distances[n + n_Anchors, :n_Anchors]
        idx = np.argwhere(vect > 0.0)
        for id in idx:
            delta = Agents[n, :] - Anchors[id, :]
            d = Distances[n + n_Anchors, id]
            out[n, :] = out[n, :] + delta - CR_proj(delta, d).T
    return out.flatten()


def ConvexRelax_localization(Anchors, Distances, Region, N_iter, show_results):
    """
    Returns agents location Agents_xy
    :param Anchors:  array of Anchor locations
    :param Distances:   matrix of measured distances between nodes 0 = no connection, [Anchors, Agents]
    :param Region: Region of operation
    :param N_iter: number of iterations
    :return: Agents locations
    """
    Anchors = np.array(Anchors)
    Anchors_shape = Anchors.shape
    Distances_shape = Distances.shape
    n_Anchors = Anchors_shape[0]
    n_Agents = Distances_shape[0] - n_Anchors
    L = CR_Lipschitz_const(Distances, n_Anchors)    # Lipschitz constant

    # calculate arc node matrix
    mtrx = Distances[n_Anchors:, n_Anchors:]
    dim = Anchors.shape[1]
    C, dist, n_edges = CR_to_arc_node_matrix(mtrx, dim)

    # initialize Agents as random variable
    if Anchors.size < 1.:
        Agents = np.zeros((n_Agents, dim))
    else:
        Agents = np.outer(np.ones((1, n_Agents)), np.sum(Anchors, axis=0)/n_Anchors)
        Agents = Agents + 0.1 * np.random.rand(n_Agents, dim)
#        print Agents
#        Agents = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])

    Agents_shape = Agents.shape
    x = Agents.reshape(Agents.size, 1)
    x_old = x

    for i in range(0, N_iter):
        w = x + (i - 1.0) / (i + 2.0) * (x - x_old)
        #print np.round(w.T, decimals=3)
        Cx = C.dot(w)
        gradg = 0
        if n_edges > 0:
            CT = C.T
            tmp = Cx - CR_proj(Cx, dist)
            gradg = CT.dot(tmp)
        gradh = CR_gradh(w.reshape(Agents_shape), Anchors, Distances)
        gradf = gradg + gradh.reshape(gradh.size, 1)
        # print '--->', np.round(gradh.T, decimals=4)
        # print '----->', np.round(gradg.T, decimals=4)
        x_old = x
        x = w - gradf/L
    return x.reshape(Agents_shape)



if __name__ == '__main__':
    print "BP messages test!"
    reg_max = 10.0
    reg_min = -10.0
    Delta = 1.0
    diff = reg_max - reg_min
    # map region definition
    Area = (reg_min, reg_min, Delta, Delta, int(diff/Delta), int(diff/Delta) + 4)

    # nodes locations
    Nodes = [[reg_min, reg_min], [reg_max, reg_min],
             [reg_max, reg_max], [reg_min, reg_max],
             [reg_min + 0.2*diff, reg_min + 0.3*diff],
             [reg_min + 0.8 * diff, reg_min + 0.2 * diff],
             [reg_min + 0.7 * diff, reg_min + 0.6 * diff],
             [reg_min + 0.3 * diff, reg_min + 0.8 * diff]]

    Anchors = Nodes[0:4]

    # calculation of distance matrix
    n_nodes = len(Nodes)
    Nodes = np.array(Nodes)
    d = np.zeros([n_nodes, n_nodes])
    i = -1
    for n in Nodes:
        i += 1
        j = -1
        for m in Nodes:
            j += 1
            d[i, j] = np.sqrt(np.sum(np.square(np.subtract(n, m))))
    # connection matrix 0 no connection
    conn_mtrx = [[0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [1, 1, 1, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]
    conn_mtrx = [[0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 1, 1, 1],
                 [0, 1, 0, 0, 1, 0, 1, 1],
                 [0, 0, 1, 0, 1, 1, 0, 1],
                 [0, 0, 0, 1, 1, 1, 1, 0]]
    dist_matrix = np.multiply(conn_mtrx, d)
    dist_matrix = d
    dist_matrix = dist_matrix.round(decimals=3)

    # test belief propagation localization
    #out = BP_localization(Anchors, dist_matrix, Area, 5, False)
    out = ConvexRelax_localization(Anchors, dist_matrix, Area, 5, False)
    print out

    # # test CR_proj
    # x = np.array([0.1592, -0.2677, 0.0794, -0.2734, -0.0798, -0.0057, 0.0667, -0.2791, -0.0926, -0.0114, -0.0128, -0.0057])
    # x = x.reshape(x.size, 1)
    # d = np.array([204.4683, 553.7458, 636.8057, 183.4324, 184.4449, 453.8733])
    # d = d.reshape(d.size, 1)
    # o_mat = [0.1592, -0.2677, 0.0794, -0.2734, -0.0798, -0.0057, 0.0667, -0.2791, -0.0926, -0.0114, -0.0128, -0.0057]
    # o = CR_proj(x, d)
    # print o

    # # test CR_to_arc:node_matrix
    # dist_test = [ [ 0,        1.0549,    1.0473,    1.3736,    0.8858,    1.1177,    0.7439,    0.8608],
    #               [  1.0549,         0,    1.4895,    0.9441,    0.6680,    0.6117,    0.2266,    0.7246],
    #               [  1.0473,    1.4895,         0,    1.0802,    0.7503,    0.9460,    1.3051,    0.6701],
    #               [  1.3736,    0.9441,    1.0802,         0,    0.5478,    0.3144,    1.0329,    0.5326],
    #               [  0.8858,    0.6680,    0.7503,    0.5478,         0,    0.2045,    0.5537,    0.1834],
    #               [  1.1177,    0.6117,    0.9460,    0.3144,    0.2045,         0,    0.6368,    0.1844],
    #               [ 0.7439,    0.2266,    1.3051,    1.0329,    0.5537,    0.6368,         0,    0.4539],
    #               [ 0.8608,    0.7246,    0.6701,    0.5326,    0.1834,    0.1844,    0.4539,         0]]
    # dist_test = np.array(dist_test)
    #
    # dim = 2
    # mtrx = dist_test[4:, 4:]
    # out = CR_to_arc_node_matrix(mtrx, dim)
    # arc_node =  [ [ 1,     0,    -1,     0,     0,     0,     0,     0],
    #               [ 0,     1,     0,    -1,     0,     0,     0,     0],
    #               [ 1,     0,     0,     0,    -1,     0,     0,     0],
    #               [ 0,     1,     0,     0,     0,    -1,     0,     0],
    #               [ 0,     0,     1,     0,    -1,     0,     0,     0],
    #               [ 0,     0,     0,     1,     0,    -1,     0,     0],
    #               [ 1,     0,     0,     0,     0,     0,    -1,     0],
    #               [ 0,     1,     0,     0,     0,     0,     0,    -1],
    #               [ 0,     0,     1,     0,     0,     0,    -1,     0],
    #               [ 0,     0,     0,     1,     0,     0,     0,    -1],
    #               [ 0,     0,     0,     0,     1,     0,    -1,     0],
    #               [ 0,     0,     0,     0,     0,     1,     0,    -1]]

    # # test gradh
    # Agents = np.array([[500.0714, 499.8684], [500.0627, 499.8669], [499.9869, 499.8173], [500.0536, 500.0693]])
    # Anchors = np.array([[0, 0], [0, 1000], [1000, 0], [1000, 1000]])
    # Distances = [ [ 0,        0.9582,   0.9680,   1.4345,   0.5967,   0.0678,   0.6811,   0.8590],
    #               [ 0.9582,        0,   1.3759,   1.0058,   1.0158,   1.0815,   0.4457,   1.2272],
    #               [ 0.9680,   1.3759,        0,   1.0504,   0.4969,   0.7822,   0.9500,   0.0958],
    #               [ 1.4345,   1.0058,   1.0504,        0,   0.8326,   1.3978,   0.8751,   0.9263],
    #               [ 0.5967,   1.0158,   0.4969,   0.8326,        0,   0.4846,   0.4014,   0.4679],
    #               [ 0.0678,   1.0815,   0.7822,   1.3978,   0.4846,        0,   0.5433,   0.6912],
    #               [ 0.6811,   0.4457,   0.9500,   0.8751,   0.4014,   0.5433,        0,   0.7310],
    #               [ 0.8590,   1.2272,   0.0958,   0.9263,   0.4679,   0.6912,   0.7310,        0]]
    # Distances = 1000.0*np.array(Distances)
    # CR_gradh(Agents, Anchors, Distances)
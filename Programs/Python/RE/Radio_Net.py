"""
    RE.Radio_Net
    ============
    package contains node and network class.
"""

import os
import numpy as np
import copy
import matplotlib.pyplot as pyplot
import misc.iniProject as iniProject
import misc.GIS
import RE.Raster_Map as Raster_Map
import RE.Cooperative_Localization as CoopLoc

ErrorStr = "--- ERROR: "
WarningStr = "--- Warning: "

def to_num(x):
    """
    Converts string to number.

    :param str x: string to convert

    :returns: number
    :rtype: number
    """
    try:
        return float(x)
    except ValueError:
        print ErrorStr + "Input is not a number!  " + x


def to_str(x):
    """
    Converts number to sting.

    :param float x: number

    :returns: string of x
    :rtype: str
    """
    try:
        return str(x)
    except ValueError:
        return x


def RSSI_to_dist(RSSI_dBm, fc_MHz, tx_pow_dBm, pl_coef):
    """
    Converts measured RSSI level to the distance using FSPL channel model.

    :param float RSSI_dBm: RSSI value in [dBm]
    :param float fc_MHz: carrier frequency in [MHZ]
    :param float tx_pow_dBm: transmit power in [dBm]
    :param float pl_coef: path loss coefficient

    :returns: distance in [m]
    :rtype: float
    """
    x = float(tx_pow_dBm) - float(RSSI_dBm) - 20.0 * np.log10(float(fc_MHz)) + 27.55
    dist = np.power(10, x/10.0/float(pl_coef))
    return dist


class BP_message(object):
    """
    Defines a message for beliefe propaation algorithm
    """
    def __init__(self, origin, destination, shape):
        self.org = origin                               # origin node of message
        self.dest = destination                         # destination node of message
        self.value = np.zeros(shape, dtype=float)       # value of message, np.array
        return

    def get_org_dest(self):
        return self.org, self.dest

    def get_value(self):
        return self.value

    def set_org_dest(self, origin, destination):
        self.org = origin
        self.dest = destination
        return


class RadioNode(object):
    """
    **Defines RadioNode class.**
    """

    def __init__(self, *args):
        if len(args) < 2:
            print ErrorStr + " In Radio Node! At least Node Id has to be specified!"
            return

        self.Id = None                  # node ID
        self.Name = None                # node Name
        self.Loc = [None, None, None]   # radio node location [x, y, Antenna Above ground level]
        self.WGS84 = [None, None, None] # Lat, Long, Altitude above see level
        self.PowdBm = None              # Transmit power in dBm if Tx, Rx power in dBm if Rx
        self.FcMHz = None               # Carrier frequency in MHz
        self.BwMHz = None               # Channel bandwidth in MHz
        self.Chan = None                # Logical channel number
        self.AntType = None             # Antenna name of msi diagram
        self.AntAzimuth = None          # Antenna azimuth in degrees
        self.AntTilt = None             # Antenna tilt in degrees +down, -up
        self.Range = None               # if '0' fixed station, if ~'0': mobile station
        self.TxRx = None                # Transmission type S = simplex, D = duplex

        if (len(args) > 0):
            ArgList = []
            for i in range(len(args)):
                ArgList.append(args[i])
            NumArgs = 0
            # Node Id
            try:
                i = ArgList.index('Id')
                self.Id = to_str(args[i + 1])
                NumArgs = NumArgs + 1
            except ValueError:
                pass
            # Radio Node location [xyz]
            try:
                i = ArgList.index('Loc')
                try:
                    x = to_num(args[i + 1])
                    y = to_num(args[i + 2])
                    z = to_num(args[i + 3])
                except ValueError:
                    print ErrorStr + "Check location coordinate! "
                self.Loc = []
                self.Loc.append(to_str(x))
                self.Loc.append(to_str(y))
                self.Loc.append(to_str(z))
                Lat, Long = misc.GIS.GKtoWGS84(x, y)
                self.WGS84 = []
                self.WGS84.append(to_str(Lat))
                self.WGS84.append(to_str(Long))
                self.WGS84.append(None)
                NumArgs = NumArgs + 1
            except ValueError:
                pass
            # Radio Node location [WGS84]
            try:
                i = ArgList.index('WGS84')
                self.WGS84 = []
                try:
                    Lat = to_num(args[i + 1])
                    Long = to_num(args[i + 2])
                    Alt = to_num(args[i + 3])
                except ValueError:
                    print ErrorStr + "Check location coordinate! "
                self.WGS84.append(to_str(Lat))
                self.WGS84.append(to_str(Long))
                self.WGS84.append(to_str(Alt))
                x, y = misc.GIS.WGS84toGK(Lat, Long)
                if len(self.Loc) == 0:
                    self.Loc.append(to_str(x))
                    self.Loc.append(to_str(y))
                    self.Loc.append(to_str(Alt))
                else:
                    try:
                        self.Loc[0] = to_str(x)
                        self.Loc[1] = to_str(y)
                        self.Loc[2] = to_str(z)
                    except ValueError:
                        pass
                NumArgs = NumArgs + 1
            except ValueError:
                pass
            # Transmission bandwidth in MHz BwMHz
            try:
                i = ArgList.index('BwMHz')
                self.BwMHz = to_str(args[i + 1])
                NumArgs = NumArgs + 1
            except ValueError:
                pass
            # Carrier frequency in MHz
            try:
                i = ArgList.index('FcMHz')
                self.FcMHz = to_str(args[i + 1])
                NumArgs = NumArgs + 1
            except ValueError:
                pass
            # Power in dBm
            try:
                i = ArgList.index('PowdBm')
                self.PowdBm = to_str(args[i + 1])
                NumArgs = NumArgs + 1
            except ValueError:
                pass
            # Antenna type
            try:
                i = ArgList.index('Antenna')
                self.AntType = to_str(args[i + 1])
                NumArgs = NumArgs + 1
            except ValueError:
                pass
            # Antenna azimuth
            try:
                i = ArgList.index('AntAzimuth')
                self.AntAzimuth = to_str(args[i + 1])
                NumArgs = NumArgs + 1
            except ValueError:
                pass
            # Antenna tilt
            try:
                i = ArgList.index('AntTilt')
                self.AntTilt = to_str(args[i + 1])
                NumArgs = NumArgs + 1
            except ValueError:
                pass
            # Node Range
            try:
                i = ArgList.index('Range')
                self.Range = to_str(args[i + 1])
                NumArgs = NumArgs + 1
            except ValueError:
                pass
            # Node Function
            try:
                i = ArgList.index('TxRx')
                self.TxRx = to_str(args[i + 1])
                NumArgs = NumArgs + 1
            except ValueError:
                pass

            if NumArgs == 0:
                print ErrorStr + "In Radio Node! Argument error!"
        return

    def copy(self):
        new_node = copy.deepcopy(self)
        return new_node

    def set_WGS84(self, x):
        self.WGS84 = x
        return

    def set_Chan(self, x):
        self.Chan = x
        return

    def set_Range(self, x):
        self.Range = x
        return

    def set_TxRx(self, x):
        self.TxRx = x
        return

    def get_Id(self):
        """
        :returns: Node Id
        """
        try:
            return self.Id
        except ValueError:
            return None

    def get_Name(self):
        """
        :returns: node name
        """
        try:
            return self.Name
        except ValueError:
            return None

    def set_Name(self, Name):
        """
        Sets node name.

        :param Name: node Name
        :returns:
        """
        self.Name = Name
        return

    def get_Loc(self):
        """
        :returns: node location
        """
        try:
            return self.Loc
        except ValueError:
            return None

    def to_WGS84(self):
        """
        Calculates lat, long from Guass Kreuger coordinates
        and set it in self.WGS84.

        :returns:
        """
        if (self.WGS84[0] == None or self.WGS84[1] == None):
            try:
                xlat, xlong = misc.GIS.GKtoWGS84(self.Loc[0], self.Loc[1])
                self.WGS84[0] = xlat
                self.WGS84[1] = xlong
                self.WGS84[2] = "-100"
            except ValueError:
                print WarningStr + "Lat, Long and altitide not set!"
                self.WGS84[0] = None
                self.WGS84[1] = None
                self.WGS84[2] = None
                return
        else:
            return

    def get_WGS84(self):
        """
        :returns: node location in WGS84 format [lat, long, altitude above see level]
        """
        self.to_WGS84()
        try:
            return self.WGS84
        except ValueError:
            print WarningStr + "WGS84 coordinate not returned!"
            return None

    def get_LatLong(self):
        """
        :returns: node location in WGS84 format lat, long
        """
        self.to_WGS84()
        try:
            return self.WGS84[0], self.WGS84[1]
        except ValueError:
            print WarningStr + "WGS84 coordinate not returned!"
            return None

    def get_Alt(self):
        """
        :returns: altitude above see level in [m]
        """
        try:
            return self.WGS84[2]
        except ValueError:
            return None

    def get_Z(self):
        """
        :returns: altitude above ground level in [m]
        """
        try:
            return self.Loc[2]
        except ValueError:
            return None

    def get_AntAvgl(self):
        """
        :returns: altitude above ground level in [m]
        """
        return self.get_Z()

    def set_Loc(self, loc):
        """
        Sets node location.

        :param loc: location in [x, y, z] format
        :returns:
        """
        try:
            idx = -1
            for x in loc:
                idx += 1
                self.Loc[idx] = to_str(x)
            x = to_num(loc[0])
            y = to_num(loc[1])
            xlat, xlong = misc.GIS.GKtoWGS84(x, y)
            self.WGS84[0] = xlat
            self.WGS84[1] = xlong
        except ValueError:
            print ErrorStr + "In Radio Node! Argument error!"
        return

    def set_LatLong(self, xlat, xlong):
        """
        Sets node location in WGS 84 format.

        :param xlat: latitude in WGS84 dec format
        :param xlong: longitude in WGS84 dec format

        :returns:
        """
        self.WGS84[0] = to_str(xlat)
        self.WGS84[1] = to_str(xlong)
        x, y = misc.GIS.WGS84toGK(to_num(xlat), to_num(xlong))
        self.Loc[0] = to_str(x)
        self.Loc[1] = to_str(y)
        return

    def set_Alt(self, alt):
        """
        Sets alititude above the see level in [m].

        :param alt: altitude in [m]

        :returns:
        """
        try:
            self.WGS84[2] = to_str(alt)
            return
        except ValueError:
            print WarningStr + "Altitude not set!"

    def set_XY(self, x, y):
        """
        Sets node location.

        :param x: node x (east - west)
        :param y: node y (south - north)

        :returns:
        """
        self.setLoc([x, y])
        self.to_WGS84()
        return

    def set_AntAvgl(self, alt):
        """
        Sets node altitude above the ground level.

        :param alt:

        :returns:
        """
        try:
            self.Loc[2] = to_str(alt)
        except ValueError:
            print WarningStr + "Antenna altitude above ground level not set!"
        return

    def set_Z(self, alt):
        """
        Sets node altitude above the ground level.

        :param alt:
        :returns:
        """
        self.setAntAvgl(alt)
        return

    def get_BwMHz(self):
        """
        :returns: node bandwidth in MHz
        """
        return self.getBwMHz

    def get_FcMHz(self):
        """
        :returns: node carrier frequency in MHz
        """
        return self.FcMHz

    def get_PowdBm(self):
        """
        :returns: transmit power in dBm if Tx, Rx power in dBm if Rx
        """
        return self.PowdBm

    def get_AntType(self):
        """
        :returns: antenna type
        """
        return self.AntType

    def get_AntAzimuth(self):
        """
        :returns: antenna azimuth in degrees
        """
        return self.AntAzimuth

    def get_AntTilt(self):
        """
        :returns: antenna tilt in degrees
        """
        return self.AntTilt

    def set_BwMHz(self, bw):
        """
        Sets node frequency bandwidth in MHz.

        :param bw: frequency bandwidth in MHz
        :returns:
        """
        try:
            self.BwMHz = to_str(bw)
        except ValueError:
            print ErrorStr + "Bandwidth not set!"
        return

    def set_FcMHz(self, fc):
        """
        Sets carrier frequency.

        :param fc: carrier frequency in MHz
        :returns:
        """
        try:
            self.FcMHz = to_str(fc)
        except ValueError:
            print WarningStr + "Carrier frequency not set!"
        return

    def set_PowdBm(self, p):
        """
        Sets Rx or Tx power in dBm.

        :param p: power in dBm
        :returns:
        """
        try:
            self.PowdBm = to_str(p)
        except ValueError:
            print WarningStr + "Tx/Rx power not set!"
        return

    def set_AntType(self, ant):
        """
        Sets antenna type.

        :param ant: antenna type
        :returns:
        """
        try:
            self.AntType = to_str(ant)
        except ValueError:
            print WarningStr +  "Antenna type power not set!"
        return

    def set_AntAzimuth(self, azimuth):
        """
        Sets antenna azimuth in degrees.

        :param azimuth: antenna azimuth in degreee
        :returns:
        """
        try:
            self.AntAzimuth = azimuth
        except ValueError:
            print WarningStr + "Antenna azimuth not set!"
        return

    def set_AntTilt(self, tilt):
        """
        Sets antenna tilt.

        :param tilt: antenna tilt in degrees

        :returns:
        """
        try:
            self.AntTilt = tilt
        except ValueError:
            print WarningStr + "Antenna tilt not set!"
        return

    def print_node(self):
        """
        Prints node configuration.

        :returns:
        """
        print "   Radio Node ID: ", self.Id
        print "      Lat, Long, Alt:             [", self.WGS84[0], " ", self.WGS84[1], " ", self.WGS84[2], "]"
        print "      x, y, Avgl:                 [", self.Loc[0], " ", self.Loc[1], " ", self.Loc[2], "]"
        print "      Bw, fc[MHz], P[dBm]:        [", self.BwMHz, " ", self.FcMHz, " ", self.PowdBm, "]"
        print "      Antenna Type, Azim, Tilt:   [", self.AntType, " ", self.AntAzimuth, " ", self.AntTilt, "]"
        return

    def print_node_line(self):
        """
        Prints node configuration.

        :returns:
        """
        print "   Radio Node ID, x, y: \t", self.Id, "\t", self.Loc[0], "\t", self.Loc[1], "\t", self.Loc[2]
        return


    def est_loc_LS(self, radio_env, Measurements_Id, Anchors_Id, Mobiles_Id, pl_coef, ref_anchor_index):
        """
        This method estimates the location of  node using least square method.

        :param radio_env: radio environment
        :param Measurements_Id: Id of measurements
        :param Anchors_Id: Id of Anchor network
        :param Mobiles_Id: Id of Mobile network
        :param pl_coef: Id of path loss coefficient
        :param ref_anchor_index: index of reference Anchor node

        :returns: location x, y, z coordinate of the node
        """
        #    ki = xi^2 + yi^2 + zi^2
        #    | x1-x0, y1-y0, z1-z0 |      | d0^2 - d1^2 - k0 + k1 |       | x |
        #A = | x2-x0, y2-y0, z2-z0 |  B = | d0^2 - d2^2 - k0 + k2 |   r = | y |
        #    | x3-x0, y3-y0, z3-z0 |      | d0^2 - d3^2 - k0 + k3 |       | z |
        #ki = xi^2 + yi^2 + zi^2
        #r = (A^T * A)^-1 * A^T * b /2

        values = []         # measured values
        types = []          # type of values
        Tx_Pow_dBm = []     # transmit power value in dBm
        Loc = []            # location of Anchor nodes
        Freq = []           # Tx frequency in MHz

        anchors = radio_env.get("Network", Anchors_Id)
        measurs = radio_env.get("Measurements", Measurements_Id)
        tx_node_Ids = anchors.get_RadioNode_Ids()

        for tx_node_Id in tx_node_Ids:
            tmp = measurs.get(Anchors_Id, tx_node_Id, Mobiles_Id, self.Id)
            for m in tmp:
                values.append(m.get_Value())
                types.append(m.get_Type())
                loc = anchors.get_RadioNode_Loc(tx_node_Id)
                Loc.append(loc)
                fc_MHz = anchors.get_RadioNode_FcMHz(tx_node_Id)
                Freq.append(fc_MHz)
                pow = anchors.get_RadioNode_PowdBm(tx_node_Id)
                Tx_Pow_dBm.append(pow)
        Loc = np.array(Loc, dtype=float)

        D = []
        for value, typ, fc, pow in zip(values, types, Freq, Tx_Pow_dBm):
            if typ == "RSSI":
                d = RSSI_to_dist(value, fc, pow, pl_coef)
            elif typ == "Dist":
                d = value
            elif typ == "ToA" or typ == "TDoA":
                d = value * 3.0e08
            else:
                print ErrorStr + " Wrong type of measurement " + typ
                return
            D.append(d)

        D = np.array(D)
        Loc_0 = Loc[ref_anchor_index]
        A = np.delete(Loc, ref_anchor_index, 0) - Loc_0

        K = np.sum(Loc * Loc, 1)
        D = D * D
        D_0 = D[ref_anchor_index]
        K_0 = K[ref_anchor_index]
        b = np.delete(K, ref_anchor_index, 0) - np.delete(D, ref_anchor_index, 0) - K_0 + D_0

        out = np.dot(A.transpose(), A)
        out = np.linalg.pinv(out)
        out = np.dot(out, A.transpose())
        out = np.dot(out, b)/2.0
        self.Loc = out
        return

    def est_loc_FP(self, radio_env, Measurements_Id, Anchors_Id, Mobiles_Id, pl_coef, ref_anchor_index):
        """
        This is method returns a location of node using fingerprint method.

        :param radio_env: radio environment
        :param Measurements_Id: Id of measurements
        :param Anchors_Id: Id of Anchor network
        :param Mobiles_Id: Id of Mobile network
        :param pl_coef: Id of path loss coefficient
        :param ref_anchor_index: index of reference Anchor node

        :returns: location x, y, z coordinates of node
        """

        # find finger print from measurements
        values = []         # measured values
        types = []          # type of values
        anchors = radio_env.get("Network", Anchors_Id)
        measurs = radio_env.get("Measurements", Measurements_Id)
        tx_node_Ids = anchors.get_RadioNode_Ids()

        finger_print = []   # finger print
        for tx_node_Id in tx_node_Ids:
            tmp = measurs.get(Anchors_Id, tx_node_Id, Mobiles_Id, self.Id)
            values = []
            for m in tmp:
                if m.get_Type() == "RSSI":
                    values.append(m.get_Value())
            finger_print.append( np.median( np.array(values, dtype="float") ) )

        RSSI_maps = radio_env.get("Raster Map", 'RSSI maps')
        Region = RSSI_maps.get_Region()
        X, Y = Region.generate_xy_mesh()

        dist_min = 999999.00
        x_min = np.nan
        y_min = np.nan
        for x_lin, y_lin in (zip(X, Y)):
            for x, y in zip(x_lin, y_lin):
                Values = RSSI_maps.get_Values(x, y)
                dist = np.linalg.norm(np.array(Values, dtype="float") - finger_print)
                if dist < dist_min:
                    dist_min = dist
                    x_min = x
                    y_min = y
        self.Loc = [x_min, y_min, 0]
        return


class RadioNetwork(object):
    """
    **Defines radio network class:**
    a set or radio nodes
    """

    def __init__(self, *args):
        self.Id = args[0]
        self.RadioNodes = []
        self.Connections = []
        if len(args) > 1:
            # add empty nodes
            for i in range(0, args[1]):
                node = RadioNode("Id", i)
                self.append_RadioNode(node)
        return

    def est_loc_LS(self, radio_env, Measurements_Id, Anchors_Id, pl_exp, ref_anchor_index):
        """
        Estimates locations of nodes in radio network applying least square method.

        :param radio_env: radio environment
        :param Measurements_Id: id of measurements
        :param Anchors_Id: id of anchor network
        :param pl_exp: pathloss exponent
        :param ref_anchor_index: reference anchor  index

        :returns:
        """
        for node in self.RadioNodes:
            node.est_loc_LS(radio_env, Measurements_Id, Anchors_Id, self.Id, pl_exp, ref_anchor_index)
        return

    def est_loc_FP(self, radio_env, Measurements_Id, Anchors_Id, pl_exp, ref_anchor_index):
        """
        Estimate locations of nodes in radio network applying finger printing method.

        :param radio_env: radio environment
        :param Measurements_Id: id of measurements
        :param Anchors_Id: id of anchor network
        :param pl_exp: pathloss exponent
        :param ref_anchor_index: reference anchor index

        :returns:
        """
        for node in self.RadioNodes:
            node.est_loc_FP(radio_env, Measurements_Id, Anchors_Id, self.Id, pl_exp, ref_anchor_index)
        return

    def est_loc_BP(self, radio_env, Measurements_Id, Anchors_Id, Agents_Id, pl_exp, n_iter):
        """
        Estimate locations of nodes in radio network applying BP method.
        :param radio_env:  radio environment
        :param Measurements_Id: id of measurements
        :param Anchors_Id: id of anchor network
        :param pl_exp: pathloss exponent
        :param n_iter: number of iterations
        :return:
        """
        Anchors = radio_env.get("Network", Anchors_Id)
        Agents = radio_env.get("Network", Agents_Id)
        Measurs = radio_env.get("Measurements", Measurements_Id)
        Anchors_Ids = Anchors.get_RadioNode_Ids()
        Agents_Ids = Agents.get_RadioNode_Ids()

        Maps = radio_env.get("Raster Map", "RSSI")
        Region = Maps.get_Region().get_Region()

        # get Anchor locations
        Anchors_Loc = []
        for id in Anchors_Ids:
            loc = Anchors.get_RadioNode_Loc(id)
            fc_MHz = Anchors.get_RadioNode_FcMHz(id)
            tx_pow = Anchors.get_RadioNode_PowdBm(id)
            Anchors_Loc.append([loc[0], loc[1]])

        Node_Ids = Anchors_Ids + Agents_Ids
        Num_measure = Measurs.len()

        dist_matrix = []
        row = []
        i_row = 0
        for ix in range(0, Num_measure):
            powdB = Measurs.get_by_index(ix).get_Value()
            dist = RSSI_to_dist(powdB, fc_MHz, tx_pow, pl_exp)
            row.append(dist)
            i_row = i_row + 1
            if i_row >= len(Node_Ids):
                dist_matrix.append(row)
                row = []
                i_row = 0
        dist_matrix = np.array(dist_matrix)
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = 0.5 *(dist_matrix + np.transpose(dist_matrix))

        Locs = []
        for x in Anchors_Loc:
            Locs.append([float(x[0]), float(x[1])])

        Agents_Loc = CoopLoc.BP_localization(Locs, dist_matrix, Region, n_iter, False)
        Ids = self.get_RadioNode_Ids()
        ii = len(Anchors_Ids)

        for id in Agents_Ids:
            agent = Agents.get_RadioNode(id)
            loc = Agents_Loc[ii]
            agent.set_Loc(loc)
            ii = ii + 1
        return

    def est_loc_CR(self, radio_env, Measurements_Id, Anchors_Id, Agents_Id, pl_exp, n_iter):
        """
        Estimate locations of nodes in radio network applying BP method.
        :param radio_env:  radio environment
        :param Measurements_Id: id of measurements
        :param Anchors_Id: id of anchor network
        :param pl_exp: pathloss exponent
        :param n_iter: number of iterations
        :return:
        """
        Anchors = radio_env.get("Network", Anchors_Id)
        Agents = radio_env.get("Network", Agents_Id)
        Measurs = radio_env.get("Measurements", Measurements_Id)
        Anchors_Ids = Anchors.get_RadioNode_Ids()
        Agents_Ids = Agents.get_RadioNode_Ids()

        Maps = radio_env.get("Raster Map", "RSSI")
        Region = Maps.get_Region().get_Region()

        # get Anchor locations
        Anchors_Loc = []
        for id in Anchors_Ids:
            loc = Anchors.get_RadioNode_Loc(id)
            fc_MHz = Anchors.get_RadioNode_FcMHz(id)
            tx_pow = Anchors.get_RadioNode_PowdBm(id)
            Anchors_Loc.append([loc[0], loc[1]])

        Node_Ids = Anchors_Ids + Agents_Ids
        Num_measure = Measurs.len()

        dist_matrix = []
        row = []
        i_row = 0
        for ix in range(0, Num_measure):
            powdB = Measurs.get_by_index(ix).get_Value()
            dist = RSSI_to_dist(powdB, fc_MHz, tx_pow, pl_exp)
            row.append(dist)
            i_row = i_row + 1
            if i_row >= len(Node_Ids):
                dist_matrix.append(row)
                row = []
                i_row = 0
        dist_matrix = np.array(dist_matrix)
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = 0.5 *(dist_matrix + np.transpose(dist_matrix))

        Locs = []
        for x in Anchors_Loc:
            Locs.append([float(x[0]), float(x[1])])

        Locs = np.array(Locs)
        dist_matrix = np.array(dist_matrix)
        Agents_Loc = CoopLoc.ConvexRelax_localization(Locs, dist_matrix, Region, n_iter, False)
        Ids = self.get_RadioNode_Ids()
        ii = 0 #len(Anchors_Ids)
        print Agents_Loc
        for id in Agents_Ids:
            agent = Agents.get_RadioNode(id)
            loc = Agents_Loc[ii]
            tmp = [loc[0, 0], loc[0, 1]]
            agent.set_Loc(tmp)
            ii = ii + 1
        return

    def loc_error(self, ref_network, plothist, D3):
        """
        Esitmates location error between reference network and radio network.

        :param ref_network: reference network
        :param plothist: flag for printng error histogram
        :param D3: flag to estimate error in three dimensions

        :returns: [mean, median, error standard deivation]
        """
        Error = []
        N = 0
        ref_nodes = ref_network.get_RadioNodes()
        for node, node_ref in zip(self.RadioNodes, ref_nodes):
            Loc = node.get_Loc()
            Loc_ref = node_ref.get_Loc()
            Delta_Loc = np.array(Loc, dtype=float) - np.array(Loc_ref, dtype=float)
            Delta_Loc = Delta_Loc * Delta_Loc
            Error.append(Delta_Loc)
            N = N + 1
        Error = np.array(Error)

        dist_Error = np.sqrt(Error[:,0] + Error[:,1])
        if D3:
            dist_Error = np.sqrt(Error[:, 0] + Error[:, 1] + Error[:, 2] )

        if plothist:
            print "\n---------------------------------------------------"
            print "     Mean distance error =   " + str(np.mean(dist_Error))
            print "     Medina distance error = " + str(np.median(dist_Error))
            print "     Std. deviation error =  " + str(np.std(dist_Error))
            print "\n---------------------------------------------------"

            num_bins = 20
            n, bins, patches = pyplot.hist(dist_Error, num_bins, normed=1, facecolor='green', alpha=0.5)
            pyplot.show()
        return [np.mean(dist_Error), np.median(dist_Error), np.std(dist_Error), np.max(dist_Error)]

    def set_Id(self, Id):
        """
        Sets network Id

        :param str Id: network id

        :returns:
        """
        self.Id = Id
        return

    def get_Id(self):
        """
        :returns: network Id
        """
        return self.Id

    def get_RadioNodes(self):
        """
        :returns: list of network Ids
        """
        return self.RadioNodes

    def get_RadioNode_Index(self, NodeId):
        """
        Returns radio node index

        :param NodeId: Id of radio node

        :returns: RadioNode
        """
        idx = -1
        for node in self.RadioNodes:
            idx = idx + 1
            if (NodeId == node.get_Id()):
                return idx
        print ErrorStr + "In Radio Network! No node " + str(NodeId) + " found!"
        return None

    def get_Len(self):
        """
        :returns: number of nodes in radio network
        """
        return len(self.RadioNodes)

    def print_Net(self):
        """
        Prints the radio nodes on console.

        :returns:
        """
        print "   Radio network:"
        for node in self.RadioNodes:
            node.print_node_line()

    def get_RadioNode(self, NodeId):
        """
        Returns RadioNode with NodeId.

        :param NodeId: Id of radion node
        :returns: RadioNode
        """
        for node in self.RadioNodes:
            if (NodeId == node.get_Id()):
                return node
        print ErrorStr + "In Radio Network! No node " + str(NodeId) + " found!"
        return None

    def get_RadioNode_Ids(self):
        """
        Returns list of RadioNode Ids.

        :returns: list of node Ids
        """
        out = []
        for node in self.RadioNodes:
            out.append(node.get_Id())
        return out

    def append_RadioNode(self, radionode):
        """
        Adds the RadioNode Node to the network.

        :param Node: RadioNode

        :returns:
        """
        if len(self.RadioNodes) < 1:
            self.RadioNodes.append(radionode)
        else:
            id = radionode.get_Id()
            NodeList = self.get_RadioNode_Ids()
            if str(id) in NodeList:
                print ErrorStr + " No node added. Node - " + radionode.get_Id() + " - already exists!"
                return
            else:
                self.RadioNodes.append(radionode)
        return

    def set_RadioNode_BwMHz(self, nodeId, Bw):
        """
        Sets radio node bandwidth.

        :param nodeId: node Id
        :param Bw: bandwidth in MHz

        :returns:
        """
        list = self.get_RadioNode_Ids()
        idx = list.index(nodeId)
        x = self.RadioNodes[idx]
        x.set_BwMHz(Bw)
        self.RadioNodes.pop(idx)
        self.RadioNodes.insert(idx, x)
        return

    def set_RadioNode_FcMHz(self, nodeId, Fc):
        """
        Sets radio node carrier frequency.

        :param nodeId: node Id
        :param Fc: node carrie frequency in MHz

        :returns:
        """
        list = self.get_RadioNode_Ids()
        idx = list.index(nodeId)
        x = self.RadioNodes[idx]
        x.set_FcMHz(Fc)
        self.RadioNodes.pop(idx)
        self.RadioNodes.insert(idx, x)
        return

    def set_RadioNode_PowdBm(self, nodeId, Pow):
        """
        Sets radio node Tx/Rx power.

        :param nodeId: node Id
        :param Pow: bandwidth in dBm

        :returns:
        """
        list = self.get_RadioNode_Ids()
        idx = list.index(nodeId)
        x = self.RadioNodes[idx]
        x.set_PowdBm(Pow)
        self.RadioNodes.pop(idx)
        self.RadioNodes.insert(idx, x)
        return

    def get_RadioNode_PowdBm(self, nodeId):
        """
        Returns RadioNode power.

        :param nodeId: node Id

        :returns: Tx/Rx power of node in dBm
        """
        node = self.get_RadioNode(nodeId)
        return node.get_PowdBm()

    def get_RadioNode_FcMHz(self, nodeId):
        """
        Returns RadioNode carrier frequency.

        :param nodeId: node Id

        :returns: carrier frequncy in MHz
        """
        node = self.get_RadioNode(nodeId)
        return node.get_FcMHz()

    def set_RadioNode_Alt(self, nodeId, alt):
        """
        Sets radio node altitide above sea level.

        :param nodeId: node Id
        :param alt: altitude in [m]

        :returns:
        """
        list = self.get_RadioNode_Ids()
        idx = list.index(nodeId)
        x = self.RadioNodes[idx]
        x.set_Alt(alt)
        self.RadioNodes.pop(idx)
        self.RadioNodes.insert(idx, x)
        return

    def get_AreaLimits(self, fract):
        """
        Finds the rectengular area, where nodes are located.

        :param fract: fraction of area to extend display

        :returns:  [west, south, east, north]
        """
        loc = self.RadioNodes[0].get_Loc()
        low_left = loc
        up_right = loc

        for node in self.RadioNodes:
            loc = node.get_Loc()
            low_left = [min(x, y) for (x, y) in zip(low_left, loc)]
            up_right = [max(x, y) for (x, y) in zip(up_right, loc)]

        if fract != 0.0:
            diff = [round((float(x) - float(y))*fract) for (x, y) in zip(up_right, low_left)]
            low_left = [(float(x) - float(y)) for (x, y) in zip(low_left, diff)]
            up_right = [(float(x) + float(y)) for (x, y) in zip(up_right, diff)]
        return [low_left, up_right]

    def plot(self, Marker, FigNum, ShowPlot, delta):
        """
        Plots the radio network on the map background.

        :param Marker: marker
        :param FigNum: figure number
        :param ShowPlot: show plot flag
        :param delta: size of boarder

        :returns:
        """
        fig = pyplot.figure(FigNum)
        ax = fig.add_subplot(111)

        X = []
        Y = []
        Z = []
        NodeName = []
        for node in self.RadioNodes:
            loc = node.get_Loc()
            X.append(float(loc[0]))
            Y.append(float(loc[1]))
            Z.append(float(loc[2]))
            NodeName.append(node.get_Name())
        pyplot.plot(X, Y, Marker)
        pyplot.title(self.Id)

        if delta != 0:
            for x, y, name in zip(X,Y, NodeName):
                ax.annotate(name, xy=(x, y), xytext=(float(x) + float(delta), float(y) + float(delta)),
                        arrowprops=dict(facecolor='blue', shrink=0.05),)

        if ShowPlot:
            pyplot.show()
        return fig

    def plot_compare(self, netw, Marker, FigNum, ShowPlot, delta):
        self.plot(Marker, FigNum, False, delta)
        netw.plot(Marker + "r", FigNum, False, delta)
        Ids = netw.get_RadioNode_Ids()
        for id in Ids:
            loc_self = self.get_RadioNode_Loc(id)
            loc_netw = netw.get_RadioNode_Loc(id)
            x = [float(loc_self[0]), float(loc_netw[0])]
            y = [float(loc_self[1]), float(loc_netw[1])]
            pyplot.plot(x, y, "k--")
        if ShowPlot:
            pyplot.show()

    def plot_connects(self, Marker, FigNum, ShowPlot, delta):
        self.plot(Marker, FigNum, False, delta)
        i = 0
        j = 0
        for node_i in self.RadioNodes:
            j = 0
            loc = node_i.get_Loc()
            x_i = loc[0]
            y_i = loc[1]
            for node_j in self.RadioNodes:
                if self.Connections[i][j] > 0:
                    loc = node_j.get_Loc()
                    pyplot.plot([x_i, loc[0]], [y_i, loc[1]], "-")
                j = j + 1
            i = i + 1
        if ShowPlot:
            pyplot.show()
        return

    def read_fromCsvFile(self, csvFilename):
        """
        Reads newtwork from csv file.

        :param csvFilename:
        :returns:
        """
        import csv

        # read information about node Id, loc, antenna azimuth, tilt and type from RaPlaT csvFile
        try:
            print "   Reading Radio network [", self.Id, "] csv file: ", csvFilename
            csvRows = []
            csvFileObj = open(csvFilename, 'rb')
            readerObj = csv.reader(csvFileObj)
            for row in readerObj:
                if (row[0][0] == "#"):
                    continue  # skip commented rows
                if (row == []):
                    continue  # skip empty rows
                csvRows.append(row)
            csvFileObj.close()
        except ValueError:
            print ErrorStr + "In Radio Network! File ", csvFilename, " does not exists!"
            return

        # rows processing
        fileHeader = csvRows[0]
        del csvRows[0]
        if ('antID' in fileHeader) == False:
            print ErrorStr + "In Radio Network! The first row in file ", csvFilename, " does not include antID!"
            return

        # process radio node rows
        for row in csvRows:
            try:
                idx_Id = fileHeader.index("antID")
                idx_x = fileHeader.index("antEast")
                idx_y = fileHeader.index("antNorth")
                idx_z = fileHeader.index("antHeightAG")
                tmpNode = RadioNode("Id", row[idx_Id], "Loc", row[idx_x], row[idx_y], row[idx_z])
                try:
                    idx = fileHeader.index("cellName")
                    tmpNode.set_Name(row[idx])
                except ValueError:
                    tmpNode.set_Name(None)
                try:
                    idx = fileHeader.index("antType")
                    tmpNode.set_AntType(row[idx])
                except ValueError:
                    tmpNode.set_AntType(None)
                try:
                    idx = fileHeader.index("antDirection")
                    tmpNode.set_AntAzimuth(row[idx])
                except ValueError:
                    tmpNode.set_AntAzimuth(None)
                try:
                    idx = fileHeader.index("antElecTilt")
                    tmp = float(row[idx])
                    idx = fileHeader.index("antMechTilt")
                    tmp = tmp + float(row[idx])
                    tmpNode.set_AntTilt(tmp)
                except ValueError:
                    tmpNode.set_AntTilt(None)
                self.append_RadioNode(tmpNode)
            except ValueError:
                print WarningStr + "No Radio Node added!"
                print row

        # Adding information about power, channels, frequencies, if exists
        file_name, file_ext = os.path.splitext(csvFilename)
        csvFilename = file_name + "chan" + file_ext
        print "   Reading channel information [" +  self.Id + "] csv file: "  + csvFilename
        csvRows = []
        try:
            csvFileObj = open(csvFilename, 'rb')
            readerObj = csv.reader(csvFileObj)
            for row in readerObj:
                if (row[0][0] == "#"):
                    continue  # skip commented rows
                if (row == []):
                    continue  # skip empty rows
                csvRows.append(row)
            csvFileObj.close()
        except ValueError:
            print ErrorStr + "In Radio Network! File " +  csvFilename + " does not exists!"
            return

        fileHeader = csvRows[0]
        del csvRows[0]
        if ('antID' in fileHeader) == False:
            print ErrorStr + "In Radio Network! The first row in file " + csvFilename + " does not include antID!"
            return

        # process radio node rows
        idx_Id = fileHeader.index("antID")
        for row in csvRows:
            try:
                ant_Id = row[idx_Id]
                try:
                    idx = fileHeader.index("bw")
                    x = row[idx]
                    self.set_RadioNode_BwMHz(ant_Id, x)
                except ValueError:
                    pass
                try:
                    idx = fileHeader.index("freq")
                    x = row[idx]
                    self.set_RadioNode_FcMHz(ant_Id, x)
                except ValueError:
                    pass
                try:
                    idx = fileHeader.index("powdBm")
                    x = row[idx]
                    self.set_RadioNode_PowdBm(ant_Id, x)
                except ValueError:
                    pass
                try:
                    idx = fileHeader.index("nmv")
                    x = row[idx]
                    self.set_RadioNode_Alt(ant_Id, x)
                except ValueError:
                    pass

            except ValueError:
                pass
        return

    def get_RadioNode_Loc(self, nodeId):
        """
        Returns node location of node.

        :param nodeId: nodes Id

        :returns: [west-east, south-north, alatitude above ground level]
        """
        node = self.get_RadioNode(nodeId)
        Loc = node.get_Loc()
        return Loc

    def get_Locations(self):
        """
        Returns node locations of nodes in network.

        :param nodeId: nodes Id

        :returns: [west-east, south-north, alatitude above ground level]
        """

        Ids = self.get_RadioNode_Ids()
        loc = []
        for id in Ids:
            tmp = self.get_RadioNode_Loc(id)
            loc.append(tmp)
        return loc

    def set_RadioNode_Loc(self, nodeId, Loc):
        """
        Sets locations of radio nodes.

        :param nodeId: node Id
        :param Loc: node location

        :returns:
        """
        node = self.get_RadioNode(nodeId)
        idx = self.get_RadioNode_Index(nodeId)
        self.RadioNodes.pop(idx)
        node.set_Loc(Loc)
        self.RadioNodes.insert(idx, node)
        #self.RadioNodes.append(node)
        return

    def get_Region_Map(self, margin, delta_west = 1.0, delta_south = 1.0):
        """
        Returns radio map region.

        :param margin: margin beyond the area limits
        :returns: region map [west, south, delta_west, delta_south, cols, rows]
        """
        limits = self.get_AreaLimits(margin)
        west = limits[0][0]
        south = limits[0][1]
        east = limits[1][0]
        north = limits[1][1]
        cols = int((east - west) / delta_west)
        rows = int((north - south) / delta_south)
        Region = Raster_Map.RasterMap("Region", west, south, delta_west, delta_south, cols, rows)
        Region.set_Values(0)
        return Region

    def add_rnd_Loc(self, Region, N_nodes):
        """
        Adds nodes random locations.

        :param Region: region of the map
        :param N_nodes: number of nodes

        :returns:
        """
        margins = np.array(Region.get_Region_WSEN())  # south, west, north, east
        delta = np.array([margins[2] - margins[0], margins[3] - margins[1], 0])
        loc_sw = np.array([margins[0], margins[1], 0])

        for i in range(0, N_nodes):
            loc_rnd = np.random.rand(1, 3) * delta
            loc = loc_sw + loc_rnd
            node = RadioNode("Id", i)
            node.set_Loc(loc[0])
            node.set_Name(str(i))
            self.append_RadioNode(node)
        return

    def est_distance_matrix(self, dim):
        """
        Estimates the distances between nodes
        in 2 or 3 dimensions

        :param dim: 2 for in 2 dimensions, 3 in 3 dimensions

        :return: distance matrix
        """
        i = 0
        N_nodes = len(self.RadioNodes)
        dist = np.empty([N_nodes, N_nodes], dtype=float)
        for node_a in self.RadioNodes:
            loc_a = node_a.get_Loc()
            j = 0
            for node_b in self.RadioNodes:
                loc_b = node_b.get_Loc()
                x = float(loc_a[0]) - float(loc_b[0])
                y = float(loc_a[1]) - float(loc_b[1])
                z = float(loc_a[2]) - float(loc_b[2])
                d = x*x + y*y
                if dim == 3:
                    d = d + z*z
                d = np.sqrt(d)
                dist[i, j]  = d
                j = j + 1
            i = i + 1
        return dist

    def est_normal_matrix(self, dim):
        """
        Estimates the normals vectors connecting  node i and j
        in 2 or 3 dimensions

        :param dim: 2 for in 2 dimensions, 3 in 3 dimensions

        :return: normal matrix
        """

        i = 0
        N_nodes = len(self.RadioNodes)
        normals = np.empty([N_nodes, N_nodes, 3], dtype=float)
        for node_a in self.RadioNodes:
            loc_a = node_a.get_Loc()
            j = 0
            for node_b in self.RadioNodes:
                loc_b = node_b.get_Loc()
                x = float(loc_a[0]) - float(loc_b[0])
                y = float(loc_a[1]) - float(loc_b[1])
                z = float(loc_a[2]) - float(loc_b[2])
                norm = x*x + y*y
                if dim == 3:
                    norm = norm + z*z
                norm = np.sqrt(norm)
                if norm  <= 0:
                    normals[i, j] = np.array([0,0,0])
                else:
                    normals[i, j] = np.array([x/norm, y/norm, z/norm])
                j = j + 1
            i = i + 1
        return normals

    def set_connects(self, connects):
        self.Connections = connects
        return

    def est_connects(self, type, value, dim):
        """
        Method finds connections between nodes in network
        :param type: type of connection max_dist: maximal distance, num_conn: number of connections
        :param value: threshold to find connection
        :param dim: dimensions 2 => 2D, 3 => 3D
        :return:
        """

        dist_matrix = self.est_distance_matrix(dim)
        self.Connections = np.empty(dist_matrix.shape, dtype=float)

        if type == "max_dist":
            for i in range(len(dist_matrix)):
                for j in range(len(dist_matrix[i])):
                    self.Connections[i][j] = dist_matrix[i][j]
                    if dist_matrix[i][j] > value:
                        self.Connections[i][j] = 0.0
            return

        if type == "num_conn":
            for i in range(len(dist_matrix)):
                dist_vec = dist_matrix[i]
                n = dist_vec.size - (value + 1)
                ind = np.argpartition(dist_vec, -n)[-n:]
                dist_vec[ind] = 0.0
                self.Connections[i] = dist_vec
            return

    def get_connects_hist(self):
        """

        :return: connection histogram of network
        """
        node_connections = []
        bins = range(0, self.get_Len())

        for i in range(len(self.Connections)):
            x = self.Connections[i]
            node_connections.append(np.count_nonzero(x))

        a, b = np.histogram(node_connections, bins)
        pyplot.hist(node_connections, bins)
        ind = np.argmax(a)
        print "   Number of nodes: ",  a[ind], ", with ", b[ind], " connections."
        pyplot.show()

        return

    def get_connect(self):
        return self.Connections

    def copy_netw(self, netw_name):
        """
        :param netw_name: network name
        :return: copy of the network
        """
        out_netw = copy.deepcopy(self)
        out_netw.set_Id(netw_name)
        return out_netw

    def add_rnd_dist(self, *arg):
        # generate mask to affect only connections
        mask = self.Connections.copy()
        mask[mask > 0] = 1.0
        shape = self.Connections.shape

        if arg[0] == "Normal":
            mu = arg[1]
            sigma = arg[2]
            noise = np.random.normal(mu, sigma, shape)
        elif arg[0] == "Uniform":
            low = arg[1]
            high = arg[2]
            noise = np.random.uniform(low, high, shape)
        elif arg[0] == "Rayleigh":
            val = abs(arg[1])
            noise = np.random.rayleigh(val, shape)
        elif arg[0] == "LogNormal":
            mu = arg[1]
            sigma = arg[2]
            noise = np.random.lognormal(mu, sigma, shape)
        else:
            print ErrorStr  + "The distribution of noise is not correct!"
            print ErrorStr  + "No noise is added!"
            return
        self.Connections = self.Connections + noise * mask
        return



#        while (error[0] > mean_error) and (iter < iter_max):
#            iter = iter + 1

        return np.array(out)

    def coop_loc_LS(self, n_anchors, mean_error, delta, dim, iter_max):
        print "    Start coop loc LS: ",
        iter = 0
        error = []
        error.append(9999.0)
        out = []
        # self.plot('ro', 1, False, 0)
        netw = self.copy_netw("Working network")
        Ids = netw.get_RadioNode_Ids()
        loc = [500, 500, 0]
        for i in range(n_anchors, len(Ids)):
            netw.set_RadioNode_Loc(Ids[i], loc)

        print "    Error [mean, max]: ",
        while (error[0] > mean_error) and (iter < iter_max):
            iter = iter + 1
            # netw.plot('s', 2, False, 0)
            normals = netw.est_normal_matrix(dim)
            dist_ref = self.est_distance_matrix(dim) * self.get_connect()

            dist = netw.est_distance_matrix(dim) * self.get_connect()
            delta_dist = dist_ref - dist

            node_num = -1
            for d, e in zip(delta_dist, normals):
                node_num = node_num + 1
                if (node_num >= n_anchors):
                    loc = []
                    for x in np.transpose(e):
                        loc.append(np.dot(x, d))
                    loc_old = netw.get_RadioNode_Loc(Ids[node_num])
                    loc_new = []
                    for l,x  in zip(loc_old, loc):
                        loc_new.append(float(l) + delta * x)
                    netw.set_RadioNode_Loc(Ids[node_num], loc_new)

            error = self.loc_error(netw, False, False)
            out.append(error)
            if iter % int(iter_max/5) == 0:
                print_str = [int(error[0]), int(error[3])]
                print print_str,

        # netw.plot('go', 3, True, 0)
        print_str = [int(error[0]), int(error[3]), iter]
        print " \n       [Mean error, Max error, Iter]: ", print_str
        print "    End coop loc LS: "
        for idx in Ids:
            node = netw.get_RadioNode(idx)
            self.set_RadioNode_Loc(idx, node.get_Loc())
        return np.array(out)

if __name__ == '__main__':

    print "\n--------------------------------------------"
    print "|        Node and RadioNetwork class testing        |"
    print "--------------------------------------------\n"

    Rx = RadioNode("Id", 1, "Loc", 12, 13, 0, "BwMHz", 1.0)
    Rx.set_FcMHz(10.0)
    Rx.set_PowdBm(None)
    Rx.print_node()

    path = os.getcwd()
    for i in range(0, 3):
        [path, tail] = os.path.split(path)
    projDir = iniProject.iniDir(path)
    csv_file = "Log-a-Tec_2_JSI.csv"
    csv_file = os.path.join(projDir["csv"], csv_file)

    BaseStations = RadioNetwork("BaseStations")
    BaseStations.read_fromCsvFile(csv_file)

    print BaseStations.get_Id()
    print BaseStations.get_RadioNode_Ids()
    print BaseStations.get_AreaLimits(0.1)
    BaseStations.plot("s", 1, True, 10)


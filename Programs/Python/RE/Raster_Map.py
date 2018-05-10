"""
    Raster_Map
    ==========
    contains:
        - RasterMap class
        - RasterMaps class, a set of raster maps
"""

ErrorStr = "--- ERROR: "
WarningStr = "--- Warning: "


import numpy as np
import matplotlib.pyplot as pyplot


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

class RasterMap(object):
    """
        **Defines raster map object**
    """

    def __init__(self, Id, west, south, delta_west, delta_south, cols, rows):
        self.Id = Id                    # raster map Id
        self.West = west                # raster map region west point
        self.South = south              # raster map region south point
        self.d_West = delta_west        # raster map bin size in east-west direction
        self.d_South = delta_south      # raster map bin size in south-north direction
        self.Cols = cols                # number of oolumns in raster map
        self.Rows = rows                # number of rows in raster map
        X, Y = self.generate_xy_mesh()
        self.Values = np.empty([(self.Rows + 1), (self.Cols + 1)], dtype=float)
        self.Values[:] = np.NaN         # numpy two dimensional array of values in raster map
        return

    def get_Region(self):
        """
        :returns: region [west, south, d_west, d_south, Cols, Rows]
        """
        return self.West, self.South, self.d_West, self.d_South, self.Cols, self.Rows,

    def set_Id(self, MapId):
        """
        Sets map Id.

        :param MapId: map Id

        :returns:
        """
        self.Id = MapId
        return

    def get_Id(self):
        """
        :returns: map Id
        """
        return self.Id

    def get_Region_WSEN(self):
        """
        :returns: region in west, south, east, north format
        """
        West = self.West - self.d_West * 0.5
        South = self.South - self.d_South * 0.5
        East = self.West + self.d_West * (self.Cols + 0.5)
        North = self.South + self.d_South * (self.Rows + 0.5)
        return West, South, East, North

    def xy_to_ColsRows(self, x, y):
        """
        Converts east-west, south-east location to map column and map row.

        :param x: east-west coordinate
        :param y: south-east coordinate

        :returns: column, row
        """
        cols = (x - self.West)/self.d_West
        rows = (y - self.South)/self.d_South
        cols = round(cols)
        rows = round(rows)
        if (cols > self.Cols) or (cols < 0):
            print WarningStr + "X value out of range! " +  str(cols)
            return None, None
        if (rows > self.Rows) or (rows < 0):
            print WarningStr + "Y value out of range! " +  str(rows)
            return None, None
        return cols, rows

    def ColsRows_to_xy(self, cols, rows):
        """
        Converts map column, map row location to east-west, south-east location.

        :param cols: map column
        :param rows: map row

        :returns: east-west, south-east location
        """
        x = self.West + float(cols) * self.d_West
        y = self.South + float(rows) * self.d_South
        if (cols > self.Cols) or (cols < 0):
            print WarningStr + "X value out of range! " +  str(cols)
        if (rows > self.Rows) or (rows < 0):
            print WarningStr + "Y value out of range! " +  str(rows)
        return x, y

    def print_Region(self):
        """
        Prints the region of the map.

        :returns:
        """
        East, South, West, North = self.get_Region_WSEN()
        print "   Raster Map Id: ", self.Id
        print "     Origin:     [" + str(self.West) + ", " + str(self.South) + "]"
        print "     Resolution: [" + str(self.d_West) + ", " + str(self.d_South) + "]"
        print "     Region:     [" + str(West) + ", " + str(South) + "], [" + str(East) + ", " + str(North) + "]"
        return

    def generate_xy_mesh(self):
        """
        Generates the mesh grid of west-east and south north values.

        :returns: X -> west-east, Y -> south - north  array of raster points (x, y)
        """
        X, Y = generate_xy_mesh(self.West, self.South, self.d_West, self.d_South, self.Cols, self.Rows)
        return X, Y

    def set_Values(self, in_values):
        """
        Sets values to raster map.

        :param in_values: np array of values

        :returns:
        """
        # if size of np array == 1, all values in map obtained this values
        try:
            in_values = np.asarray(in_values)
            if in_values.size == 1:
                self.Values[:] = in_values
                return
            else:
                if in_values.shape == self.Values.shape:
                    self.Values = in_values
                else:
                    print ErrorStr + " Check the array size in Raster Map set_Values!"
        except ValueError:
            print ErrorStr + " Check the input in Raster Map set_Values!"
            return

    def get_Values(self):
        """
        :returns: the values of raster map
        """
        return self.Values

    def get_aValue(self, x, y):
        """
        Finds the value of map at [x, y]  location.

        :param x: east-west
        :param y: south-north

        :returns: raster map value
        """
        cols, rows = self.xy_to_ColsRows(x, y)
        if (cols is None) or (rows is None):
            print ErrorStr + "Coordinates out of map! " + str(x) + ",  " + str(y)
            return None
        return self.Values[rows, cols]

    def set_aValue(self, x, y, value):
        """
        Sets a raster map value at coordinate x, y.

        :param x: west-east coordinate
        :param y: south-north coordinate
        :param value: value

        :returns:
        """
        cols, rows = self.xy_to_ColsRows(x, y)
        self.Values[rows, cols] = value
        return

    def copy(self, MapId):
        """
        Copies the map.

        :param MapId: Id of new map

        :returns: new map
        """
        newMap = RasterMap(MapId, self.West, self.South, self.d_West, self.d_South, self.Cols, self.Rows)
        newMap.set_Values(self.Values)
        return newMap

    def plot(self, FigNum, ShowPlot, ColorBar, Region):
        """
        Plots raster map.

        :param FigNum: figure number
        :param ShowPlot: flag to show plot on the screen
        :param ColorBar: flag to show color bar
        :param Region: map region

        :returns:
        """
        fig = pyplot.figure(FigNum)
        ax = fig.gca()
        X, Y = self.generate_xy_mesh()
        X = X - float(Region[0])
        Y = Y - float(Region[1])
        cmap = pyplot.get_cmap('seismic')

        if ColorBar:
            CS = pyplot.pcolormesh(X, Y, self.Values, cmap=cmap)
            pyplot.colorbar(CS)
        else:
            CS = pyplot.pcolormesh(X, Y, self.Values, cmap=cmap, vmin=-1, vmax=1)

        if ShowPlot: pyplot.show()
        return fig, ax

    def plot_Network(self, FigNum, ShowPlot, Marker, MarkerSize, Network, ColorBar, Legend_label, Region):
        """
        Plots the network.

        :param FigNum: figure number
        :param ShowPlot: True/False if we want plot is shown
        :param Marker: marker type ["s", "o", "x", ...]
        :param MarkerSize: size of marker in a plot
        :param Network: network to be plotted
        :param ColorBar: True/False plot color bar
        :param Legend_label: Legend labels at nodes
        :param Region: map region

        :returns:
        """
        X = []
        Y = []
        Node_Ids = Network.get_RadioNode_Ids()
        for Id in Node_Ids:
            Node = Network.get_RadioNode(Id)
            Loc = Node.get_Loc()
            X.append(float(Loc[0]) - Region[0])
            Y.append(float(Loc[1]) - Region[1])
        fig, ax = self.plot(FigNum, False, ColorBar, Region)
        ax.plot(X, Y, Marker, label = Legend_label, markersize=MarkerSize)

        if ShowPlot:
            legend = ax.legend(loc='upper right', numpoints=1)
            pyplot.xlabel('m')
            pyplot.ylabel('m')
            #pyplot.show()
        return fig, ax

    def dist(self, x_point, y_point, val_point):
        """
        Returns map distance from point (x,y,val).

        :param x_point: east-west coordinate
        :param y_point: south-east coordinate
        :param val_point: value

        :returns:
        """
        Xmesh, Ymesh = self.generate_xy_mesh()
        x = Xmesh - x_point
        y = Ymesh - y_point
        z = self.Values - val_point
        dist = x * x + y * y + z * z
        dist = np.sqrt(dist)
        self.Values = dist
        return

    def RSSI_FSPL(self, node):
        """
        Calculates the FSPL map for the transmitter at node.

        :param node: Tx node

        :returns: FSPL map
        """
        map_name = "RSSI_" + str(node.get_Id())
        map_out = self.copy(map_name)
        Loc =  node.get_Loc()
        x_point = float(Loc[0])
        y_point = float(Loc[1])
        z_point = float(Loc[2])
        map_out.dist(x_point, y_point, z_point)
        dist = map_out.get_Values()
        dist[dist < 0.001] = 0.001
        # d[km], freq[GHz] +92.45; d[m], freq[kHz] -87.55;
        # d[m],  freq[MHz] -27.55; d[km], freq[MHz] +32.45;
        FSPL = 20.0 * np.log10(dist) + 20.0 * np.log10(float(node.get_FcMHz())) - 27.55
        FSPL = float(node.get_PowdBm()) - FSPL
        map_out.set_Values(FSPL)
        return map_out

    def add_Random(self, Distribution, Params):
        """
        Adds random number to the map pixel values.

        :param Distribution: distribution ["Normal", "Rayleigh", "LogNormal"]
        :param Params: distrubution parameters

        :returns:
        """
        shape = self.Values.shape
        if Distribution == "Normal":
            Mean = float(Params[0])
            Std = float(Params[1])
            Random = np.random.normal(Mean, Std, shape)
        elif Distribution == "Rayleigh":
            Std = float(Params[0])
            Random = np.random.rayleigh(Std, shape)
        elif Distribution == "LogNormal":
            Mean = float(Params[0])
            Std = float(Params[1])
            Random = np.random.lognormal(Mean, Std, shape)
        else:
            print WarningStr + " Distribution " + Distribution + " not implemented!"
            return
        self.Values = self.Values + Random
        return

    def read_from_xyz_RaPlaT(self, csv_file, delimiter, not_a_number):
        """
        Imports Grass/RaPlaT file na generate raster map.

        :param ascii_file: ascii file from Grass/RaPlaT toolbox
        :param delimiter: delimiter between data
        :param: not_a_number: sign for not a number value

        :returns: raster map
        """
        import csv
        i = 0
        array = []
        with open(csv_file, 'rb') as csvfile:
            data = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
            for row in data:
                iline =+1
                if row[0] == "north:":
                    north = float(row[1])
                elif row[0] == "south:":
                    south = float(row[1])
                elif row[0] == "east:":
                    east = float(row[1])
                elif row[0] == "west:":
                    west = float(row[1])
                elif row[0] == "rows:":
                    self.Rows = float(row[1])
                elif row[0] == "cols:":
                    self.Cols = float(row[1])
                else:
                    row.remove("")
                    array.append(row)
            array = np.array(array)
            array_shape = array.shape
            try:
                vals = array.astype(np.float)
            except ValueError:
                j = 0
                vals = np.empty(array_shape,  dtype=float)
                for row in array:
                    i = 0
                    for x in row:
                        try:
                            vals[j, i] = float(x)
                        except ValueError:
                            vals[j, i] = np.nan
                        i += 1
                    j += 1
        self.d_West = (east - west)/self.Cols
        self.d_South = (north - south)/self.Rows
        self.West = west
        self.South = south
        self.Values = vals
        return

class RasterMaps(object):
    """
    **Defines a set of raster maps.**
    """

    def __init__(self, *args):
        self.Id = args[0]           # Id of raster maps
        self.Net = None             # Network Id related to the ma
        if len(args) > 1:
            self.Net = args[1]
        self.Maps = []              # list of raster maps
        self.Region = RasterMap("Region", 0, 0, 1, 1, 1, 1)     # region map
        return

    def get_Id(self):
        """
        :returns: raster maps Id
        """
        return self.Id

    def get_Net(self):
        """
        :returns: network associated with the map
        """
        return self.Net

    def set_Net(self, network):
        """
        Sets network associated with the maps.

        :param network: network Id

        :returns:
        """
        self.Net = network
        return

    def set_Id(self, Id):
        """
        Sets maps Id.

        :param Id: maps Id

        :returns:
        """
        self.Id = Id
        return

    def get_Region(self):
        """
        Returns a Region map of maps.

        :returns:
        """
        return self.Region

    def set_Region(self, Region):
        """
        Sets a Region map of maps.

        :param Region: region map
        :returns:
        """
        self.Region = Region
        return

    def append_Map(self, Map):
        """
        Adds a map to the maps.

        :param Map: raster map
        :returns:
        """
        region = Map.get_Region()
        if set(region) == set(self.Region.get_Region()):
            self.Maps.append(Map)
        return

    def append_Maps(self, Maps):
        """
        Adds a map to the maps.

        :param Map: raster map
        :returns:
        """
        reg_1 = set(self.get_Region().get_Region())
        reg_2 = set(Maps.get_Region().get_Region())
        if reg_1 == reg_2:
            Ids = Maps.get_Map_Ids()
            for id in Ids:
                map = Maps.get_Map(id)
                self.append_Map(map)
        return


    def get_Map(self, MapId):
        """
        Gets a map with MapId from the maps.

        :param MapId: map id
        :returns:
        """
        for map in self.Maps:
            if( map.get_Id() == MapId):
                return map
        return None

    def get_Map_Ids(self):
        """
        Map Ids         :returns:
        """
        out = []
        for map in self.Maps:
            out.append(map.get_Id())
        return out

    def get_Length(self):
        """
        Returns a number of maps in raster_maps.

        :returns: umber of maps
        """
        return len(self.Maps)

    def get_Values(self, x, y):
        """
        Returns values of all maps in a set at location x, y.

        :param x: east-west location
        :param y: norht-south location

        :returns:
        """
        Values = []
        for map in self.Maps:
            value = map.get_aValue(x, y)
            Values.append(value)
        return Values

    def set_Values(self, x, y, value):
        """
        Sets a value to all maps in a set.

        :param x: east-west location
        :param y: north-south location
        :param value: value

        :returns:
        """
        for map in self.Maps:
            value = map.set_aValue(x, y, value)
        return

    def add_Random(self, Distribution, Params):
        """
        Adds a random noise to the map.

        :param Distribution: distribution
        :param Params: distribution parameters

        :returns:
        """
        for map in self.Maps:
           map.add_Random(Distribution, Params)
        return

    def calc_RSSI_FSPL(self, Region, Txs):
        """
        Calculates received signal level using FSPL channel model.

        :param Region: set region
        :param Txs: set of transmitters

        :returns:
        """
        self.set_Region(Region)
        Tx_Ids = Txs.get_RadioNode_Ids()
        for i in Tx_Ids:
            tmp_map = self.Region.RSSI_FSPL(Txs.get_RadioNode(i))
            self.append_Map(tmp_map)
        return


if __name__ == '__main__':
    print "\n--------------------------------------------"
    print "|        Raster map class testing        |"
    print "--------------------------------------------\n"

    # Region map
    ini_region = [0, 0, 1, 1, 21, 21]
    Region = RasterMap("Region", *ini_region)
    Region.set_Values(0)

    test_map = RasterMap("Test map", *ini_region)
    test_map.set_Values(0)          # set values to 0
    test_map.dist(10, 10, 0)        # calculate value from location (x, y, z)
    test_map.plot(1, True, True, ini_region)

    # Read from GrassRaPlaT file
    RaPlaT_map = RasterMap("Grass RaPLaT map", *ini_region)
    csv_file = "r_out_ascii.dat"
    import misc.iniProject as iniProject
    import os
    projDir = iniProject.iniDir(iniProject.get_proj_dir(3))
    csv_file = os.path.join(projDir["xyz"], csv_file)
    RaPlaT_map.read_from_xyz_RaPlaT(csv_file, " ", "*")
    Reg = RaPlaT_map.get_Region()
    RaPlaT_map.plot(2, True, True, Reg)

    # Test maps class
    RSSI_maps = RasterMaps("RSSI")
    RSSI_maps.set_Region(Region)

    for i in range(0, 10):
        inimap = (i,) + Region.get_Region()
        map_tmp = RasterMap(*inimap)
        map_tmp.set_Values(i)
        RSSI_maps.append_Map(map_tmp)

    map_plot = RSSI_maps.get_Map(7)
    map_plot.add_Random("Normal", [100, 10])
    Reg = map_plot.get_Region()
    print "Map value at (x,y) is: ", map_plot.get_aValue(10,10)
    map_plot.plot(2, True, True, Reg)

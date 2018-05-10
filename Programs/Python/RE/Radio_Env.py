"""
    RE.Radio_Env
    ============
    contains the definition of the radio environment class.
"""

import RE.Raster_Map as Map
import RE.Cooperative_Localization as CoopLoc

ErrorStr = "--- ERROR: "
WarningStr = "--- Warning: "

class RadioEnvironment(object):
    """
    Class stores and process radio environment:
    """
    def __init__(self, Id):
        Region = Map.RasterMap("Region", 0, 0, 1, 1, 1, 1)
        # Region = [west, south, delta_west, delta_north, cols, rows]
        self.Id = Id
        self.Region = Region                # region, observed area
        self.Networks = []                  # list of networks
        self.Network_Ids = []               # list of network Ids
        self.Raster_Maps = []               # list of raster maps
        self.Raster_Map_Ids = []            # list of Raster map Ids
        self.Measurements = []              # list of Measurements
        self.Measurement_Ids = []           # list of Measurement Ids
        return

    def set_Region(self, Region):
        """
        Sets the region of the radio environment.

        :param float, float, float, float, int, int Region: [west, south, delta_west, delta_north, cols, rows]
        :return:

        IJS_Outdoor.set_Region([0.0, 0.0, 1.0, 1.0, 10, 10])

        """
        if type(Region) is list:
            x = Map.RasterMap("Region", Region[0], Region[1], Region[2], Region[3], Region[4], Region[5])
            self.Region = x
        else:
            self.Region = Region
        return

    def get_Region(self):
        """
        :returns: the region of the radio environment [west, south, delta_west, delta_north, cols, rows]

        """
        return self.Region

    def append(self, typ, Id, Element):
        """
        Appends a new radio environment element.

        :param str typ: type of radio environment ["Network", "Raster Map", "Measurements"]
        :param str Id: Id of imported element
        :param radio_network, Element: element to be imported

        :return:

        IJS_Outdoor.append("Network", "Anchors", Anchors)
        """
        if typ == "Network":
            self.Networks.append(Element)
            self.Network_Ids.append(Id)
            return
        elif typ == "Raster Map":
            self.Raster_Maps.append(Element)
            self.Raster_Map_Ids.append(Id)
            return
        elif typ == "Measurements":
            self.Measurements.append(Element)
            self.Measurement_Ids.append(Id)
            return
        else:
            print ErrorStr + "Radio environment type --- " + str(typ) + " --- is not defined!"
        return

    def get_Ids(self, typ):
        """
        Returns a list of radio environment Ids of type typ.

        :param str typ: type of radio environment ["Network", "Raster Map", "Measurements"]
        :returns: a list of radio environment of type typ
        :rtype: [str, str, str, ...]
        """
        if typ == "Network":
            return self.Network_Ids
        elif typ == "Raster Map":
            return self.Raster_Map_Ids
        elif typ == "Measurements":
            return self.Measurement_Ids
        else:
            print ErrorStr + "Radio environment type --- " + str(typ) + " --- is not defined!"
        return

    def get(self, typ, Id):
        """
        Returns a radio environment element.

        :param str typ: type of radio environment
        :param str Id: Id of radio environment

        :returns: radio environment element
        :rtype: network, raster_maps, measurements

        IJS_Outdoor.get("Network", "Anchors")
        """
        if typ == "Network":
            idx = self.Network_Ids.index(Id)
            return self.Networks[idx]
        elif typ == "Raster Map":
            idx = self.Raster_Map_Ids.index(Id)
            return self.Raster_Maps[idx]
        elif typ == "Measurements":
            idx = self.Measurement_Ids.index(Id)
            return self.Measurements[idx]
        else:
            print ErrorStr + "Radio environment type --- " + str(typ) + " --- is not defined!"
        return

    def delete(self, typ, Id):
        """
        Deletes the radio element from the radio environment.

        :param str typ: type of radio environment
        :param str Id: Id of radio environment

        :returns:
        """
        if typ == "Network":
            idx = self.Network_Ids.index(Id)
            self.Networks.pop(idx)
            self.Network_Ids.pop(idx)
            return
        elif typ == "Raster Map":
            idx = self.Raster_Maps.index(Id)
            self.Raster_Maps.pop(idx)
            self.Raster_Map_Ids.pop(idx)
            return
        elif typ == "Measurements":
            idx = self.Measurement_Ids.index(Id)
            self.Measurements.pop(idx)
            self.Measurement_Ids.pop(idx)
            return
        else:
            print ErrorStr + "Radio environment type --- " + str(typ) + " --- is not defined!"
        return

    def replace(self, typ, Id, element):
        """
        Replaces the element in radio environment.

        :param str typ: element type
        :param str Id: element Id
        :param element: new element

        :returns:
        """
        try:
            self.delete(typ, Id)
            self.append(typ, Id, element)
        except ValueError:
            self.append(typ, Id, element)
        return

    def experiment_est_loc_LS(self, pl_exp, ref_node_index, hist_plot):
        """
        Describe experiment in radio environment:

        Estimates the agent locations using least square method.

        :param float pl_exp: path loss exponent to convert RSSI to distance
        :param int ref_node_index: index of the reference node
        :param bool hist_plot: flag for plotting histogram of errors

        :returns:

        """
        print "\n   Least square localization method for a node location:"
        Anchors_Id = "Anchors"
        Measurements_Id = "RSSI"
        agents = self.get("Network", "Agents")
        agents.est_loc_LS(self, Measurements_Id, Anchors_Id, pl_exp, ref_node_index)
        ref_network = self.get("Network", "refAgents")
        agents.loc_error(ref_network, hist_plot, False)
        return

    def experiment_est_loc_FP(self, pl_exp, ref_node_index, hist_plot):
        """

        Describes experiment in radio environment:

        estimates the agent locations based using finger print method

        :param float pl_exp: path loss exponent to convert RSSI to distance
        :param int ref_node_index: index of the reference node
        :param bool hist_plot: flag for plotting histogram of errors

        :returns:
        """
        print "\n   Finger print localization method for a node location:"
        Anchors_Id = "Anchors"
        Measurements_Id = "RSSI"
        agents = self.get("Network", "Agents")
        agents.est_loc_FP(self, Measurements_Id, Anchors_Id, pl_exp, ref_node_index)
        ref_network = self.get("Network", "refAgents")
        agents.loc_error(ref_network, hist_plot, False)
        return

    def experiment_est_loc_BP(self, pl_exp, ref_node_index, hist_plot, n_iter):
        """

        Describes experiment in radio environment:

        estimates the agent locations based using believe propagation method

        :param float pl_exp: path loss exponent to convert RSSI to distance
        :param int ref_node_index: index of the reference node
        :param bool hist_plot: flag for plotting histogram of errors

        :returns:
        """
        print "\n   Believe propagation localization method for a node location:"
        Anchors_Id = "Anchors"
        Agents_Id = "Agents"
        Measurements_Id = "RSSI"
        agents = self.get("Network", Agents_Id)

        agents.est_loc_BP(self, Measurements_Id, Anchors_Id, Agents_Id, pl_exp, n_iter)

        ref_network = self.get("Network", "refAgents")

        agents.loc_error(ref_network, hist_plot, False)
        return




if __name__ == '__main__':
    print "Test radio environment"
    IJS_Outdoor = RadioEnvironment("IJS Outdoor")
    Region = [0.0, 0.0, 1.0, 1.0, 10, 10]
    IJS_Outdoor.set_Region(Region)
    print IJS_Outdoor.get_Region()
    print "Test finished"
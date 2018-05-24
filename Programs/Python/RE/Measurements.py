"""
    Measurement
    ===========
    contains the definition of measurement classes:
    - measurement
    - trace - a set of measurements
"""

import numpy as np
import misc.iniProject as iniProject
import misc.GIS as GIS

from RE.Radio_Net import to_num as to_num
from RE.Radio_Net import to_str as to_str

ErrorStr = "--- ERROR: "
WarningStr = "--- Warning: "

class Measurement(object):
    """
        Measurement class stores and process measurements
    """
    def __init__(self, Id):
        self.Id = Id
        self.Unit = None                    # unit of measurements
        self.Type = None                    # type of measurements
        self.Value = None                   # numerical value of measurement
        self.Tx_Node_Id = None              # Id of the Tx node
        self.Tx_Network_Id = None           # Id of the Tx network
        self.Rx_Node_Id = None              # Id of the Rx node / define location
        self.Rx_Network_Id = None           # Id of the Rx network / define location
        self.TimeStamp = None               # time stamp of measurement Unix epoche time
        return

    def get_Id(self):
        """
        :returns: Id of measurement
        """
        return self.Id

    def set_Id(self, Id):
        """
        Sets Id of measurement.

        :param Id: measurement Id
        :returns:
        """
        self.Id = Id
        return

    def get_Unit(self):
        """
        :returns: unit of measurement
        """
        return self.Unit

    def set_Unit(self, x):
        """
        Sets unit of measurement.

        :param x:
        :returns:
        """
        if x in iniProject.Data_Units:
            self.Unit = x
        else:
            print ErrorStr + x + " unit not allowed!"
        return

    def get_Type(self):
        """
        :returns: type of measurement
        """
        return self.Type

    def set_Type(self, x):
        """
        Sets type of measurement.

        :param x: type of measurement

        :returns:
        """
        if x in iniProject.Data_Types:
            self.Type = x
        else:
            print ErrorStr + x + " type not allowed!"
        return

    def get_Value(self):
        """
        :returns: value of measurement
        """
        return self.Value

    def set_Value(self, x):
        """
        Sets value of measurement.

        :param x: value of measurement

        :returns:
        """
        self.Value = x
        return

    def get_Rx_Network_Id(self):
        """
        :return id of rx network:
        """
        return self.Rx_Network_Id

    def set_Rx_Network_Id(self, x):
        """
        Sets id of Rx network.

        :param x: Rx network id
        :returns:
        """
        self.Rx_Network_Id = x
        return

    def get_Tx_Network_Id(self):
        """
        :returns: id of Tx network
        """
        return self.Tx_Network_Id

    def set_Tx_Network_Id(self, x):
        """
        Sets od of Tx network.

        :param x: Tx network id
        :returns:
        """
        self.Tx_Network_Id = x
        return

    def get_Rx_Node_Id(self):
        """
        :returns: Rx node Id
        """
        return self.Rx_Node_Id

    def set_Rx_Node_Id(self, x):
        """
        Sets Rx node Id.

        :param x: Rx node Id
        :returns:
        """
        self.Rx_Node_Id = x
        return

    def get_Tx_Node_Id(self):
        """
        :returns: Tx node Id
        """
        return self.Tx_Node_Id

    def set_Tx_Node_Id(self, x):
        """
        Sets Tx node Id.

        :param x: Rx node Id

        :returns:
        """
        self.Tx_Node_Id = x
        return

    def get_TimeStamp(self):
        """
        :returns: returns the time stamp of measurement
        """
        return self.TimeStamp

    def set_TimeStamp(self, x):
        """
        Setd time stamp of measurement.

        :param x: time stamp of measurement
        :returns:
        """
        self.TimeStamp = x
        return

    def print_Measure(self):
        """
        Prints the measurement on the console.

        :returns:
        """
        print_str = str(self.Id) + "\t"
        print_str = print_str + str(self.Value) + "\t"
        print_str = print_str + str(self.Unit) + "\t"
        print_str = print_str + str(self.Type) + "\t"
        print_str = print_str + str(self.TimeStamp) + "\t"
        print_str = print_str + str(self.Rx_Node_Id) + "\t"
        print_str = print_str + str(self.Tx_Node_Id) + "\t"
        print print_str
        return


class Trace(object):
    """
    **Class defines a set of measurement.**
    """
    def __init__(self, Id):
        self.Id = Id
        self.Measures = []
        return

    def set_Id(self, Id):
        """
        Sets trace id.

        :param Id: trace id
        :returns:
        """
        self.Id = Id
        return

    def get_Id(self):
        """
        :returns: trace id
        """
        return self.Id

    def len(self):
        """
        :returns: number of measurements
        """
        return len(self.Measures)

    def append(self, x):
        """
        Appends measurement.

        :param x: measurement in form of dictionary:

        :returns:
        """
        #{"Id": "1", "Unit": "m", "Type": "Dist",
        # "Value": 10.0, "Tx_Node_Id": "BS1", "Rx_Node_Id": "Agent 1",
        #"Tx_Network_Id": "Anchors", "Tx_Network_Id": "Agents", "TimeStamp", "To be defined"}

        try:
            new_itm = Measurement(x["Id"])
        except KeyError:
            print ErrorStr + " No measurement Id specified!"
            return
        try:
            y = x["Loc"]
            new_itm.set_Loc(y)
        except KeyError:
            pass
        try:
            new_itm.set_Unit(x["Unit"])
        except KeyError:
            pass
        try:
            new_itm.set_Type(x["Type"])
        except KeyError:
            pass
        try:
            new_itm.set_Value(x["Value"])
        except KeyError:
            pass
        try:
            new_itm.set_Tx_Node_Id(x["Tx_Node_Id"])
        except KeyError:
            pass
        try:
            new_itm.set_Rx_Node_Id(x["Rx_Node_Id"])
        except KeyError:
            pass
        try:
            new_itm.set_Tx_Network_Id(x["Tx_Network_Id"])
        except KeyError:
            pass
        try:
            new_itm.set_Rx_Network_Id(x["Rx_Network_Id"])
        except KeyError:
            pass
        try:
            new_itm.set_TimeStamp(x["TimeStamp"])
        except KeyError:
            pass
        self.Measures.append(new_itm)
        return

    def remove_measure(self, measure_Id):
        """
        Removes measurement from trace.

        :param measure_Id: measurement id

        :returns:
        """
        index = self.get_measure_index(measure_Id)
        del self.Measures[index]
        return

    def replace_measure(self, measure_Id, x):
        """
        Replaces measurement in trace.

        :param measure_Id: measurement id
        :param x: new measurement

        :returns:
        """
        self.remove_measure(measure_Id)
        x["Id"] = str(measure_Id)
        self.append(x)
        return

    def get_measure(self, Id):
        for x in self.Measures:
            if (x.get_Id() == str(Id)):
                y  = {"Id": x.get_Id(),
                        "Unit": x.get_Unit(),
                        "Type": x.get_Type(),
                        "Value": x.get_Value(),
                        "Tx_Node_Id": x.get_Tx_Node_Id(),
                        "Rx_Node_Id": x.get_Rx_Node_Id(),
                        "Tx_Network_Id": x.get_Tx_Network_Id(),
                        "Rx_Network_Id": x.get_Rx_Network_Id(),
                        "TimeStamp": x.get_TimeStamp()
                      }
                return y
        return None

    def set_measure_value(self, measure_Id, value):
        """
        Replaces the value of measurements.

        :param measure_id: measurement id
        :param value: new value

        :returns:
        """
        y = self.get_measure(measure_Id)
        y["Value"] = value
        self.replace_measure(measure_Id, y)
        return

    def get_measure_index(self, Id):
        """
        :param id: measurement Id
        :returns: measurement index
        """
        ii = range(0, len(self.Measures))
        for x, i in zip(self.Measures, ii):
            if (x.get_Id() == str(Id)):
                return i
        return None

    def print_Trace(self):
        """
        Prints trace on the console.

        :returns:
        """
        print "Trace Id: " + self.Id
        for x in self.Measures:
            x.print_Measure()

    def get(self, tx_net_Id, tx_node_Id, rx_net_Id, rx_node_Id):
        """
        Gets all measurements which corresponds to following parameters:

        :param tx_net_Id: tx network id
        :param tx_node_Id: tx node id
        :param rx_net_Id: rx network id
        :param rx_node_Id: rx node id

        :returns: list of measurements
        """
        out = []
        for x in self.Measures:
            if tx_net_Id == x.get_Tx_Network_Id() and tx_node_Id == x.get_Tx_Node_Id() and \
               rx_net_Id == x.get_Rx_Network_Id() and rx_node_Id == x.get_Rx_Node_Id():
                out.append(x)
        return out

    def get_by_index(self, idx):
        return self.Measures[idx]

    def set_values_from_maps(self, Maps, Locs, Rx_Ids, Tx_Ids, x):
        """
        Sets measurement value from the raster maps.

        :param Maps: list of raster maps
        :param Locs: list of locations
        :param Rx_Ids: list of Rx Ids
        :param Tx_Ids: listo of Tx Ids
        :param x: measurement

        :returns:
        """
        for rx, loc in zip(Rx_Ids, Locs):
            vals = Maps.get_Values(float(loc[0]), float(loc[1]))
            for val, tx in zip(vals, Tx_Ids):
                x["Id"] = str(tx) + "_" + str(rx)
                x["Value"] = val
                x["Tx_Node_Id"] = tx
                x["Rx_Node_Id"] = rx
                self.append(x)
        return

    def add_error(self, *args):
        """
        add error to the measurement
        :param args: type of error
        - Rayleigh: add a Rayleigh fading
        - LogNormal: add a LogNormal fading, args[1] = mean, sigma
        - Rice: add a Rice fading with the with K factor ) args[2]
        :returns:
        """
        Vals = []
        Ids = []
        for x in self.Measures:
            if args[0] == "Rayleigh":
                error = 10.0*np.log10(np.random.rayleigh(1, 1))
            if args[0] == "LogNormal":
                error = 10.0*np.log10(np.random.lognormal(args[1], args[2], 1))
            if args[0] == "Rice":
                error = 10.0*np.log10(np.random.rayleigh(args[1], 1))

            Vals.append(x.get_Value() + error)
            Ids.append(x.get_Id())

        for val, id in zip(Vals, Ids):
            self.set_measure_value(id, val)
        return

    def quantize_vals(self, quant):
        """
        Quantizes the measurement.

        :param quant: quantizaion of measurement results
        :returns:
        """
        Vals = []
        Ids = []

        for x in self.Measures:
            z = x.get_Value()
            y = round(z / quant) * (quant)
            Vals.append(y)
            Ids.append(x.get_Id())

        for val, id in zip(Vals, Ids):
            self.set_measure_value(id, val)
        return


if __name__ == '__main__':
    print "\n*******************************"
    print "*         Measurement Test    *"
    print "*******************************\n"

    Trace = Trace("trace_N0")
    for i in range(0, 5):
        x = {"Id": str(i+10),
             "Unit": "m",
             "Type": "Dist",
             "Value": i*10.0,
             "Tx_Node_Id": "BS1",
             "Rx_Node_Id": "Agent 1"}
        Trace.append(x)
    Trace.print_Trace()

    Trace.add_error("Rayleigh")
    Trace.print_Trace()

    Trace.quantize_vals(2.0)
    Trace.print_Trace()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    misc.JSON
    =========
    package contains routines for JSON file.
"""

import json
import re
import os
import misc.iniProject as iniProject
import GIS as GIS

def json_to_RaPlaTcsv(infile, outfile, radius, chan_param):
    """
    Reads json file and coverts it to the csv files for RaPlaT.
    
    :param str infile: input json file
    :param str outfile: csv file names
    :param str radius: calculation radius
    :param str chan_param: channel parameters RaPlaT format
    
    :returns: antenna Id
    :rtype: str
    """

    data = json.loads(open(infile).read())
    # jhslhH  HKASDKFHKASHD 
    Network = data["nodes"]
    HeaderLine = "cellName,antID,antType,antEast,antNorth,antHeightAG,"
    HeaderLine = HeaderLine + "antDirection,antElecTilt,antMechTilt,freq,power,radius,"
    HeaderLine = HeaderLine + "model,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11"
    HeaderLine_chan = "antID,powdBm,chNum,freq,bw,nmv,nodetype,comtype,antPolar"
    filename, ext = os.path.splitext(outfile)
    outfile_chan = filename + "chan" + ext

    file_RaPlaT = open(outfile, "w")
    file_RaPlaT.write(HeaderLine + "\n")
    file_RaPlaT_chan = open(outfile_chan, "w")
    file_RaPlaT_chan.write(HeaderLine_chan + "\n")

    ant_ID = 0
    for node in Network:
        ant_ID += 1

        name = node["name"]
        # Remove all non-word characters (everything except numbers and letters)
        name = re.sub(r"[^\w\s]", '', name)
        # Replace all runs of whitespace with a single dash
        name = re.sub(r"\s+", '_', name)

        tmp = node["position"]
        pos = tmp["GK5"]
        x = int(round(pos["y"]))  - 5000000
        y = int(round(pos["x"])) - 5000000
        alt = tmp["altitude"]

        tmp = node["experimental_network"]
        pow_dBm = tmp["output_pwr_dbm"]
        freq_MHz = tmp["frequency"]
        ant_type = tmp["antenna_type"]
        strRaPlaT = name + "," + str(ant_ID) + "," + str(ant_type) + ","
        strRaPlaT = strRaPlaT + str(x) + "," + str(y) + "," + str(alt) + ","
        strRaPlaT = strRaPlaT + str(0) + "," + str(0) + "," + str(0) + ","
        strRaPlaT = strRaPlaT + str(freq_MHz) + "," + str(pow_dBm) + "," + str(radius) + ","
        strRaPlaT = strRaPlaT + chan_param

        strRaPlaT_chan = str(ant_ID) + "," + str(pow_dBm) + "," + str(0) + "," + str(freq_MHz) + ","
        strRaPlaT_chan = strRaPlaT_chan + "," + str("") + "," + str("") + "," + str("") + "," + str("") + "," + str("")

        file_RaPlaT.write(strRaPlaT + "\n")
        file_RaPlaT_chan.write(strRaPlaT_chan + "\n")
    file_RaPlaT.close()
    file_RaPlaT_chan.close()
    return ant_ID


def json_show_Google_Maps(infile, map_title, html_file, map_zoom, map_type, marker):
    """
    Reads radio network form JSON file and generates html file with nodes location.

    :param str infile: json file with network description
    :param str map_title: title of the map
    :param str html_file: html file name
    :param str map_zoom: google maps zoom parameter
    :param str map_type: type of the map: roadmap, satellite, hybrid, terrain
    :param str marker: type of marker: o, s, x, a, f; colour of marker: r, g, b, k

    :returns:
    """
    data = json.loads(open(infile).read())
    Network = data["nodes"]

    ant_ID = 0
    for node in Network:
        ant_ID += 1

        name = node["name"]
        # Remove all non-word characters (everything except numbers and letters)
        name = re.sub(r"[^\w\s]", '', name)
        # Replace all runs of whitespace with a single dash
        name = re.sub(r"\s+", '_', name)

        tmp = node["position"]
        pos = tmp["GPS"]
        lat = pos["lat"]
        lng = pos["lng"]
        alt = tmp["altitude"]

        if ant_ID == 1:
            map = GIS.google_map(map_title, html_file, [lat, lng], map_zoom, map_type)
        map.add_node(name, [lat, lng], [name], marker)

    return


if __name__ == '__main__':
    print "\n***********************************"
    print "*  JSON read/parse files test     *"
    print "*************************************\n"

    path = os.getcwd()
    for i in range(0, 3):
        [path, tail] = os.path.split(path)
    projDir = iniProject.iniDir(path)

    infile = "Log-a-Tec_2_JSI.json"
    outfile = "Log-a-Tec_2_JSI.csv"

    infile = os.path.join(projDir["json"], infile)
    outfile = os.path.join(projDir["csv"], outfile)

    json_show_Google_Maps(infile, "LOG-a-TEC-IJS", "bla.html", 20, "road", "x")

    radius = "10"
    chan_param = "ITUR1546-4,rural,50,,,,,,,,,"

    num_cells = json_to_RaPlaTcsv(infile, outfile, radius, chan_param)
    print "Number of cells added in file: ", num_cells



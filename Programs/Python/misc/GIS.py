#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    misc.GIS
    =========
    package contains GIS functions.
"""

import re
import math
import os
import misc.iniProject as iniProject
import matplotlib.pyplot as pyplot
import numpy as np
import RE.Radio_Net
ErrorStr = "--- ERROR: "


def dms2dec(dms_str):
    """
    Returns decimal representation of DMS.

    :param str dms_str: string in degrees minutes second format

    :returns: value of dms_str in decimal format
    :rtype: float
    """
    # dms2dec(utf8(48°53'10.18"N))  -->  48.8866111111
    # dms2dec(utf8(2°20'35.09"E))   -->   2.34330555556
    # dms2dec(utf8(48°53'10.18"S))  --> -48.8866111111
    # dms2dec(utf8(2°20'35.09"W))   -->  -2.34330555556
    # dms2dec(012deg20min35.09secW)   --> -12.3430805556

    if (type(dms_str) is str) is False:
        print "Error in dms2dec. Input type must be string"
        return
    if len(dms_str) == 4:
        dms_str = dms_str + '0'
        print "Warning. Unusual string length of DMS string. Adding zero at the end: ", dms_str
    dms_str = re.sub(r'\s', '', dms_str)
    if re.search('[swSW]', dms_str):
        sign = -1
    else:
        sign = 1

    numbers = filter(len,
                     re.split('\D+', dms_str, maxsplit=4))  # Use filter function to remove empty strings from result
    if len(numbers) > 1:
        degree = numbers[0]
        minute = numbers[1] if len(numbers) >= 2 else '0'
        second = numbers[2] if len(numbers) >= 3 else '0'
        if len(minute) > 2:
            second = minute[2:]
            minute = minute[:2]
    else:
        strtmp = numbers[0]
        second = strtmp[-2:]
        minute = strtmp[-4:-2]
        degree = strtmp[0:-4]
    if len(numbers) >= 4:
        frac_seconds = numbers[3]
        lenfrac = len(frac_seconds)
        frac_seconds = str(float(frac_seconds)/pow(10.0,lenfrac))
    else:
        frac_seconds = '0'
    return sign * (float(int(degree)) + float(minute) / 60.0 + float(second) / 3600.0 + float(frac_seconds) / 3600.0)


def dec2dms(indeg):
    """
    Returns value written in decimal format in degrees, minutes, seconds format

    :param float indeg: value in decimal format
    :returns: value in degrees, minutes, seconds format
    :rtype: str
    """

    if (type(indeg) is str) is True:
        if re.search('[jzJZswSW]', indeg):
            indeg = indeg[0:-1]
        indeg = float(indeg)
    if abs(indeg) > 360.0:
        print " Error in dec2dms. Input should be between [-360, 360]."
        return
    deg = int(indeg)
    fmin = (indeg - deg) * 60.0
    imin = int(fmin)
    fsec = (fmin - imin) * 60.0
    fsec = round(fsec*1000.0)/1000
    return str(deg) + 'deg' + str(imin) + 'min' + str(fsec) + 's'


def WGS84toGK(in1, in2):
    """
    Converts latitude, longitude in WGS84 format to Gauss Kreuger coordinates

    :param str in1: latitude in dec format
    :param str in2: longitude in dec format

    :returns: Gauss Kreuger east-west, south-north coordinates
    :rtype: [float, float]

    x,y = WGS84toGK(14.50, 46.50)
    """

    # old LagLong SI system is shifted by 0.28min West and 0.01min South
    # old GK SI system is shifted by 360m West and 7m South
    # 1s NS cca 32 m
    # 1s WE cca 21 m

    switchLongLat = 0
    if (type(in1) is str) is True:
        if re.search('[eEvV]', in1):
            in1 = in1[0:-1]
        if re.search('[nNsS]', in1):
            in1 = in1[0:-1]
            switchLongLat = True
        Long = float(in1)
        if (type(in2) is str) is True:
            if re.search('[nNsS]', in2):
                in2 = in2[0:-1]
            if re.search('[eEvV]', in2):
                in2 = in2[0:-1]
                switchLongLat = True
            Lat = float(in2)
        else:
            print "Error! Longitude na Latitude must be strings!"
        if switchLongLat is True:
            tmp = Long
            Long = Lat
            Lat = tmp
    else:
        Long = in1
        Lat = in2

    a1 = 0.06719697
    b1 = -11.8802
    c1 = -1395.3306
    d1 = 77453.91
    b3 = 0.002563441
    c3 = 0.002563441
    d3 = -0.1310713
    c0 = 111138.9306
    d0 = 5095568.458
    b2 = -0.29620698
    c2 = -0.56423733
    d2 = 486.21103
    c4 = -0.00133149
    d4 = 0.0235618
    b0 = 9.736719

    COR = 0.9999
    ORG = 5500000

    # Correction according to wikipedia
    Lat = Lat + 0.01801 / 60.0  # Correction for Slovenia
    Long = Long + 0.28300 / 60.0  # Correction for Slovenia

    l = Long - 15.0
    f = Lat - 46.0

    x = (a1 * math.pow(f, 3.0) + b1 * math.pow(f, 2.0) + c1 * f + d1) * l + (b3 * math.pow(f, 2.0) + c3 * f + d3) * math.pow(l, 3.0)
    x = x * COR + ORG

    y = b0 * math.pow(f, 2.0) + c0 * f + d0 + (b2 * math.pow(f, 2.0) + c2 * f + d2) * math.pow(l, 2.0) + (c4 * f + d4) * math.pow(l, 4.0)
    y = y * COR

    x = x - 5.0e6
    y = y - 5.0e6
    return [round(x), round(y)]


def GKtoWGS84(x, y):
    """
    Converts Gauss Kreuger EW/NS coordinate to latitude and longitude in WGS84 dec format

    :param float x: east-west value in Gauss Kreuger
    :param float y: south-north value in Gauss Kreuger

    :returns: [latitude, longitude]
    :rtype: [str, str]

    lat, lng = GKtoWGS84(x, y)
    """

    if (type(x) is str) is True:
        x = float(x)
    if (type(y) is str) is True:
        y = float(y)

    A1 = 6366742.52045
    A2 = 15988.63816
    A4 = 16.72994
    A6 = 0.02178
    C = 6398786.84764
    E2 = 0.0067192186624
    E = 0.00500787553
    F = 0.005829637
    G = 0.0080915

    Lambda0 = 15
    FaktMod = 0.9999
    ORDCONST = 5.5E6
    x = x + 5e6
    y = y + 5e6

    y = y / FaktMod
    x = (x - ORDCONST) / FaktMod
    omega = y / A1
    tmp = math.cos(omega)
    tmp = tmp * tmp
    temp = 1 + (1 + G * tmp) * F * tmp
    fx = omega + E / 2 * math.sin(2 * omega) * temp
    tmp = math.cos(fx)
    vx = math.sqrt(1 + E2 * tmp * tmp)
    temp = x / C
    l = math.atan(vx * (math.exp(temp) - math.exp(-temp)) / tmp / 2.0)
    Lat = l * 180 / math.pi + Lambda0
    Long = 180 / math.pi * (math.atan(math.tan(fx) * math.cos(vx * l)))

    Long = Long - 0.01801 / 60.0  # Correction for Slovenia
    Lat = Lat - 0.28300 / 60.  # Correction for Slovenia

    return [str(Lat)+"N", str(Long)+"E"]


def is_LatLong(s):
    """
    Returns float value of s: positive for N, E; negative for W, S

    :param str s: latitude/longitude in dec format

    :returns: float(s): positive for N, E; negative for W, S
    :rtype: float

    """
    try:
        float(s)
        return float(s)
    except ValueError:
        if "N" in s:
            f = s.replace("N", "", 1)
            return float(f)
        elif "S" in s:
            f = s.replace("S", "", 1)
            return -float(f)
        elif "E" in s:
            f = s.replace("E", "", 1)
            return float(f)
        elif "W" in s:
            f = s.replace("W", "", 1)
            return -float(f)
        else:
            return "NaN"


class google_map(object):
    """
    Google maps object.

    Creates a google_map object:

    :param str title: title of page
    :param str html_file: filename of html_file
    :param [float, float] map_center: center of map
    :param float map_zoom: zoom of map
    :param str map_type: map type: "roadmap", "satellite", "hybrid,", "terrain"

    :returns: google_map object

    html_map = google_map("LOG.a.TEC", plot_html_file, [46.0422, 14.4885], 20, "road")

    """

    def __init__(self, title, html_file, map_center, map_zoom, map_type):
        self.title = title
        self.html_file = html_file
        self.map_center = map_center
        self.map_zoom = map_zoom
        self.map_type = map_type
        self.node_Ids = []
        self.html = []

        path = os.getcwd()
        for i in range(0, 3):
            [path, tail] = os.path.split(path)

        projDir = iniProject.iniDir(path)
        html_template_file = os.path.join(projDir["template"], "google_map_template.html")
        with open(html_template_file, "r") as myfile:
            self.html = myfile.readlines()

        if (str(map_type) in iniProject.Google_Map_Types):
            pass
        else:
            map_type = "roadmap"

        map_type = "'" + str(map_type) + "'"
        out_html_string = []
        for x in self.html:
            y = x.replace("$map_center$", "lat:" + str(map_center[0]) + ", lng:" + str(map_center[1]))
            x = y.replace("$map_zoom$", str(map_zoom))
            y = x.replace("$map_type$", str(map_type))
            x = y.replace("$title$", str(self.title))
            out_html_string.append(x)

        self.html = []
        for x in out_html_string:
            self.html.append(x)

        with open(self.html_file, "w") as myfile:
            myfile.writelines(self.html)
        return

    def add_node(self, node_Id, position, info_str, marker):
        """
        Adds new location (node) on the google map

        :param str node_Id: Id of node
        :param [float, float] position: [latitude, longitude]  in dms format
        :param [str, str, ] info_str: display at clicking on the node
        :param str marker: node marker: [x,s,o,f,a][r,g,b,k], "xg"

        :returns:

        html_map.add_node(4, [46.0423773, 14.4882363], ["Node 4", "text 4", "text 4"], "ak")
        """
        node_Id = str(node_Id)
        if str(node_Id) in self.node_Ids:
            print ErrorStr + " node with node_Id = " + node_Id + " exists! No node is added in html file"
            return
        else:
            self. node_Ids.append(node_Id)

        next_node_str = "/*** next node ***/"
        position_str = "{lat:" + str(position[0]) + ", lng:" + str(position[1]) + "}"

        # processing marker
        marker_image = "'https://developers.google.com/maps/documentation/javascript/examples/full/images/beachflag.png';"
        if marker == 'o':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/o_Icon.png';"
        elif marker == 's':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/s_Icon.png';"
        elif marker == 'x':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/x_Icon.png';"
        elif marker == "a":
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/antenna_tower_Icon.png';"
        elif marker == 'f':
            marker_image = "'https://developers.google.com/maps/documentation/javascript/examples/full/images/beachflag.png';"

        elif marker == 'or':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/R_o_Icon.png';"
        elif marker == 'sr':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/R_s_Icon.png';"
        elif marker == 'xr':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/R_x_Icon.png';"
        elif marker == "ar":
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/R_antenna_tower_Icon.png';"

        elif marker == 'og':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/G_o_Icon.png';"
        elif marker == 'sg':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/G_s_Icon.png';"
        elif marker == 'xg':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/G_x_Icon.png';"
        elif marker == "ag":
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/G_antenna_tower_Icon.png';"

        elif marker == 'ob':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/B_o_Icon.png';"
        elif marker == 'sb':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/B_s_Icon.png';"
        elif marker == 'xb':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/B_x_Icon.png';"
        elif marker == "ab":
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/B_antenna_tower_Icon.png';"

        elif marker == 'ok':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/K_o_Icon.png';"
        elif marker == 'sk':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/K_s_Icon.png';"
        elif marker == 'xk':
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/K_x_Icon.png';"
        elif marker == "ak":
            marker_image = "'http://www-e6.ijs.si/~javornik/Share/Icons/K_antenna_tower_Icon.png';"

        else:
            marker_image = "'https://developers.google.com/maps/documentation/javascript/examples/full/images/beachflag.png';"

        # processing info
        marker_info = ""
        i = 0
        for x in info_str:
            if i == 0:
                marker_info = marker_info + "<h3>" + str(x) + "</h3>"
            else:
                marker_info = marker_info + "<p>" + str(x) + "</p>"
            marker_info = marker_info + " \\\n                              "
            i=+1

        marker_info = "<div style=\"line-height: 1pt;\">" + marker_info + "</div>"
        marker_info = "'" + marker_info + "'\n"

        # adding node to html
        out_html = []
        for x in self.html:
            if (x.find(next_node_str) > 0):

                str_add = "\n            /* node " + str(node_Id) + " */ \n"
                str_add = str_add + "            var image = " + marker_image + "\n"
                str_add = str_add + "            var info = " + marker_info + "\n"
                out_html.append(str_add)

                str_add = "            marker_" + str(node_Id)
                str_add = str_add + " = addNode(map, " + position_str + ", image, info);\n\n"

                out_html.append(str_add)
                out_html.append(x)
            else:
                out_html.append(x)

        self.html = []
        for x in out_html:
            self.html.append(x)

        with open(self.html_file, "w") as myfile:
            myfile.writelines(self.html)
        return

    def add_overlay(self, overlay_Id, figure, bounds, opacity):
        """
        Adds the transparent overlay on the google map object

        :param str overlay_Id: overlay ID
        :param str figure: figure file of the overlay
        :param [float, float, float] bounds: bounds of figure: [north, south, east, west]
        :param float opacity: opacity [0 - 1.0] [transparent, non-transparent]

        :return:

        google_map.add_overlay(1, "test.png", region, 0.5)
        """
        bounds_str = ""
        bounds_str = bounds_str + "{north: " + str(bounds[0])
        bounds_str = bounds_str + ", south: " + str(bounds[1])
        bounds_str = bounds_str + ", east: " + str(bounds[2])
        bounds_str = bounds_str + ", west: " + str(bounds[3]) + "}"

        overlay_var = "            OverLay_" + str(overlay_Id)

        out_html = []
        for x in self.html:
            if (x.find("/*** next overlay ***/") > 0):
                str_add = "\n            /* Overlay " + str(overlay_Id) + " */ \n"
                str_add = str_add + "            var " + overlay_var + ";\n"
                str_add = str_add + "            var imageBounds = " + bounds_str + ";\n"
                str_add = str_add + "            var overlayOpts = {opacity:" + str(opacity) + "};\n"

                str_add = str_add + overlay_var + " = " + "new google.maps.GroundOverlay('"
                str_add = str_add + str(figure) + "', imageBounds, overlayOpts);\n"

                str_add = str_add + overlay_var + ".setMap(map);\n\n"

                out_html.append(str_add)
                out_html.append(x + "\n")
            else:
                out_html.append(x)

        self.html = []
        for x in out_html:
            self.html.append(x)

        with open(self.html_file, "w") as myfile:
            myfile.writelines(self.html)
        return

    def get_number_of_nodes(self):
        """
        Returns number of nodes in google_map object

        :returns: number of nodes
        :rtype: int

        """
        return len(self.node_Ids)

    def get_file_name(self):
        """
        Returns html file name of the google_map object

        :returns: name of html file
        :rtype: str
        """
        return self.html_file

    def add_net_from_csv(self, csvfile, network_name, marker):
        """
        Adds network defined by csv file to the google_map object

        :param str csvfile:  csv file name with the network
        :param str network_name: name of the netwok
        :param str marker: marker of nodes

        :return:

        google_map.add_net_from_csv(csv_file, "Anchors", "sg")
        """
        # Load nodes from csv file
        nodes = RE.Radio_Net.RadioNetwork(network_name)
        nodes.read_fromCsvFile(csvfile)
        self.add_network(nodes, marker)
        return

    def add_network(self, network, marker):
        """
        Adds a network network to the google_map object

        :param str network: name on network
        :param str network_name: name of the network
        :param marker: marker type

        :return:
        """
        NodeIds = network.get_RadioNode_Ids()
        network_id = network.get_Id()
        for id in NodeIds:
            plt_id = str(network_id) + "_" + str(id)
            node = network.get_RadioNode(id)
            LatLong = node.get_LatLong()
            Long = LatLong[1]
            Long = is_LatLong(str(Long))
            Lat = LatLong[0]
            Lat = is_LatLong(str(Lat))
            self.add_node(plt_id, [Long, Lat], [network_id, "N: " + str(id)], marker)
        return


if __name__ == '__main__':
    print "\n*******************************"
    print "*         miscGIS Test        *"
    print "*******************************\n"
    print "Conversion dms --> dec format: \n", "16.30dms", " = ", dms2dec("16.30"), "degree\n"
    print "Conversion GK --> WGS84 format: \n", "[469747, 97474] GK[Eest-West, North-South]", " = ", GKtoWGS84("469747", "97474"), "[Lat,Long] \n"
    print "\nTesting google_map class."
    print "Results at."


    # Test google map interface
    x,y = WGS84toGK(14.50, 46.50)
    lat, lng = GKtoWGS84(x, y)
    print x, y, lat, lng


    plot_html_file = "test.html"  # html file
    plot_png_file = "test.png"   # png file
    projDir = iniProject.iniDir(iniProject.get_proj_dir(3))
    plot_html_file = os.path.join(projDir["results"], plot_html_file)
    plot_png_file_save = os.path.join(projDir["results"], plot_png_file)

    map_center_LatLog = [46.0422027737, 14.4885690719]
    LOG_a_TEC_map = google_map("LOG.a.TEC", plot_html_file, map_center_LatLog, 20, "road")

    LOG_a_TEC_map.add_node(1, [46.0421967, 14.4885716], ["Node 1", "text 1", "text 1"], "ak")
    LOG_a_TEC_map.add_node(2, [46.0420794, 14.4877991], ["Node 2", "text 2", "text 2"], "ak")
    LOG_a_TEC_map.add_node(3, [46.0418141, 14.4879949], ["Node 3", "Node 3", "text 3"], "ak")
    LOG_a_TEC_map.add_node(4, [46.0423773, 14.4882363], ["Node 4", "text 4", "text 4"], "ak")

    csv_anchors = "Log-a-Tec_2_JSI.csv"  # all nodes in in Log-a-tec 2
    csv_file = os.path.join(projDir["csv"], csv_anchors)
    LOG_a_TEC_map.add_net_from_csv(csv_file, "Anchors", "sg")


    # generation of jpg file
    map_center_GK = WGS84toGK(map_center_LatLog[1], map_center_LatLog[0])
    x = np.linspace(-100, 100, 201)
    y = np.linspace(-100, 100, 201)
    xx, yy = np.meshgrid(x, y)
    zz = np.sqrt(xx ** 2 + yy ** 2)
    fig = pyplot.imshow(zz)
    pyplot.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    pyplot.savefig(plot_png_file_save, bbox_inches='tight', pad_inches = 0)

    print map_center_LatLog
    south = map_center_GK[1] - 100.0
    north = map_center_GK[1] + 100.0
    east = map_center_GK[0] + 100.0
    west = map_center_GK[0] - 100.0
    SW = GKtoWGS84(west, south)
    NE = GKtoWGS84(east, north)

    region = [is_LatLong(NE[1]), is_LatLong(SW[1]), is_LatLong(NE[0]), is_LatLong(SW[0])]
    print region
    LOG_a_TEC_map.add_overlay(1, "test.png", region, 0.5)

    print "Number on nodes: ", LOG_a_TEC_map.get_number_of_nodes()

    print "\n Process stopped"
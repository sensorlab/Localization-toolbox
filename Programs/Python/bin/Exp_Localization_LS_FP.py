"""
    Exp_Localization_LS_FP:
    ********************
    is an example how to use a toolbox to build an experiment

    Experiment objective:
    =====================
    test a localisation algorithms in a real outdoor environment

    Building experiment consists of following steps: \n
    - setup of radio environment
    - reading or generation of the measurements
    - running experiments \n
        - LS (least square) localization
        - FP (finger printing) localization

    Version 0.0: Tomaz Javornik (January 2018)

"""

print "\n************************************************************"
print "*                                                          *"
print "*   Experiment:                                            *"
print "*     Estimate node location                               *"
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
import RE.Raster_Map as REM_Maps
import RE.Radio_Net as REM_Network
import RE.Measurements as REM_Measure
import misc.iniProject as iniProject
import misc.GIS as GIS
import datetime
import json

now = datetime.datetime.now()


"""
    Setting radio environment
"""
IJS_Outdoor = REM_Env.RadioEnvironment("IJS Outdoor")

# Anchors nodes: read from csv file
#csv_anchors = "Log-a-Tec_2_JSI.csv"         # all nodes in in Log-a-tec 2
csv_anchors = "Log-a-Tec_2_JSI - 4A.csv"     # only 4 node in Log-a-tec

Agents_Size = 100                             # number of Agents/refAgents in network
quant_RSSI = 4.0                             # precision of RSSI measurement  in dBm
plot_net = True                              # plot the network in jpg file
plot_net_file = "Network.jpg"                # name of experimental jpg file
plot_html = True                             # plot in html file
plot_html_file = "Network.html"              # html file
calc_error = True                            # calculate error
number_of_experiments = 10                   # number of experiments
str_now = now.strftime("%Y-%m-%d_%H_%M")
save_results_to_file = "R_" + str_now       # simulation time
loc_method = "LS"                           # FP/LS
pl_exp = 2.0                                # path loss exponent index relevant for LS experiment
prop_channel = "LogNormal"                  # type of propagation channel "", "Rayleigh", "Rice", "LogNormal"
ch_param = 0.1                              # channel parameter

hist_plot = False                           # plot histogram
if Agents_Size == 1:                        # do not plot histogram for only one test
    hist_plot = False

projDir = iniProject.iniDir(iniProject.get_proj_dir(3))
# Save simulation parameters to file
results_file_json = os.path.join(projDir["results"], save_results_to_file + ".json")
sim_param = {"Simulation parameters":
    {"Anchors file" : csv_anchors,
     "Number of agents" : Agents_Size,
     "Quantization step" : quant_RSSI,
     "Number of experiments" : number_of_experiments,
     "Localization method" : loc_method,
     "Path Loss exponent" : pl_exp,
     "Propagation channel" : prop_channel,
     "Channel parameters" : ch_param}}

with open(results_file_json, 'w') as outfile:
    json.dump(sim_param, outfile,  indent=2)

# Setting project dir, results dir, plot files
plot_net_file = os.path.join(projDir["results"], plot_net_file)
plot_html_file = os.path.join(projDir["results"], plot_html_file)

# Load anchor nodes from csv file
csv_anchors = os.path.join(projDir["csv"], csv_anchors)
Anchors = REM_Network.RadioNetwork("Anchors")
Anchors.read_fromCsvFile(csv_anchors)
IJS_Outdoor.append("Network", "Anchors", Anchors)
del projDir, csv_anchors

# SetUp region map based on radio Anchor nodes
Region = Anchors.get_Region_Map(0.10)
IJS_Outdoor.set_Region(Region)
del Anchors

"""
    Generate or read measurements at specified location/locations
"""
# SetUp a set of maps with a RSSI values using FSPL channel
RSSI_maps = REM_Maps.RasterMaps('RSSI maps', "Anchors")
Region = IJS_Outdoor.get_Region()
Txs = IJS_Outdoor.get("Network", "Anchors")
RSSI_maps.calc_RSSI_FSPL(Region, Txs)
IJS_Outdoor.append("Raster Map", "RSSI", RSSI_maps)

"""
    Experiment loop
"""
for i_exp in range(0, number_of_experiments):
    # SetUp a network of Agents, nodes with unknown locations
    Agents = REM_Network.RadioNetwork("Agents", Agents_Size)
    IJS_Outdoor.append("Network", "Agents", Agents)
    del Agents

    # SetUp a network of refAgents, reference nodes which estimates measurements
    refAgents = REM_Network.RadioNetwork("refAgents")
    Region = IJS_Outdoor.get_Region()
    refAgents.add_rnd_Loc(Region, Agents_Size)
    IJS_Outdoor.append("Network", "refAgents", refAgents)
    del refAgents

    # Extract measurements from the maps
    RSSI_trace = REM_Measure.Trace("RSSI")
    Rx_Locs = IJS_Outdoor.get("Network", "refAgents").get_Locations()
    Rx_Ids = IJS_Outdoor.get("Network", "refAgents").get_RadioNode_Ids()
    Tx_Ids = IJS_Outdoor.get("Network", "Anchors").get_RadioNode_Ids()
    x = {"Id": None, "Unit": "dBm", "Type": "RSSI", "Value": None,
         "Tx_Node_Id": None, "Tx_Network_Id": "Anchors",
         "Rx_Node_Id": None, "Rx_Network_Id": "Agents"}

    RSSI_trace.set_values_from_maps(RSSI_maps, Rx_Locs, Rx_Ids, Tx_Ids, x)
    if prop_channel == "Rayleigh":
        RSSI_trace.add_error("Rayleigh")            # add Rayleigh fading
    if prop_channel == "LogNormal":
        RSSI_trace.add_error("LogNormal", 0, ch_param)            # add Rayleigh fading
    if prop_channel == "Rice":
        RSSI_trace.add_error("Rice", 0, ch_param)            # add Rayleigh fading

    RSSI_trace.quantize_vals(quant_RSSI)               # add Measurement quantization
    IJS_Outdoor.append("Measurements", "RSSI", RSSI_trace)

    """
        Perform a localization experiment
    """
    ref_node_index = 1              # reference node index relevant for LS experiment

    if loc_method == "LS":
        IJS_Outdoor.experiment_est_loc_LS(pl_exp, ref_node_index, hist_plot)    # least square localization experiment
    if loc_method == "FP":
        IJS_Outdoor.experiment_est_loc_FP(pl_exp, ref_node_index, hist_plot)   # finger printing localization experiment

    """
        Plot and analise experiment results
        
    """
    Anchors = IJS_Outdoor.get("Network", "Anchors")
    refAgents = IJS_Outdoor.get("Network", "refAgents")
    Agents = IJS_Outdoor.get("Network", "Agents")

    # Plot Anchors, refAgents, Agents
    if plot_net:
        Region = IJS_Outdoor.get_Region()
        reg = Region.get_Region()
        fig, ax = Region.plot_Network(2, False, 'sc', 10, Anchors, False, "Anchors", reg)
        fig, ax = Region.plot_Network(2, False, 'or', 8, Agents, False, "Loc-Est", reg)
        fig, ax = Region.plot_Network(2, True, 'xk', 8, refAgents, False, "Loc-Real", reg)
        ax.grid()
        ax.set_aspect('equal')
        fig.suptitle("Log-A-Tec")
        fig.savefig(plot_net_file)
        fig.clear()

    # Plot in html
    if plot_html:
        LOG_a_TEC_map = GIS.google_map("LOG.a.TEC", plot_html_file, [46.0422027737, 14.4885690719], 20, "road")
        LOG_a_TEC_map.add_network(Anchors, "ak")
        LOG_a_TEC_map.add_network(refAgents, "og")
        LOG_a_TEC_map.add_network(Agents, "xb")

    if calc_error:
        err_mean, err_median, err_std, err_max = Agents.loc_error(refAgents, False, False)
        data_to_save = {"Simulation number": i_exp,
                        "Results":
                            {"Mean error": err_mean,
                             "Median error": err_median,
                             "Std error": err_std,
                             "Max error": err_max}}
        with open(results_file_json, 'a') as outfile:
            outfile.write('\n')
            json.dump(data_to_save, outfile, indent=2)

        print "\t", err_mean, err_median, err_std, err_max

    # Clean Radio environment for new simulation
    IJS_Outdoor.delete("Network", "Agents")
    IJS_Outdoor.delete("Network", "refAgents")
    IJS_Outdoor.delete("Measurements", "RSSI")

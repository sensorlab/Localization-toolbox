Localization toolbox

Localization toolbox is a python framework, which enables developing, testing and evaluation of various localization algorithms. The localization algorithms evaluation can be based on simulated radio environments as well on radio environment based on measurements obtained from various testbeds.

The detail user maunal can be found on Programs/Python/docs/_build/html/index.html. 

The project directory structure is as follows:
Localization toolbox:
* Data
    o csv
    o json
    o npy
    o txt
    o xls
    o xyz
* Figures
    o Raster
    o Vector
* Programs
    o Python
* bin
* docs
* misc
* RE
* Results
* temp
* tests
Data directory consists of various data files for conducting the experiment. Comma separated value files are stored in csv subdirectory, JSON files in json subdirectory, python data file in npy directory, different data files in txt subdirectory, MS excel files in xls subdirectory, and txt data with xyz format in xyz subdirectory. 

Figures directory contain two subdirectories namely, the first for storing raster files, and the second one contains the figure vector figures.

The main python files of the toolbox are stored in Programs/Python. The directory contains following subdirectories: bin, docs, RE, misc, Results, temp and tests.

The experiments are stored in bin subdirectory. 

docs subdirectory contains the documentation of a localization toolbox (\docs\_build\html\index.html) and this readme file.
 
Three application interfaces to JSON files, project initialization file and various GIS routines are stored in misc subdirectory. 

RE directory contains the description of radio environment:
* Radio_env.py
    o class: RadioEnvironment: describes radio environment
* Raster_map.py
    o class: RasterMap: describes raster map
    o class: RasterMaps: set of raster maps
* Radio_Net.py
    o class: RadioNode: describes radio node
    o class: RadioNetwork: describes radio network, a set of radio nodes
* Measurements.py:
    o class: Measurement: describes a single measurement
    o class: Trace: a set of radio measurements
Results directory contains temporary simulation results. 
Temporary files should be stored in temp directory.
Building experiment:
The experiment using toolbox contains following steps:
* Setting up radio environment
    o Setup a Region map
    o Setup a Anchor network
    o Setup a Agents network
    o Setup a reference network
* Generate or read measurements at specified location/locations
* Implement a localization algorithm in python script
* Perform a localization experiment
* Plot/Analyse experiment results

"""
    misc.iniProject
    ===============
    initializes a set of project directories:
        - projDir     directory \n
        - dataDir:    data directry \n
        - figDir:     figure directory \n
        - progDir:    program directory \n

    and data units, data types, google map types:
        - Data_Units = ["m", "km", "s", "ms", "dBm", "dBu"] \n
        - Data_Types = ["RSSI", "ToA", "TDoA", "Dist"]  \n
        - Google_Map_Types = ["roadmap", "satellite", "hybrid,", "terrain"] \n

"""

import os
# valid data units in project
Data_Units = ["m", "km", "s", "ms", "dBm", "dBu"]

# valid data types in project
Data_Types = ["RSSI", "ToA", "TDoA", "Dist"]

# valid google map tpyes in project
Google_Map_Types = ["roadmap", "satellite", "hybrid,", "terrain"]


def get_proj_dir(N):
    """
    Gets N-th ancestor director of the project dir.

    :param int N: N-th ancestor directory: N = 0 project  dir
    :returns: path to current working directory

    :rtype: str
    """
    path = os.getcwd()
    for i in range(0, N):
        [path, tail] = os.path.split(path)
    return path


def iniDir(projDir):
    """
    iniDir is used to specify directory structure of the project.

    :param str projDir: path to project directory

    :return:
    """
    if os.path.isdir(projDir):
        dirStruct = {'main': projDir}
    else:
        print "Error! Project directory " + projDir + " does not exists!"
        return

    # temp directory
    tmp = os.path.join(projDir, "temp")
    dirStruct['temp'] = tmp
    
    dataDir = os.path.join(projDir, "Data")
    dirStruct['data'] = dataDir

    figDir = os.path.join(projDir, "Figures")
    dirStruct['fig'] = figDir

    progDir = os.path.join(projDir, "Programs")
    dirStruct['program'] = progDir


    # Data directories
    dirStruct['csv'] = os.path.join(dataDir, "csv")
    dirStruct['html'] = os.path.join(dataDir, "html")
    dirStruct['kml'] = os.path.join(dataDir, "kml")
    dirStruct['mat'] = os.path.join(dataDir, "mat")
    dirStruct['xyz'] = os.path.join(dataDir, "xyz")
    dirStruct['npy'] = os.path.join(dataDir, "npy")
    dirStruct['xls'] = os.path.join(dataDir, "xls")
    dirStruct['txt'] = os.path.join(dataDir, "txt")
    dirStruct['json'] = os.path.join(dataDir, "json")

    # Figure directories
    dirStruct['eps'] = os.path.join(figDir, "eps")
    dirStruct['matfig'] = os.path.join(figDir, "matfig")
    dirStruct['raster'] = os.path.join(figDir, "raster")
    dirStruct['vector'] = os.path.join(figDir, "vector")

    # Program directories
    tmp = os.path.join(progDir, "Python")

    dirStruct['python'] = tmp
    dirStruct['template'] = os.path.join(tmp, "docs")
    dirStruct['results'] = os.path.join(tmp, "Results")

    #dirStruct['c'] = os.path.join(progDir, "C")
    #dirStruct['cc'] = os.path.join(progDir, "CC")
    #dirStruct['java'] = os.path.join(progDir, "Java")
    #dirStruct['Matlab'] = os.path.join(progDir, "mat")
    #dirStruct['AMS'] = os.path.join(progDir, "AMS_DEMO")
    #dirStruct['bin'] = os.path.join(progDir, "bin")

    # http sftp dir
    return dirStruct


if __name__ == '__main__':
    import iniProject
    import os
    print "\n*******************************"
    print "*         iniProj testing     *"
    print "*******************************\n"
    projDir = iniProject.iniDir(os.getcwd())
    xyzFile = os.path.join(projDir['xyz'], 'test.xyz')
    print "Path to xyz type of files: ", xyzFile

    parentDir = iniProject.get_proj_dir(1)
    print "Parent directory: ", parentDir
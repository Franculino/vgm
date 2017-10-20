from __future__ import division, with_statement

import cPickle
import csv
from operator import itemgetter
import numpy as np
import scipy as sp

#import guiTools
import units
from vascularGraph import VascularGraph
import vgm

__all__ = ['read_csv', 'read_amira_spatialGraph', 'read_amira_spatialGraph_v2',
           'read_pkl', 'read_landmarks']
log = vgm.LogDispatcher.create_logger(__name__)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def read_csv(vertexFile='vertices.csv', edgeFile='edges.csv'):
    """Construct a VascularGraph from two CSV files, containing 1) vertex 
    coordinates and 2) edgelist and vessel diameters respectively.
    INPUT: vertexFile: The name of the vertex file. The expected format is
                       x1, y1, z1
                       x2, y2, z2
                       ...
                       Where x,y,z are the floating point coordinates of the
                       vertices.
           edgeFile: The name of the edge file. The expected format is
                     a1, b1, d1
                     a2, b2, d2
                     ...
                     Where a,b are the vertex indices associated to an edge and
                     d is its diameter.
    OUTPUT: VascularGraph                  
    """
    # Read vertex coordinates:
    f = open(vertexFile)
    reader = csv.reader(f)
    coordinates = []
    kind = []
    for line in reader:
        coordinates.append(np.array([float(line[0]), 
                                     float(line[1]),
                                     0.0]))
        kind.append(int(line[2]))
    f.close()
    
    # Read edgelist and diameters:
    f = open(edgeFile)
    reader = csv.reader(f)
    edgelist = []
    diameters = []
    for line in reader:
        edgelist.append((int(line[0])-1, int(line[1])-1))
        diameters.append(float(line[2]))
    f.close()
    
    # Construct VascularGraph:
    G = VascularGraph(edgelist)
    G.vs['r'] = coordinates
    G.vs['nkind'] = kind
    G.es['diameter'] = diameters
    return G


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def read_amira_spatialGraph(filename=None, resolution=1.0, lUnit='um', 
                            **kwargs):
    """Reads an AMIRA spatial graph file (AmiraMesh 3D ASCII 2.0 format) and 
    constructs the corresponding vascular graph from it.
    
    INPUT: filename: AMIRA spatial graph file (including path). If not 
                     provided, a graphical file-selection dialog will display.
           resolution: The resolution of the srXTM scan, i.e. voxel size 
                       (isotropic resolution is assumend). The default is 1.0.
           lUnit: The unit of length properties in the spatialGraph (i.e. 
                  radii, and position vectors). The default is microns.
           **kwargs:       
           defaultUnits: The defaultUnits of the VascularGraph created (see 
                         __init__ of the VascularGraph class for details).        
    OUTPUT: G: Vascular graph in iGraph format. 
    """    

    #if filename is None:
    #    filename = guiTools.uigetfile('Select Amira spatialGraph (.am) file')
        
    f = open(filename,'r')

    def advanceToToken(token,searchdepth=2):
        done = False
        while not done:
            l = f.readline()
            if l == '': # eof
                break
            if l[0:searchdepth] == token:
                done = True
                log.debug('Found ' + token)
                return l
    
    # Get file statistics
    l = advanceToToken('define VERTEX',13)
    n_vertices = int(l.split()[-1])
    l = advanceToToken('define EDGE',11)
    n_edges = int(l.split()[-1])
    l = advanceToToken('define POINT',12)
    n_points = int(l.split()[-1])
    log.info("Vertices: " + str(n_vertices))
    log.info("Edges: "    + str(n_edges))
    log.info("Points: "   + str(n_points))
    
    # Create VascularGraph and compute scaling factor
    G = VascularGraph(n_vertices, **kwargs)
    sf = units.scaling_factor_du(lUnit, G['defaultUnits'])
    
    # Add vertices
    log.info("reading and adding vertices")
    advanceToToken('@1')
    edgeConnectivity     = []
    numEdgePoints        = []
    r = [[] for i in xrange(n_vertices)]    
    for i in xrange(n_vertices):
        l = f.readline()
        l = l.split()
        r[i] = sp.array([float(l[0]),float(l[1]),float(l[2])]) * resolution * sf   
    
    G.vs['r'] = r
    
    
    # Read connectivity information
    log.info("reading connectivity information")
    advanceToToken('@2')
    for i in xrange(n_edges):
        l = f.readline()
        l = l.split()
        edgeConnectivity.append((int(l[0]),int(l[1])))
    
    # Read number of edge points
    log.info("reading number of edge points")
    advanceToToken('@3')
    for i in xrange(n_edges):
        l = f.readline()
        numEdgePoints.append(int(l))
    
    # Read edge point coordinates
    # Compute length
    log.info("reading point coordinates")
    advanceToToken('@4')
    l_edge = []
    edgepoints = []
    for edge in xrange(n_edges):
        tmppoints = sp.zeros([numEdgePoints[edge],3])
        for point in xrange(numEdgePoints[edge]):
            l = f.readline() 
            tmppoints[point] = map(float, l.split())
        l_edge.append([ np.linalg.norm(tmppoints[i]-tmppoints[i+1]) 
                        for i in xrange(len(tmppoints)-1) ])
        l_edge[edge].insert(0, 0.0)
        l_edge[edge].extend([0.0])
        l_edge[edge] = [(l_edge[edge][i-1] + l_edge[edge][i]) / 2.0 
                        for i in xrange(1,len(l_edge[edge]))]        
        
        # iGraph, unlike Amira orders edge vertices by index. Make sure that
        # this is reflected in the points that make up the edge:
        if edgeConnectivity[edge][0] > edgeConnectivity[edge][1]:
            tmppoints = tmppoints[::-1,:]        
        edgepoints.append(tmppoints)
        if sp.mod(edge+1,sp.floor(n_edges/10)) == 0:
            log.info(str(10*(edge+1)/sp.floor(n_edges/10)) + "%")

            
    
    # Read edge radii
    log.info("reading radii")
    advanceToToken('@5')   
    d_edge = []
    for edge in xrange(n_edges):
        tmpdiameters = []
        for point in xrange(numEdgePoints[edge]):
            l = f.readline()
            tmpdiameters.append(float(l)*2.0) # conversion radius to diameter
        d_edge.append(tmpdiameters)
    f.close()  
    
    # Add edges
    log.info("adding edges")
    G.add_edges(edgeConnectivity)
    d_edge = [sp.array(d_edge[i],dtype=float) * resolution * sf
              for i in xrange(n_edges)]
    l_edge = [sp.array(l_edge[i],dtype=float) * resolution * sf
              for i in xrange(n_edges)]               
    G.es['diameters'] = d_edge
    G.es['points'] = [p * resolution * sf for p in edgepoints]
    # Compute average diameter, weighted by length:
    # - volume = pi * d**2 /4 * length
    # - morphologically representative because of length-weighting
    # - individual diameters scale with identical factors as average diameter
    G.es['diameter'] = [np.sqrt(np.average(d_edge[i]**2, weights=l_edge[i])) 
                        for i in xrange(n_edges)]
    G.es['lengths'] = l_edge
    G.es['length'] = [sum(l) for l in l_edge]
                      
    return G
    
    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def read_amira_spatialGraph_v2(filename=None, resolution=1.0, lUnit='um', 
                               **kwargs):
    """Reads an AMIRA spatial graph file (AmiraMesh 3D ASCII 2.0 format) and 
    constructs the corresponding vascular graph from it. This version accounts
    for the fact that some spatialGraph files have colocalized vertices, which 
    are actually identical (this happens especially after conversion from an
    mv3d file).
    
    INPUT: filename: AMIRA spatial graph file (including path). If not 
                     provided, a graphical file-selection dialog will display.
           resolution: The resolution of the srXTM scan, i.e. voxel size 
                       (isotropic resolution is assumend). The default is 1.0.   
           lUnit: The unit of length properties in the spatialGraph (i.e. 
                  radii, and position vectors). The default is microns.
           **kwargs:                  
           defaultUnits: The defaultUnits of the VascularGraph created (see 
                         __init__ of the VascularGraph class for details).
    OUTPUT: G: Vascular graph in iGraph format. 
    """
    
    #if filename is None:
    #    filename = guiTools.uigetfile('Select Amira spatialGraph (.am) file')
        
    f = open(filename,'r')

    def advanceToToken(token,searchdepth=2):
        done = False
        while not done:
            l = f.readline()
            if l == '': # eof
                break
            if l[0:searchdepth] == token:
                done = True
                log.debug('Found ' + token)
                return l
    
    # Get file statistics
    l = advanceToToken('define VERTEX',13)
    n_vertices = int(l.split()[-1])
    l = advanceToToken('define EDGE',11)
    n_edges = int(l.split()[-1])
    l = advanceToToken('define POINT',12)
    n_points = int(l.split()[-1])
    log.info("Vertices: " + str(n_vertices))
    log.info("Edges: "    + str(n_edges))
    log.info("Points: "   + str(n_points))
    log.info("reading and adding vertices")
        
    # Add vertices            
    advanceToToken('@1')
    edgeConnectivity     = []
    numEdgePoints        = []

    rDict = {} # relates coordinates to first vertex occurence
    vDict = {} # relates later vertex occurences to first one
    vCounter = 0
    for i in xrange(n_vertices):
        l = f.readline()
        l = l.split()
        rTmp = (float(l[0]),float(l[1]),float(l[2]))
        if rDict.has_key(rTmp):
            vDict[i] = rDict[rTmp]
        else:
            rDict[rTmp] = vCounter
            vDict[i] = vCounter
            vCounter += 1
            
    # Create VascularGraph and compute scaling factor                
    G = VascularGraph(len(rDict), **kwargs) # len(rDict) <= n_vertices
    sf = units.scaling_factor_du(lUnit, G['defaultUnits'])
    
    # sort r by vertex number and scale with 'resolution' and 'sf':   
    G.vs['r'] = sp.array([x[0] for x in sorted(rDict.items(),
                         key=itemgetter(1))]) * resolution * sf

    # Read connectivity information
    log.info("reading connectivity information")
    advanceToToken('@2')
    for i in xrange(n_edges):
        l = f.readline()
        l = l.split()
        edgeConnectivity.append((vDict[int(l[0])],vDict[int(l[1])]))
    
    # Read number of edge points
    log.info("reading number of edge points")
    advanceToToken('@3')
    for i in xrange(n_edges):
        l = f.readline()
        numEdgePoints.append(int(l))
    
    # Read edge point coordinates
    # Compute length
    log.info("reading point coordinates")
    advanceToToken('@4')
    # Assign a length to each point (i.e.: #lengths == #points):
    l_edge = []
    edgepoints = []
    for edge in xrange(n_edges):
        tmppoints = sp.zeros([numEdgePoints[edge],3])
        for point in xrange(numEdgePoints[edge]):
            l = f.readline() 
            tmppoints[point] = map(float, l.split())
        l_edge.append([ np.linalg.norm(tmppoints[i]-tmppoints[i+1]) 
                        for i in xrange(len(tmppoints)-1) ])
        l_edge[edge].insert(0, 0.0)
        l_edge[edge].extend([0.0])
        l_edge[edge] = [(l_edge[edge][i-1] + l_edge[edge][i]) / 2.0 
                        for i in xrange(1,len(l_edge[edge]))]
        
        # iGraph, unlike Amira orders edge vertices by index. Make sure that
        # this is reflected in the points that make up the edge:
        if edgeConnectivity[edge][0] > edgeConnectivity[edge][1]:
            tmppoints = tmppoints[::-1,:]
        edgepoints.append(tmppoints)
        if sp.mod(edge+1,sp.floor(n_edges/10)) == 0:
            log.info(str(10*(edge+1)/sp.floor(n_edges/10)) + "%")
            

            
    
    # Read edge radii
    log.info("reading radii")
    advanceToToken('@5')   
    d_edge = []
    for edge in xrange(n_edges):
        tmpdiameters = []
        for point in xrange(numEdgePoints[edge]):
            l = f.readline()
            tmpdiameters.append(float(l)*2.0) # conversion radius to diameter
        d_edge.append(tmpdiameters)
    f.close()                    

    # Add edges:
    log.info("adding edges")
    G.add_edges(edgeConnectivity)    
    d_edge = [sp.array(d_edge[i],dtype=float) * resolution * sf
              for i in xrange(n_edges)]    
    l_edge = [sp.array(l_edge[i],dtype=float) * resolution * sf
              for i in xrange(n_edges)]    
    # Compute average diameter, weighted by length:
    # - volume = pi * d**2 /4 * length
    # - morphologically representative because of length-weighting
    # - individual diameters scale with identical (multiplicative) factors as 
    #   the average diameter
    try:    
        G.es['diameter'] = [np.sqrt(np.average(d_edge[i]**2, weights=l_edge[i])) 
                            for i in xrange(n_edges)]
    except:
        # Length-weighting is not possible for zero-length edges. These
        # pathological edges need to be removed:
        length = [sum(l) for l in l_edge]    
        zeroLength = np.nonzero(np.array(length) == 0.0)[0]
        nonzeroLength = np.nonzero(np.array(length) != 0.0)[0]
        d_edge = [x for i,x in enumerate(d_edge) if i in nonzeroLength]
        l_edge = [x for i,x in enumerate(l_edge) if i in nonzeroLength]
        edgepoints = [x for i,x in enumerate(edgepoints) if i in nonzeroLength]
        G.delete_edges(zeroLength.tolist())
        n_edges = G.ecount()
        log.warning('%i zero-length edges were removed. %i edges remaining' % \
                   (len(zeroLength), n_edges))
        G.es['diameter'] = [np.sqrt(np.average(d_edge[i]**2, weights=l_edge[i])) 
                            for i in xrange(n_edges)]

    G.es['diameters'] = d_edge
    G.es['points'] = [p * resolution * sf for p in edgepoints]        
    G.es['lengths'] = l_edge
    G.es['length'] = [sum(l) for l in l_edge]               

    return G
    
    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def read_pkl(filename=None):
    """Reads a Python pickle file and returns the data stored therein. This can
    be used, for example, to read vascular graphs written to disk using
    'write_pkl'.
    INPUT: filename: Pickle file (including path). If not provided, a graphical
                     file-selection dialog will display.
    OUTPUT: G: Data stored in the pickle file, e.g. vascular graph in iGraph 
               format. 
    """

    #if filename is None:
    #    filename = guiTools.uigetfile('Select pickle file')

    with open(filename,'rb') as f:
        G = cPickle.load(f)

    return G    


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def read_landmarks(filename=None, scalingFactor=1.0):
    """Reads an Amira generated file containing landmarks (coordinates), which
    are returned in list form.
    INPUT: filename: Absolute or relative path to the landmarks file. If not 
                     provided, a graphical file-selection dialog will display.
           scalingFactor: The factor by which the coordinates in the landmark
                          file should be multiplied with (this is useful if the
                          landmark file uses units other than desired). The
                          default is 1.0, i.e. the coordinates will not be 
                          modified.
    OUTPUT: coordinates: List of lists, specifying the coordinates of the 
                         landmarks (i.e.: [[x1,y1,z1],[x2,y2,z2],...]).
    """

    #if filename is None:
    #    filename = guiTools.uigetfile('Select landmarks file')

    f = open(filename,'r')
    
    def advance_to_token(token):
        done = False
        while not done:
            l = f.readline()
            if l == '': # eof
                break
            if l.find(token) != -1:
                done = True
                return l
    
    l = advance_to_token('define Markers')
    l = l.split()
    num_markers = int(l[2])
    
    advance_to_token('@1')
    advance_to_token('@1')    
    
    
    coordinates = []
    for i in xrange(num_markers):
        l = f.readline()
        l = l.split()
        coordinates.append([float(l[0]) * scalingFactor,
                            float(l[1]) * scalingFactor,
                            float(l[2]) * scalingFactor])
    
    f.close()
    
    return coordinates    
    
    

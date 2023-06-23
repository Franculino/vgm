from __future__ import division                   
import vgm
import csv
import cPickle
import igraph as ig
import numpy as np
from pylab import flatten
import scipy as sp
from scipy import (array, arccos, argmin, concatenate, dot, ones, mean, pi, 
                   shape, unique, finfo)
from scipy.linalg import norm
from scipy.spatial import kdtree
from scipy.interpolate import griddata
#THOSE lines produce error sometimes switch them off
from sympy.solvers import solve
from sympy import Symbol

from linearSystem import LinearSystem
from physiology import Physiology 

__all__ = ['add_geometric_edge_properties', 'add_fluiddynamical_properties',
           'vertices_from_coordinates', 'add_kind_and_conductance', 
           'add_conductance', 'edge_property_vs_depth', 'update_lengths',
           'update_length', 'update_volume', 'update_depth','exchange_graph_purePython']

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def add_geometric_edge_properties(G):
    """ Adds angle to cortical surface (in degrees), cortical depth, volume and
    cross section to each edge in the graph.
    INPUT: G:  Vascular graph in iGraph format.
    OUTPUT: None - the vascular graph G is modified in place.
    """
           
    depth = []
    angle = []
    crossSection = [] 
    volume = []

    ez = array([0,0,1])
    
    for edge in G.es:         
        
        a = G.vs[edge.source]['r']
        b = G.vs[edge.target]['r']
        v = a-b    
        depth.append((a[2]+b[2])/2.0)
        
        theta=arccos(dot(v,ez)/norm(v))/2/pi*360
        if theta > 90:
            theta = 180-theta
        angle.append(theta)
    
        crossSection.append(np.pi * edge['diameter']**2. / 4.)
        volume.append(crossSection[-1] * edge['length'])
    
    
    G.es['depth'] = depth
    G.es['angle'] = angle
    G.es['volume'] = volume
    G.es['crossSection'] = crossSection 
           

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def add_fluiddynamical_properties(G):
    """Adds transit-time and velocity to each edge in the graph.
    INPUT: G:  Vascular graph in iGraph format.
    OUTPUT: None - the vascular graph G is modified in place.
    WARNING: The edge property 'flow' is expected to be assigned and is 
             interpreted as volume flow.
    """

    transitTime = []
    velocity = []
    
    ez = array([0,0,1])
    for edge in G.es: 
        points = edge['points']
        diameters = edge['diameters']
        flow = edge['flow']
        volume = edge['volume']
        tmpVelocity = []
        for i in xrange(len(points)-1):
            l = norm(points[i+1] - points[i])
            A = pi * mean([diameters[i+1],diameters[i]])**2.0 / 4.0
            tmpVelocity.append(A/flow)
        transitTime.append(volume / flow)
        velocity.append(mean(tmpVelocity))

    G.es['transitTime'] = transitTime
    G.es['velocity'] = velocity
           

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def vertices_from_coordinates(G, coordinates, diameter_ll=0.0, 
                              isEndpoint=False):
    """Given a list of x,y,z coordinates, locate the most closely matching set
    of verties in a vascular graph of iGraph format.
    INPUT: G:  Vascular graph in iGraph format. 
           coordinates: List of lists specifying the coordinates (i.e.: 
                        [[x1,y1,z1],[x2,y2,z2],...]).
           diameter_ll: (Optional) lower limit of edge diameter, to select only
                        those vertices bordering a sufficiently large diameter
                        edge. Default is 0.0, i.e. all vertices are considered.
           isEndpoint: Boolean whether or not the vertex searched for is 
                       required to be an endpoint. Default is 'False'.
    OUTPUT: vertex_indices: Array of vertex indices that represent the best 
                            matches.
            distances: Array of distances that the best matching vertices are 
                       separated from the supplied coordinates. Units match 
                       those of the graph vertex coordinates.
    """
    
    # Select vertex indices based on diameter of adjacent edges:
    si = unique(flatten([G.es[x].tuple for x in 
         G.es(diameter_ge=diameter_ll).indices])).tolist()
    # Optionally filter for end-points:
    if isEndpoint:
        si = [i for i in si if G.degree(i) == 1]
    # Construct k-dimensional seach-tree:
    kdt = kdtree.KDTree(G.vs[si]['r'], leafsize=10)
    search_result = kdt.query(coordinates)
    sr_v = np.ravel([search_result[1]]).tolist()
    vertex_indices = [si[x] for x in sr_v]
    distances = np.ravel([search_result[0]]).tolist()    

    return vertex_indices, distances 
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def add_kind_and_conductance(G,aVertices,vVertices, dThreshold):
    """Adds vessel kind as well as conductance to the graph. The vessel kind is
    assigned based on a connected component analysis and vertices of known 
    kind. The vessel conductance is computed from fitted literature values. At 
    capillary level, a distinction between 'artery' and 'vein' is difficult to 
    make. Moreover, the relative apparent viscosity does not differ 
    dramatically. Therefore, all capillaries (as well as undefined vessels) are
    set to arterial conductance.    
    INPUT: G: VascularGraph.
           aVertices: Vertices that belong to arterial trees.
           vVertices: Vertices that belong to venous trees.
           dThreshold: The diameter threshold below equal which vessels are 
                       considered as capillaries. 
    OUTPUT: None - G is modified in-place.                   
    """                          
    
    # Arteries and veins:
    for kind, indices in zip(['a','v'],[aVertices, vVertices]):
        for vIndex in indices:
            treeVertices = G.get_tree_subcomponent(vIndex, dThreshold)
            edgeIndices = G.es(G.get_vertex_edges(treeVertices), 
                               diameter_gt=dThreshold).indices
            G.es(edgeIndices)['kind'] = [kind for e in edgeIndices]                   
            add_conductance(G,kind,invivo,edgeIndices)

    # Capillaries:
    capillaryIndices = G.es(diameter_le=dThreshold).indices
    G.es(capillaryIndices)['kind'] = ['c' for c in capillaryIndices]            
    add_conductance(G,'a',invivo,capillaryIndices)
    
    # Not assigned:
    notAssigned = []
    for e in G.es:
        if e['kind'] is None:
            notAssigned.append(e.index)
    G.es(notAssigned)['kind'] = ['n' for c in notAssigned]            
    add_conductance(G,'a',invivo,notAssigned)            

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def add_conductance(G,kind,invivo,edges=None):
    """Adds conductance values to the edges of the graph (consult the relevant
    functions in the physiology module for more information.
    INPUT: G: Vascular graph in iGraph format.
           kind: The vessel kind. This can be either 'a' for artery or 'v' for
                 vein.
           invivo: Boolean, whether the physiological blood characteristics 
                   are calculated using the invivo (=True) or invitro (=False)
                   equations
           edges: (Optional.) The indices of the edges to be given a 
                  conductance value. If no indices are supplied, all edges are 
                  considered.
    """
    P = Physiology(G['defaultUnits'])
    if edges is None:
        edgelist = G.es
    else:
        edgelist = G.es(edges)
    #for e in edgelist:
        #print('')
        #print(e['diameter'])
        #print(e['length'])
        #print(P.dynamic_blood_viscosity(e['diameter'],invivo,kind))
    G.es(edgelist.indices)['conductance'] = \
                                   [P.conductance(e['diameter'],e['length'],
                                       P.dynamic_blood_viscosity(e['diameter'],
                                                                 invivo,kind))
                                    for e in edgelist]

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def edge_property_vs_depth(G,property,intervals,eIndices=None,function=None):
    """Generic function to compile and optionally process edge information of a
    vascular graph versus the cortical depth.
    INPUT: G: Vascular graph in iGraph format.
           property: Which edge property to operate on.
           intervals: Intervals of cortical depth in which the sample is split.
                      (Expected 
           eIndices: (Optional.) Indices of edges to consider. If not provided,
                     all edges are taken into account.
           function: (Optional.) Function which to perform on the compiled data
                     of each interval.
    OUTPUT: The compiled (and possibly processed) information as a list (one 
            entry per interval).
    """
    
    intervals[-1] = (intervals[-1][0], intervals[-1][1] + sp.finfo(float).eps)
    database = []
    for interval in intervals:
        if eIndices:
            data = G.es(eIndices,depth_ge=interval[0], 
                        depth_lt=interval[1])[property]
        else:                
            data = G.es(depth_ge=interval[0], 
                        depth_lt=interval[1])[property]
        if function:
            database.append(function(data))
        else:
            database.append(data)
    return database   


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def update_lengths(G):
    """Updates the lengths property of all edges in the VascularGraph (adds it,
    if it does not exist).
    INPUT: G: VascularGraph.
    OUTPUT: None
    """
    lengths = []
    for e in G.es:
        tmpLengths = [np.linalg.norm(e['points'][i] - e['points'][i+1]) 
                      for i in xrange(len(e['points']) - 1)]
        tmpLengths.insert(0, 0.0)
        tmpLengths.append(0.0)
        lengths.append((np.array(tmpLengths[:-1]) + 
                       np.array(tmpLengths)[1:]) / 2.0)
    G.es['lengths'] = lengths
    
        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def update_length(G):
    """Updates the length property of all edges in the VascularGraph (adds it,
    if it does not exist).
    INPUT: G: VascularGraph.
    OUTPUT: None
    """
    if 'lengths' in G.es.attribute_names():
        G.es['length'] = [sum(e['lengths']) for e in G.es]
    else:
        G.es['length'] = [np.linalg.norm(G.vs[e.source]['r'] - 
                                         G.vs[e.target]['r']) for e in G.es]    


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def update_volume(G):
    """Updates the volume property of all edges in the VascularGraph (adds it,
    if it does not exist).
    INPUT: G: VascularGraph.
    OUTPUT: None
    """
    G.es['volume'] = [np.pi * e['diameter']**2 / 4.0 * e['length'] 
                      for e in G.es]
        

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def update_depth(G):
    """Updates the depth property of all edges in the VascularGraph (adds it,
    if it does not exist).
    INPUT: G: VascularGraph.
    OUTPUT: None
    """
    G.es['depth'] = [np.mean([G.vs[x[0]]['r'][2], G.vs[x[1]]['r'][2]]) 
                     for x in G.get_edgelist()]
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def intersection_plane_line(pP,nP,pL,vL):
    """Computes the intersection of a plane and a line
    INPUT: pPlane: point on plane
           nPlane: normal vector of plane
           pLine: point on line
           vLine: vecotr on line
    OUTPUT: coordinates of intersectionPoint
    """

    plane = lambda x1,x2,x3: (x1-pP[0])*nP[0]+(x2-pP[1])*nP[1]+(x3-pP[2])*nP[2]

    iP=Symbol('iP')

    #intersection 
    iP=solve(plane(pL[0]+iP*vL[0],pL[1]+iP*vL[1],pL[2]+iP*vL[2]),iP)

    #Compute intersection point
    Point = lambda iP: [pL[0]+iP*vL[0],pL[1]+iP*vL[1],pL[2]+iP*vL[2]]

    if iP != []:
        coordsPoint = Point(iP[0])
    else:
        coordsPoint = []

    newCoordsPoint=[]
    for coords in coordsPoint:
        newCoordsPoint.append(np.float(coords))

    return np.array(newCoordsPoint)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def make_pointGraph_based_on_points(pL,G):
    """creates a point graph with the points given (for visualization in paraview)
    INPUT: pL: point list
           G: main Graph
    OUTPUT: new graph of points
    """

    Gnew=vgm.VascularGraph(len(pL))
    r=[]
    indexOrig=[]
    for i in pL:
        r.append(G.vs[int(i)]['r'])
        indexOrig.append(G.vs[int(i)]['indexOrig'])

    Gnew.vs['r']=r
    Gnew.vs['indexOrig']=indexOrig

    return Gnew
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def make_graph_based_on_points(eL,G):
    """creates a graph with the edgeList given 
    INPUT: eL: edge list
           G: main Graph
    OUTPUT: new graph of points
    """

    vertices=[]
    edgeTuples=[]
    for e in G.es[eL]:
        vertices.append(e.source)
        vertices.append(e.target)
        edgeTuples.append(e.tuple)

    vertices=np.unique(vertices)

    Gnew=vgm.VascularGraph(len(vertices))
    r=[]
    indexOrig=[]
    for i in vertices:
        r.append(G.vs[i]['r'])
        indexOrig.append(i)

    Gnew.vs['r']=r
    Gnew.vs['indexOrig']=indexOrig

    edgeTuplesNew=[]
    for tupleCurrent in edgeTuples:
        edgeTuplesNew.append([Gnew.vs(indexOrig_eq=tupleCurrent[0]).indices[0],Gnew.vs(indexOrig_eq=tupleCurrent[1]).indices[0]])

    Gnew.add_edges(edgeTuplesNew)
    Gnew.es['indexOrig']=eL

    for e in eL:
        eNew=Gnew.es(indexOrig_eq=e).indices[0]
        for attribute in G.es.attributes():
            Gnew.es[eNew][attribute]=G.es[e][attribute]

    for v in vertices:
        vNew=Gnew.vs(indexOrig_eq=v).indices[0]
        for attribute in G.vs.attributes():
            Gnew.vs[vNew][attribute]=G.vs[v][attribute]

    return Gnew
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_edges_in_sphere(G,centerSphere,radiusSphere,nkinds,radiusSphereMin=0):
    """ Returns a list of edges which lie in the specified sphere. The function 
    uses the tortuous vessel properties (if one of the points of the vessel is 
    located inside the sphere the vessel is considered to be in the sphere)
    INPUT: G: main Graph
           centerSphere: the coordinates of the center of the sphere
           radiusSphere: the radius of the center of the sphere
           nkinds: list of vessel kinds which should be considered
           radiusSphereMin: default = 0, a value between [0,radiusSphere] can be
           given. vessels inside the sphereMin are not considered
    OUTPUT: edges: list of edges where at least one point along the edge is
            located inside the sphere
    """
    #Get tortuous values of all edges of interest
    rAll=[]
    edgeAll=[]
    for nkind in nkinds:
        for e in G.es(nkind_eq=nkind):
            for i in e['points']:
                rAll.append(i)
                edgeAll.append(e.index)

    Kdt = kdtree.KDTree(rAll, leafsize=10)
    kAll=100
    nearestAll = Kdt.query(centerSphere,k=kAll)
    nearestDist=nearestAll[0]
    nearestIndex=nearestAll[1]
    largestDist=nearestDist[-1]
    while largestDist < radiusSphere:
        kAll += 100
        nearestAll = Kdt.query(centerSphere,k=kAll)
        nearestDist=nearestAll[0]
        nearestIndex=nearestAll[1]
        largestDist=nearestDist[-1]

    edges=[]
    for i,j in zip(nearestIndex,nearestDist):
        if edgeAll[i] not in edges and j < radiusSphere and j>=radiusSphereMin:
            edges.append(edgeAll[i])
    
    return edges
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_edges_intersecting_with_plane(G,pP,nP,nkinds):
    """ Returns a list of edges which intersect with a given plane of interest ().
    The normal vector of the plane has to be parallel to one of the axes of our
    coordinate system.
    The intersection Points of each edge are also returned.
    INPUT: G: main Graph
           pP: coordinated on the plane of interest
           nP: normal vector of the plane (has to be [1,0,0],[0,1,0] or [0,0,1])
           nkinds: list of vessel kinds which should be considered
    OUTPUT: edges: list of edges which intersect with the plane of interest
            intersectionCoords: coordinates of the intersectionPoints
    """

    nP=list(nP)
    if int(nP[0]) == 1:
        case=0 #normal vector in x direction
    elif int(nP[1]) == 1:
        case=1 #normal vector in y direction
    elif int(nP[2]) == 1:
        case=2 #normal vector in z direction

    edges=[]
    intersectionCoords=[]
    for nkind in nkinds:
        for e in G.es(nkind_eq=nkind):
            if e['points'][0][case]==pP[case]:
                edges.append(e.index)
                intersectionCoords.append(e['points'][j])
            for j in range(len(e['points'])-1):
                if (e['points'][j][case] > pP[case] and e['points'][j+1][case] < pP[case]) or (e['points'][j][case] < pP[case] and e['points'][j+1][case] > pP[case]):
                      pL=e['points'][j]
                      nL=e['points'][j]-e['points'][j+1]
                      coords=intersection_plane_line(pP,nP,pL,nL)
                      edges.append(e.index)
                      intersectionCoords.append(coords)
                elif e['points'][j+1][case]==pP[case]:
                    edges.append(e.index)
                    intersectionCoords.append(e['points'][j+1])

    return edges, intersectionCoords
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def planePlots_paraview(G,edges,intersectionCoords,attribute,filename,interpMethod='linear',gridpoints=100):
    """ Interpolates values on a grid based on the intersection of edges with a plane.
    It outputs a .vtp and a .pkl file to be read into paraview. It returns the grid and the 
    interpolated values at those locations (those can be processed to produce a contour plot in matplotlib)
    INPUT: G: main Graph
           edges: edges which intersect with the plane of interst
           intersectionCoords: coordinates intersection points
           attribute: that should be plotted
           filename: filename (including path) of the .vtp  and .pkl file 
               which will be saved (without file type extension)
           interpMethod: the interpolation method of scipy.interpolate.griddata
               common choices: 'nearest','linear' (default),'cubic'
           gridpoints: number of gridpoints (default = 100)
    OUTPUT: .pkl and .vtp file is written
           grid_d1,grid_d2: grid values
           valuesGrid: interpolated attribute values for the grid
           case: integer to define if the normal vector is in x- (case=1),
               y- (case=2) or in z-direction (case=3)
    """

    values=G.es[edges][attribute]
    x=[]
    y=[]
    z=[]
    for coord in intersectionCoords:
        x.append(coord[0])
        y.append(coord[1])
        z.append(coord[2])
    
    if len(np.unique(x)) < len(np.unique(y)) and len(np.unique(x)) < len(np.unique(z)):
        case=1
        d1Min=np.min(y)
        d1Max=np.max(y)
        d2Min=np.min(z)
        d2Max=np.max(z)
        d1=y
        d2=z
    elif len(np.unique(y)) < len(np.unique(x)) and len(np.unique(y)) < len(np.unique(z)):
        case=2
        d1Min=np.min(x)
        d1Max=np.max(x)
        d2Min=np.min(z)
        d2Max=np.max(z)
        d1=x
        d2=z
    elif len(np.unique(z)) < len(np.unique(y)) and len(np.unique(z)) < len(np.unique(x)):
        case=3
        d1Min=np.min(x)
        d1Max=np.max(x)
        d2Min=np.min(y)
        d2Max=np.max(y)
        d1=x
        d2=y

    grid_d1,grid_d2=np.mgrid[d1Min:d1Max:(d1Max-d1Min)/100,d2Min:d2Max:(d2Max-d2Min)/100]
    valuesGrid=griddata(zip(d1,d2),values,(grid_d1,grid_d2),method=interpMethod)
    
    r=[]
    for coord1 in np.arange(d1Min,d1Max,(d1Max-d1Min)/100):
        for coord2 in np.arange(d2Min,d2Max,(d2Max-d2Min)/100):
            if case == 1:
                r.append(np.array([x[0],coord1,coord2]))
            elif case == 2:
                r.append(np.array([coord1,y[0],coord2]))
            elif case == 3:
                r.append(np.array([coord1,coord2,z[0]]))
    
    valuesGridList=[]
    for j in valuesGrid:
        for k in j:
            valuesGridList.append(k)
    
    planeG=vgm.VascularGraph(len(r))
    planeG.vs['r']=r
    planeG.vs[attribute]=valuesGridList
    vgm.write_vtp(planeG,filename+'.vtp',False)
    vgm.write_pkl(planeG,filename+'.pkl')

    return grid_d1,grid_d2,valuesGrid,case

#------------------------------------------------------------------------------
def make_axis_labels(minVal,maxVal,factor=1,considerLimits=1):
    """ Creates the labels for an axis. based on the min and max value provided.
    INPUT: minVal: minimum Value of the data
            maxVal: maximum Value of the date
            factor: factor between the actual values and the strings provided 
                (e.g factor = 0.001 --> value 1000 --> string '1.0')
            considerLimits: bool if the labels should be trimmed at the lower/upper bound
    OUTPUT: labels,labelsString: list with the location of the labels and the according strings
            limits: lower and upper limit for the axis
    """
    minNumberOfLabelsIn=2
    maxNumberOfLabelsIn=5
    possibleSteps=[1.,2.,5.]
    minVal=np.float(minVal)
    maxVal=np.float(maxVal)

    preferenceList=[]
    for j in possibleSteps:
        if (maxVal-minVal)/j > minNumberOfLabelsIn:
            multiplier=0
            while (maxVal-minVal)/(j*10**(multiplier)) >= minNumberOfLabelsIn:
                multiplier += 1
            stepSize=j*10**(multiplier-1)
        elif (maxVal-minVal)/j < maxNumberOfLabelsIn:
            multiplier=0
            while (maxVal-minVal)/(j*10**(multiplier)) <= maxNumberOfLabelsIn:
                multiplier += -1
            stepSize=j*10**(multiplier+1)
        if np.ceil((maxVal-minVal)/stepSize) <= maxNumberOfLabelsIn:
            preferenceList.append(stepSize)
        else:
            preferenceList.append(np.nan)

    if preferenceList.index(np.nanmin(preferenceList)) == 0:
        stepSize=preferenceList[0]
    elif preferenceList.index(np.nanmin(preferenceList)) == 1:
        stepSize=preferenceList[1]
    elif preferenceList.index(np.nanmin(preferenceList)) == 2:
        stepSize=preferenceList[2]

    start=np.floor(minVal/stepSize)*stepSize
    labels=[]
    currentLabel = start
    labels.append(currentLabel)
    #while currentLabel-maxVal < -1*finfo(float).eps: 
    while currentLabel-maxVal < 0:
        currentLabel += stepSize
        labels.append(currentLabel)
    
    if considerLimits:
        limits=[labels[0],labels[-1]]
        if minVal > labels[1]-0.5*stepSize:
            limits[0] = labels[1]-0.5*stepSize
            labels=labels[1::]
        if maxVal < labels[-1]-0.5*stepSize:
            limits[1] = labels[-1]-0.5*stepSize
            labels=labels[0:-1]
    else:
        limits=[labels[0],labels[-1]]
        #limits=[np.nan,np.nan]

    if np.floor(np.log10(stepSize*factor)) == 0 or np.floor(np.log10(stepSize*factor)) == 1 or np.floor(np.log10(stepSize*factor)) == 2 \
        or np.floor(np.log10(stepSize*factor)) == 3 or np.floor(np.log10(stepSize*factor)) == 4:
        labelsString=['%.0f' %(label*factor) for label in labels]
    elif np.floor(np.log10(stepSize*factor)) == 5 or np.floor(np.log10(stepSize*factor)) == 6 or np.floor(np.log10(stepSize*factor)) == 7 \
        or np.floor(np.log10(stepSize*factor)) == 8 or np.floor(np.log10(stepSize*factor)) == 9:
        labelsString=['%.1e' %(label*factor) for label in labels]
    elif np.floor(np.log10(stepSize*factor)) == -1:
        decimals=-1*np.floor(np.log10(stepSize))
        labelsString=['%.1f' %(label*factor) for label in labels]
    elif np.floor(np.log10(stepSize*factor)) == -2:
        decimals=-1*np.floor(np.log10(stepSize))
        labelsString=['%.2f' %(label*factor) for label in labels]
    elif np.floor(np.log10(stepSize*factor)) == -3:
        decimals=-1*np.floor(np.log10(stepSize))
        labelsString=['%.3f' %(label*factor) for label in labels]
    else:
        print('WARNING Not yet defined. Has to be implemented first')
        print(np.log10(stepSize*factor))
        labelsString=[]

    return labels,labelsString,limits
#------------------------------------------------------------------------------
def assign_edges_to_layers(G,numberOfLayers=5,layerThickness=200,nkind=None):
    """ Creates the labels for an axis. based on the min and max value provided.
    INPUT: minVal: minimum Value of the data
            maxVal: maximum Value of the date
            factor: factor between the actual values and the strings provided 
                (e.g factor = 0.001 --> value 1000 --> string '1.0')
            considerLimits: bool if the labels should be trimmed at the lower/upper bound
    OUTPUT: labels,labelsString: list with the location of the labels and the according strings
            limits: lower and upper limit for the axis
    """

    layers=[]
    for i in range(numberOfLayers):
        layers.append([])
    
    pointsAll = np.concatenate(G.es['points'], 0)
    pointsCumSum = np.cumsum([len(p) for p in G.es['points']])
    for i,p in enumerate(pointsAll):
        edgeIndex=np.nonzero(pointsCumSum > i)[0][0]
        if nkind == None:
            layerIndex=int(np.floor(p[2]/layerThickness))
            if layerIndex < 0:
                layerIndex = 0
            elif layerIndex > numberOfLayers-1:
                layerIndex = numberOfLayers-1
            layers[layerIndex].append(edgeIndex)    
        else:
            if G.es[edgeIndex]['nkind']==nkind:
                layerIndex=int(np.floor(p[2]/layerThickness))
                if layerIndex < 0:
                    layerIndex = 0
                elif layerIndex > numberOfLayers-1:
                    layerIndex = numberOfLayers-1
                layers[layerIndex].append(edgeIndex)    
    
    layersNew=[]
    for i in range(numberOfLayers):
        layersNew.append(np.unique(layers[i]).tolist())

    return layersNew

#------------------------------------------------------------------------------
def assign_edges_to_layers_RBCpathsBased(G,numberOfLayers=5,layerThickness=200,nkind=None):
    """ Creates the labels for an axis. based on the min and max value provided.
    INPUT: minVal: minimum Value of the data
            maxVal: maximum Value of the date
            factor: factor between the actual values and the strings provided 
                (e.g factor = 0.001 --> value 1000 --> string '1.0')
            considerLimits: bool if the labels should be trimmed at the lower/upper bound
    OUTPUT: labels,labelsString: list with the location of the labels and the according strings
            limits: lower and upper limit for the axis
    """

    layers=[]
    for i in range(numberOfLayers):
        layers.append([])
   
    for e in G.es:
        if nkind == None:
            for v in e['capStarts']:
                layerIndex=int(np.floor(G.vs['r'][v][2]/layerThickness))
                if layerIndex < 0:
                    layerIndex = 0
                elif layerIndex > numberOfLayers-1:
                    layerIndex = numberOfLayers-1
                layers[layerIndex].append(edgeIndex)    
        else:
            for v in e['capStarts']:
                if G.es[edgeIndex]['nkind']==nkind:
                    layerIndex=int(np.floor(G.vs['r'][v][2]/layerThickness))
                    if layerIndex < 0:
                        layerIndex = 0
                    elif layerIndex > numberOfLayers-1:
                        layerIndex = numberOfLayers-1
                    layers[layerIndex].append(edgeIndex)    

    layersNew=[]
    for i in range(numberOfLayers):
        layersNew.append(np.unique(layers[i]).tolist())

    return layersNew
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def exchange_graph_purePython(G,eAttrs=[],vAttrs=[]):
    """ Adds angle to cortical surface (in degrees), cortical depth, volume and
    cross section to each edge in the graph.
    INPUT: G:  Vascular graph in iGraph format.
            eAttrs: edge attributes which should be stored (default is only the connectivity)
            vAttrs: vertex attributes which should be stored (default is only the vertex coordinates and the index)
    OUTPUT: edgesDic, verticesDict.pkl
    """

    edgesDict={}
    verticesDict={}

    verticesDict['index']=range(G.vcount())
    verticesDict['coords']=G.vs['r']
    for vA in vAttrs:
        verticesDict[vA]=G.vs[vA]

    connectivity=[]
    for e in G.es:
        connectivity.append(e.tuple)

    edgesDict['connectivity']=connectivity
    for eA in eAttrs:
        edgesDict[eA]=G.es[eA]

    with open('edgesDict.pkl','wb') as f:
        cPickle.dump(edgesDict,f,protocol=2)

    with open('verticesDict.pkl','wb') as f:
        cPickle.dump(verticesDict,f,protocol=2)

# -----------------------------------------------------------------------------
def export_for_microBlooM_csv(G,eAttrs=[],vAttrs=[],path='.'):
    """ Export csv-files suitable for the readNetwork functions in microBlooM. Length and diameter are scaled from um to m. All other units are kept. 
    INPUT: G:  Vascular graph in iGraph format.
            eAttrs: edge attributes which should be stored (default is connectivity, length and diameter. um input expected. converted to m.)
            vAttrs: vertex attributes which should be stored (default is vertex coordinates)
            path: path to save csv file can be provided. Default is '.'
    OUTPUT: node_data.csv, edge_data.csv, node_boundary_data.csv 
    """
    scale_to_meters = 1e-6
    
    vAttrs_all = ['x','y','z']+vAttrs
    f = open(path+'/node_data.csv','w')
    writer = csv.writer(f)
    writer.writerow(vAttrs_all)
    for v in G.vs:
        coords=v['r']
        row = [scale_to_meters*coords[0],scale_to_meters*coords[1],scale_to_meters*coords[2]]
        for attr in vAttrs:
            row.append(v[attr])
        writer.writerow(row)
    f.close()
    
    eAttrs_all = ['n1','n2','d','L']+eAttrs
    f = open(path+'/edge_data.csv','w')
    writer = csv.writer(f)
    writer.writerow(eAttrs_all)
    for e in G.es:
        row = [e.source, e.target, scale_to_meters*e['diameter'], scale_to_meters*e['length']]
        for attr in eAttrs:
            row.append(e[attr])
        writer.writerow(row)
    f.close()
    
    f = open(path+'/node_boundary_data.csv','w')
    writer = csv.writer(f)
    writer.writerow(['nodeID','boundaryType','p'])
    boundary_vertices = G.vs(pBC_ne=None).indices
    for v in boundary_vertices:
        row = [v,1,G.vs[v]['pBC']]
        writer.writerow(row)
    boundary_vertices = G.vs(rBC_ne=None).indices
    for v in boundary_vertices:
        row = [v,2,G.vs[v]['rBC']]
        writer.writerow(row)
    f.close()
    

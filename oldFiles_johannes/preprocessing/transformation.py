from __future__ import division

from numpy.linalg import norm
import scipy as sp
import numpy as np

from vgm import units
from vgm import g_math
from copy import deepcopy
    
__all__ = ['invert_z', 'adjust_cortical_depth', 'shift', 'scale',
           'rotate_using_axis_and_angle', 'rotate_using_two_vectors',
           'rotate_using_quaternion']

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
               

def invert_z(G):
    """ Inverts the z-value of the verticies of the vascular graph in iGraph
    format. 
    INPUT: G:  Vascular graph in iGraph format.
    OUTPUT: None - the vascular graph G is modified in place.
    WARNING: This currently does not take care of the edge parameters points, 
             angle, depth!
    """
    
    maxZ = 0
    for vertex in G.vs:
        if vertex['r'][2] > maxZ:
            maxZ = vertex['r'][2]
    for vertex in G.vs:            
        vertex['r'][2] = maxZ - vertex['r'][2]


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def adjust_cortical_depth(G,deltaZ):
    """Adjusts the z-component of the vertices and points of the vascular 
    graph.
    INPUT: G: Vascular graph in iGraph format.
           deltaZ: Amount by which to shift the data.
    OUTPUT: None - the graph is modified in-place.
    """

    for vertex in G.vs:
        vertex['r'] = vertex['r'] + (0,0,deltaZ)
        
    if 'points' in G.es.attribute_names():
        for edge in G.es:
            edge['points'] = edge['points'] + (0,0,deltaZ)
    
    if 'depth' in G.es.attribute_names():
        for edge in G.es:
            edge['depth'] = edge['depth'] + deltaZ
            

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def shift(G, offset):
    """Shifts (moves) the geometrical properties of the vascular graph by a
    given offset. Currently, this includes the vertex properties 'r' and 
    'depth', as well as the edge propery 'points'.
    
    INPUT: G: Vascular graph in iGraph format.
           offset: Offset as list [x,y,z] by which the graph is to be shifted.
    OUTPUT: None, graph is modified in place.
    """
    
    # Shift vertex properties:
    G.vs['r'] = [x + offset for x in G.vs['r']]    
    if 'depth' in G.vs.attribute_names():
        G.vs['depth'] = [x + offset for x in G.vs['depth']]

    # Shift edge properties:        
    if 'points' in G.es.attribute_names():
        for e in G.es:
            e['points'] = [x + offset for x in e['points']]    

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def scale(G, scalingFactor,initialVertex):
    """scales vascular graph by a given scaling factor. Currently, this includes the vertex properties 'r', 
    and the edge properties 'points' and 'lengths'.
    INPUT: G: Vascular graph in iGraph format.
           offset: Offset as list [x,y,z] by which the graph is to be shifted.
    OUTPUT: None, graph is modified in place.
    """
  
    checkV=[initialVertex]
    alreadyDone=[]
    eAlreadyDone=[]

    Gnew=deepcopy(G)
    
    while checkV != []:
        checkV2=[]
        for i in checkV:
            for j,k in zip(G.neighbors(i),G.adjacent(i)):
                if j not in alreadyDone:
                    vector=G.vs['r'][j] - G.vs['r'][i]
                    Gnew.vs[j]['r'] = Gnew.vs['r'][i] + scalingFactor * vector
                    checkV2.append(j)
                if k not in eAlreadyDone:
                    points=[]
                    for m in G.es[k]['points']:
                        vector=m - G.vs['r'][i]
                        points.append(Gnew.vs['r'][i] + scalingFactor * vector)
                    Gnew.es[k]['points'] = points
            alreadyDone.append(i)
        checkV = deepcopy(checkV2)
    
    #improve length of edge attribute lengths
    for e in Gnew.es:
        lengths=[]
        for i in range(len(e['points'])):
            if i == 0:
                lengths.append(np.linalg.norm(e['points'][1]-e['points'][0])/2)
            elif i == len(e['points']) -1:
                lengths.append(np.linalg.norm(e['points'][-1]-e['points'][-2])/2)
            else:
                lengths.append(0.5*(np.linalg.norm(e['points'][i]-e['points'][i-1])+np.linalg.norm(e['points'][i+1]-e['points'][i])))
        e['lengths']=lengths

    dists=[]
    for j in Gnew.vs['r']:
        dists.append(np.linalg.norm(j[:2] - Gnew.vs[int(initialVertex)]['r'][:2]))
    
    Gnew['spanningR']=np.max(dists)

    return Gnew


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def rotate_using_axis_and_angle(G, axis, angle, cor=(0.,0.,0.)):
    """Rotates the VascularGraph around a given axis.
    INPUT: G: Vascular graph in iGraph format.
           axis: Axis around which to rotate as array.
           angle: Angle by which to rotate in radians.
           cor: Center of roation as array.
    OUTPUT: None, the VascularGraph is modified in-place.       
    """
    Quat = g_math.Quaternion.from_rotation_axis(angle, axis)
    rotate_using_quaternion(G, Quat, cor)
        
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def rotate_using_two_vectors(G, vFrom, vTo, cor=(0.,0.,0.)):
    """Rotates the VascularGraph according to the rotation of one vector to 
    match another vector.
    INPUT: G: Vascular graph in iGraph format.
           vFrom: The vector of the initial orientation.
           vTo: The vector of the final orientation after rotation.
           cor: Center of roation as array.
    OUTPUT: None, the VascularGraph is modified in-place.       
    """
    Quat = g_math.Quaternion.from_two_vectors(vFrom, vTo)
    rotate_using_quaternion(G, Quat, cor)
        
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def rotate_using_quaternion(G, Quat, cor=(0.,0.,0.)):
    """Rotates the VascularGraph using a quaternion.
    INPUT: G: Vascular graph in iGraph format.
           Quat: Quaternion that defines the rotation.
           cor: Center of roation as array.
    OUTPUT: None, the VascularGraph is modified in-place.       
    """
    # Coordinate transform to center of rotation
    if not all(np.array(cor) == np.zeros(3)):
        shift(G, -cor)
    # Rotation
    for v in G.vs:
        v['r'] = Quat.rotate(v['r'])
    for e in G.es:
        e['points'] = np.array(map(Quat.rotate, e['points']))
    # Coordinate transform back to original origin    
    if not all(np.array(cor) == np.zeros(3)):
        shift(G, cor)
        
            

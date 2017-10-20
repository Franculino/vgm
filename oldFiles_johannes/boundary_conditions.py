from __future__ import division

from g_input import read_landmarks
from misc import vertices_from_coordinates
import numpy as np
from physiology import Physiology
import vgm

__all__ = ['find_pBC_vertices_using_landmarks', 'add_pBCs', 'add_missing_pBCs']
log = vgm.LogDispatcher.create_logger(__name__)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def find_pBC_vertices_using_landmarks(G,aLandmarks,vLandmarks,method='topDown',
                                      diameter_ll=7.0, isEndpoint=False,
                                      **kwargs):
    """Vascular networks that include the pial vessels are used to identify 
    arteries and veins. The locations where penetrating arterioles / draining
    veins leave / enter the cortical surface network are saved in landmark 
    files. This function reads the landmark files and finds the corresponding 
    vertices of the VascularGraph, i.e. those vertices that connect to the 
    surface network. The indices of these vertices can later be used to set 
    e.g. pressure boundary conditions.
    INPUT: G: Vascular graph in iGraph format
           aLandmarks: Name of landmark-file containing arterial vertex coords.
           vLandmarks: Name of landmark-file containing venous vertex coords.
           method: The search method of finding graph vertices that match the 
                   landmarks. This can be either 'exact', 'zNull', or 
                   'topDown'. The first takes the landmarks coordinates as is,
                   the second sets the z-component of the landmarks to zero, 
                   the third starts searching at the lowest possible z-value 
                   and continues to higher z-values until a match is found. 
                   The setting 'topDown' requires the keyword-argument 
                   'acceptableDistance' to be defined.
           zNull: Boolean, optional. If True (the default), will set the 
                  z-value of the landmark coordinates to zero - helping to 
                  adjust the offset between uncropped samples that include the 
                  pial vessels (in which the landmarks were acquired), and 
                  cropped samples that don't.         
           diameter_ll: (Optional) lower limit of edge diameter, to select only
                        those vertices bordering a sufficiently large diameter
                        edge. Default is 7.0, i.e. only vertices with adjacent
                        edges >= 7.0 are considered.
           isEndpoint: Boolean whether or not the pBC vertices are all                      
                       endpoints. Default is 'False'.
           **kwargs
           acceptableDistance: The maximum distance between landmark and vertex
                               which is acceptable for a match. (50 micron 
                               appears to be a good value.)
           zMax: The maximum z-value (i.e. cortical depth) to which the vertex
                 search should be performed in 'topDown' mode.
    OUTPUT: aVertices: Arterial pBC vertices.
            vVertices: Venous pBC vertices.
            Note that G is not modified.
    """

    class DataCapsule(object):
        def __init__(self, landmarks, vesselName):
            self.landmarks = landmarks
            self.vesselName = vesselName
            self.vertices = []
            
    DCA = DataCapsule(aLandmarks, 'artery')
    DCV = DataCapsule(vLandmarks, 'vein')
    
    for DC in [DCA, DCV]:
        vertexCoordinates = read_landmarks(DC.landmarks)
        numberOfLandmarks = len(vertexCoordinates)
        if method == 'zNull':
            vertexCoordinates = [(x[0],x[1],0.0) for x in vertexCoordinates]
        elif method == 'topDown':
            zCoords = [r[2] for r in G.vs['r']]
            zMin = min(zCoords)
            if kwargs.has_key('zMax'):
                zMax = kwargs['zMax']
            else:
                zMax = max(zCoords)
            vertexCoordinates = [(x[0],x[1],zMin) for x in vertexCoordinates]
            
        vertices, distances = vertices_from_coordinates(G,vertexCoordinates,
                                                       diameter_ll, isEndpoint)
        
        if kwargs.has_key('acceptableDistance'):
            ad = kwargs['acceptableDistance']
            v_done = []
            d_done = []
            v_todo = []
            c_todo = []        
            for z in zip(vertices, distances, vertexCoordinates):
                if z[1] <= ad:
                    v_done.append(z[0])
                    d_done.append(z[1])
                else:
                    v_todo.append(z[0])
                    c_todo.append(z[2])
                    
        if method == 'topDown':
            currentDepth = zMin + ad * 2.0
            
            while len(v_todo) > 0 and currentDepth <= zMax:        
                vertexCoordinates = [(x[0],x[1],currentDepth) for x in c_todo]
                vertices, distances = vertices_from_coordinates(G,
                                     vertexCoordinates,diameter_ll, isEndpoint)
                v_todo = []
                c_todo = []
                for z in zip(vertices, distances, vertexCoordinates):
                    if z[1] <= ad:
                        v_done.append(z[0])
                        d_done.append(z[1])
                    else:
                        v_todo.append(z[0])
                        c_todo.append(z[2])
                currentDepth = currentDepth + ad * 2.0
        
        if kwargs.has_key('acceptableDistance'):
            vertices = v_done
            distances = d_done
                        
        log.info("Found %i matching vertices from %i landmarks." \
                 % (len(vertices), numberOfLandmarks))
        log.info("Maximum %s vertex-landmark distance is %f." \
                 % (DC.vesselName, max(distances)))
        if len(vertices) != len(np.unique(vertices)):
            log.warning("Vertex duplicates exist!")
        DC.vertices = vertices
               
    return DCA.vertices, DCV.vertices


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def add_pBCs(G,kind,vertices):
    """Adds pressure boundary conditions to the vascular graph. Pressure values
    are taken from literature (see function 'blood_pressure').
    The pressure boundary vertices recieve a kind tag of either 'a' or 'v' 
    to classify them as arteries or veins respectively.
    INPUT: G: Vascular graph in iGraph format.
           kind: The vertex kind. This can be either 'a' for arterial or 'v' 
                 for venous.
           vertices: The indices of the vertices for which the pressure 
                     boundary conditions are to be set.
    OUTPUT: G is modified in place.
    """    

    P = Physiology(G['defaultUnits'])
    
    for vertex in vertices:
        diameter = max([G.es[x]['diameter'] for x in G.adjacent(vertex,'all')])
        G.vs[vertex]['pBC'] = P.blood_pressure(diameter,kind)
        G.vs[vertex]['kind'] = kind
        

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def add_missing_pBCs(G, pressure=0.0):
    """Ensures that all components of the graph have a minimum of one pressure 
    boundary condition (this includes unconnected nodes).
    INPUT: G: Vascular graph in iGraph format.
           pressure: (Optional, default=0.0) Pressure value to be set at the 
                     previously unassigned components.
    OUTPUT: None - the graph is modified in-place!
    """

    if not G.vs[0].attributes().has_key('pBC'):
        G.vs[0]['pBC'] = None
    for component in G.components():
        if not any(G.vs(component)['pBC']):
            G.vs[component[0]]['pBC'] = pressure
            

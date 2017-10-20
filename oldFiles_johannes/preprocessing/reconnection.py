from __future__ import division, with_statement

from copy import deepcopy
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import kdtree
from scipy.special import erf
from vgm import g_math
import operator
import vgm

__all__ = ['reconnect_cf', 'reconnect_tr']
log = vgm.LogDispatcher.create_logger(__name__)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def reconnect_cf(G, r_search=70, costCutoff=1.0, border=10.,
                 allowMultiple=True, A=0.6, n=1.4, loop_diameter=80.,
                 **kwargs):
    """Improves the connectivity of a vascular graph by reconnecting dead ends.
       Suitable connection partners are found based on the following 
       cost-function (the individual parameters are explained below):
       Cost=A * distance / r_search +
            (1-A) * (1 - exp(-n angle / pi)) * (1-erf(distance / loop_diameter)
    INPUT: G: VascularGraph.
           r_search: The radius within which potential connection candidates are
                     searched for.
           costCutoff: Cost above which potential new edges are discarded.
           border: Distance to edge of domain within which endpoins are 
                   ignored.
           allowMultiple: The endpoints are treated sequentially. This boolean
                          decides whether an endpoint is deleated from the list
                          of endpoints once a connection has been made to it, 
                          or whether it should stay available for more
                          connections from other endpoints.
           A: Distance cost factor. The angle cost is (1-A), such that the 
              maximum cost possible is 1.0.
           n: Angle cost modulation factor.
           loop_diameter: The typical diameter of a vessel loop (beyond this 
                          distance, the influence of the connection angle is
                          negligible).

           **kwargs
           connType: The type of connection to be made between end-points. This
                     can be either 'linear' or 'spline'. (Default 'spline'.)
           splineOrder: The order of the splines to be fitted. (Default 3.)
           numEdgePoints: The number of edge points to create the spline 
                          (taking many points into account will yield a fit 
                          that matches the overall shape of the two 
                          ending-edges well, but may not produce a good fit at 
                          the endpoint vertices. The smaller the number of edge 
                          points, the better the match with the endpoint 
                          vertices, however the total number of points to
                          fit must always exceed the order of the spline).
                          (Default: (spline order + 1) / 2)
           endpoints: A list of selected endpoint vertices to reconnect (useful
                      for testing purposes). All other enpoints serve only as
                      potential connection partners.
    OUTPUT: None, the VascularGraph is modified in-place       
    """    
    if kwargs.has_key('connType'):
        connType = kwargs['connType']
    else:
        connType = 'spline'

    if kwargs.has_key('splineOrder'):
        k = kwargs['splineOrder']
    else:
        k = 3

    if kwargs.has_key('numEdgePoints'):
        nepFit = kwargs['numEdgePoints']
    else:
        nepFit = int(np.ceil((k + 1) / 2))        
    

    G.vs['degree'] = G.degree()
    degreeOneVertices = G.vs(degree_eq=1).indices

    if 'endpoints' in kwargs.keys():
        endpointIndices = kwargs['endpoints']
    else:
        # exclude vertices close to the domain-boundaries from search:
        insideVertices = G.vertices_away_from_border(border)
        endpointIndices = [v for v in degreeOneVertices if v in insideVertices]

    # include all vertices of degree one as possible connection partners:
    Kdt = kdtree.KDTree(G.vs(degreeOneVertices)['r'], leafsize=10)
    
    notConnected = []
    nep = len(endpointIndices)
    adjlist = G.get_adjlist()
    todo = deepcopy(endpointIndices)
    
    G.es['cost'] = [0.0 for e in G.es]
    
    log.info('Attempting to reconnect %i endpoints' % nep)
    while len(todo) > 0:
        index = todo[0]        
        neighborIndices = Kdt.query_ball_point(G.vs[index]['r'], r_search)
        neighbors = [degreeOneVertices[x] for x in neighborIndices]
        connectedNeighbor = adjlist[index][0]
        edge = G.es[G.get_eid(index,connectedNeighbor)] # takes first edge, 
                                                        # if multiedge        
        
        while index in neighbors:   # multiple connections and self-loops
            neighbors.remove(index) # possible.
        while connectedNeighbor in neighbors:
            neighbors.remove(connectedNeighbor)
        if len(neighbors) == 0:
            todo.remove(index)
            continue            
        
        # Assert that sequence points away from endpoint:
        points = deepcopy(edge['points'])
        if index > connectedNeighbor:
            points = points[::-1,:]

        # Use a length of about two diameters (less for shorter edges) to 
        # determine the edge direction:
        nop = max(np.ceil(edge['diameter'] * 2. / edge['length'] * len(points)), 2)
        points = points[:min(len(points), nop)]

        direction1 = vgm.average_path_direction(points)
        #direction1 = g_math.pca(points) # points away from the endpoint
        
        costs = []
        for neighbor in neighbors:
            # Compute edge direction analogous to above:
            nNeighbor = adjlist[neighbor][0]
            nEdge = G.es[G.get_eid(neighbor, nNeighbor)]
            points = deepcopy(nEdge['points'])            
            if neighbor > nNeighbor:
                points = points[::-1,:]
            nop = max(2, np.ceil(nEdge['diameter'] * 2. / nEdge['length'] * len(points)))
            log.debug('l: %.1f, d: %.1f, np: %i, nop: %i\n' % (nEdge['length'], nEdge['diameter'], len(nEdge['points']), nop))
            points = points[:min(len(points), nop)]
            #points = points[:min(len(points), 5)]
            direction2 = vgm.average_path_direction(points)
            #direction2 = g_math.pca(points) # points away from the endpoint                     
            d =  np.linalg.norm(G.vs[neighbor]['r'] - G.vs[index]['r'])
            q = np.dot(direction1,direction2) # Magnitudes of both directions one
            if q > 1 or q < -1: # deal with numerical inaccuracies
                q = np.sign(q) * 1.
            angle = np.pi - np.arccos(q) # If arccos(q) is 180 deg, edges are parallel
            
            costs.append(A     * d/r_search + 
                         (1-A) * (1 - np.exp(-n*angle/np.pi)) *
                                 (1 - erf(d / loop_diameter)))
        if min(costs) > costCutoff:
            todo.remove(index)
            continue            

        # Connect index to bestMatch, if the connection has not been established
        # from the other side already:    
        bestMatch = neighbors[np.nonzero([costs == min(costs)])[1][0]]
        if bestMatch not in G.neighbors(index):
            bmNeighbor = adjlist[bestMatch][0] # From initial adjlist
            bmEdge = G.es[G.get_eid(bestMatch,bmNeighbor)]
            eIndex = G.ecount()
            G.add_edges([(index, bestMatch)])
            G.es[eIndex]['cost'] = min(costs)
            diameter = np.mean([edge['diameter'], G.es[bmEdge.index]['diameter']]) 
            G.es[eIndex]['diameter'] = diameter

            if connType == 'linear':
                points = np.array([(G.vs[index]['r'] + 
                         (G.vs[bestMatch]['r'] - G.vs[index]['r']) * 
                         x).tolist() 
                         for x in np.arange(0,1.1,0.1)])

            elif connType == 'spline':
                pointsOfIndex = deepcopy(edge['points'])
                # Assert that points move towards index:
                if index < connectedNeighbor:
                    pointsOfIndex = pointsOfIndex[::-1,:] 
                pointsOfBestMatch = deepcopy(bmEdge['points'])
                # Assert that points move away from bestMatch:                
                if bestMatch > bmNeighbor: 
                    pointsOfBestMatch = pointsOfBestMatch[::-1,:]

                pointsToFit = np.vstack([pointsOfIndex[- min(len(pointsOfIndex), nepFit):], 
                                         pointsOfBestMatch[:min(len(pointsOfBestMatch), nepFit)]])
                # spline parameters
                s=3.0 # smoothness parameter
                # spline order k is set above.
                nest=-1 # estimate of number of knots needed (-1 = maximal)                
                # find the knot points:
                tckp,u = splprep([pointsToFit[:,0],
                                  pointsToFit[:,1],
                                  pointsToFit[:,2]],s=s,k=k,nest=-1)                
                # evaluate spline, including interpolated points:
                xnew,ynew,znew = splev(np.linspace(0,1,400),tckp)    
                spline = zip(xnew,ynew,znew)
                # create kdTree from spline-points and find the two 
                # corresponding to index and bestMatch:
                KdtSpline = kdtree.KDTree(spline, leafsize=10)
                croppedSplineA = KdtSpline.query(G.vs[index]['r'])[1]
                croppedSplineB = KdtSpline.query(G.vs[bestMatch]['r'])[1]
                points = spline[croppedSplineA:croppedSplineB+1]
                points.insert(0, G.vs[index]['r'])
                points.append(G.vs[bestMatch]['r'])                
                points = np.array(points)
                                
            else:
                raise ValueError
                
            # points need to reflect that iGraph orders edge vertices:            
            if index > bestMatch:
                points = points[::-1,:]
            G.es[eIndex]['points'] = deepcopy(points)        
            G.es[eIndex]['diameters'] = np.array([diameter for x in points])            
            # Assign a length to each point (i.e.: #lengths == #points):
            l_edge = [0.0]
            for i in xrange(len(points)-1):
                l_edge.append(np.linalg.norm(points[i]-points[i+1]))
            l_edge.append(0.0)
            l_edge = [(l_edge[i-1] + l_edge[i]) / 2.0 
                      for i in xrange(1,len(l_edge))]
            G.es[eIndex]['lengths'] = np.array(l_edge)
            G.es[eIndex]['length'] = sum(l_edge)
            
            if not allowMultiple:
                if bestMatch in todo:
                    todo.remove(bestMatch)            

        todo.remove(index)    
    
    # Enforce a coordination number of 3 for all original endpoints:
    restrict_coordination_number(G, 3, degreeOneVertices)    

    del G.vs['degree']                     
    degree = G.degree()                     
    notConnected = [v for v in endpointIndices if degree[v] == 1]
    
    log.info('%i of %i endpoints were reconnected (%.2f %s)' % (\
          nep - len(notConnected), nep, 
          (nep - len(notConnected)) / nep * 100, '%'))
     
     
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def restrict_coordination_number(G, limit, vertexList):
    """Loop over the given endpoints and ensure that each has a coordination 
    number number less or equal to 'limit'. The edges with the highest cost are 
    removed first. 
    Note that in this algorithm the order of the endpoint vertices has an 
    influence on the outcome. Also, it may leave some of the previously
    connected endpoints unconnected.
    INPUT: limit: The maximum coordination number allowed.
           vertexList: The indices of the vertices to be inspected.
    OUTPUT: None, G is modified in-place.
    """
    G.vs['degree'] = G.degree()
    vertexList = G.vs(vertexList, degree_gt=limit).indices
    for vertex in vertexList:
        edges = G.adjacent(vertex)
        ces = sorted(zip(G.es[edges]['cost'], edges))
        nRemove = len(edges) - limit
        deleteEdges = [x[1] for x in ces[:nRemove]]
        G.delete_edges(deleteEdges)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------        
        
        
        
        
def reconnect_tr(G_ht, G_lt, r_search=20., r_match=3.):
    """Improves the connectivity of a vascular graph by reconnecting dead ends.
       Suitable connection partners are found based on threshold relaxation. 
    INPUT: G_ht: VascularGraph to be reconnected.
           G_lt: VascularGraph of low threshold (i.e. high connectivity, 
                 potentially many artifacts)
           r_search: The radius within which potential connection candidates are
                     searched for.
           r_match: 
    OUTPUT: None, the VascularGraph is modified in-place
"""

    G_ht.vs['degree'] = G_ht.degree()
    endpoints = G_ht.vs(degree_eq=1)
    
    log.info("Setting up distance trees for fast search...")
    # Distance trees for low and high threshold graphs:
        # ht, all branch points, possibly exclude edges of the sample:
    Kdt_ht = kdtree.KDTree(G_ht.vs['r'], leafsize=10)     
        # lt, all points of the tortuous edges:
    edgepointCoords = np.vstack(G_lt.es['points'])
    Kdt_lt = kdtree.KDTree(edgepointCoords, leafsize=10)
    
    log.info("Generating point-to-edge lookup table for the lt_graph...")
    # Lookup-table with point-to-edge entries for the low threshold graph:
    lut = {}
    for e in G_lt.es:
        lut.update({}.fromkeys(map(tuple,e['points']),e.index))
    
    log.info("Connecting endpoints...")
    edges_to_be_deleted = []
    nConnections = 0
    tenPercentStep = round(0.1 * len(endpoints))
    for i, ep in enumerate(endpoints):        
        if not np.mod(i,tenPercentStep) and i != 0:
            log.info("%i %% done %i connections made" % \
                  (round(i/tenPercentStep*10), nConnections))
        # Find the edge of G_lt, to which the current G_ht endpoint lies:
        # (Note that this assumes it only belongs to one edge.)
        distance, point = Kdt_lt.query(ep['r']) 
        if distance > r_match:
            continue
        edge = lut[tuple(edgepointCoords[point])]

        # Get all points that make up the edge and place them in a kd-tree:
        edgePoints = G_lt.es[edge]['points']
        Kdt_ep = kdtree.KDTree(edgePoints, leafsize=10)
        
        # Find potential neighbors ("candidatePoints") of the current endpoint:
        candidatePoints = Kdt_ht.query_ball_point(ep['r'],r=r_search)
        candidatePoints.remove(ep.index)
        
        # Abort search if no suitable neighbors are found:
        neighbors = G_ht.neighbors(ep.index)
        if (len(neighbors) > 1) or (len(neighbors) == 0):
            continue # Ths endpoint has either been connected previously or is
                     # an island point.
        else:
            neighbor = neighbors[0]        
        try:
            candidatePoints.remove(neighbor)
            # only has one neighbor, since it is endpoint            
        except:
            pass
        if len(candidatePoints) == 0:
            continue
        distances, points = Kdt_ep.query(G_ht.vs(candidatePoints)['r'])
        dsp, startPoint = Kdt_ep.query(ep['r'])
        dn, pn = Kdt_ep.query(G_ht.vs(neighbor)['r'])
        if dn > r_match:
            continue
            
        # Determine which of the candidatePoints is the most suitable:                
        # This concept may fail if "point" belongs to multiple edges and an edge
        # was chosen that "neighbor" is not a part of.        
        sequenceDistances = map(lambda x: x-startPoint,points)
        absSequenceDistances = map(lambda x: abs(x-startPoint),points)
        filteredCandidates = filter(lambda x: (x[0]<=r_match) and
                                   (np.sign(x[1]) != np.sign(pn-startPoint)), 
                                    zip(distances,
                                        sequenceDistances,
                                        absSequenceDistances,
                                        points,
                                        xrange(len(points))))
        if filteredCandidates == []:
            continue
        bestCandidate = sorted(filteredCandidates,key=operator.itemgetter(2))[0]
        newNeighbor = candidatePoints[bestCandidate[4]]      
        stopPoint = bestCandidate[3]
        
        
        # Make sure that the edge taken from G_lt and the ones in G_ht do not
        # overlap (make changes to G_ht, if required):
        bcEdges = G_ht.adjacent(candidatePoints[bestCandidate[4]])
        Kdt_ep = kdtree.KDTree(edgePoints[min(startPoint,bestCandidate[3]):
                                          max(startPoint,bestCandidate[3])+1], 
                               leafsize=10)
        subEdge_offset = min(startPoint,bestCandidate[3])        
        for bcEdge in bcEdges:
            bcEdgePoints = G_ht.es[bcEdge]['points']
            distances, points = Kdt_ep.query(bcEdgePoints)
            if len([x for x in distances if x<r_match]) > 10:
                filteredCandidates = filter(lambda x: (x[0]<=r_match),
                                     zip(distances,
                                         points,
                                         xrange(len(points))))
                if startPoint < bestCandidate[3]:                         
                    bestCandidate =  sorted(filteredCandidates,
                                            key=operator.itemgetter(1))[0]                    
                else:                            
                    bestCandidate =  sorted(filteredCandidates,
                                            key=operator.itemgetter(1))[-1]
                                            
                newNeighbor = G_ht.vcount()    
                G_ht.add_vertices(1)
                G_ht.vs[newNeighbor]['r'] = bcEdgePoints[bestCandidate[2]]
                nnPoint = bestCandidate[2]
                eDiameters = G_ht.es[bcEdge]['diameters']
                ePoints = G_ht.es[bcEdge]['points']
                G_ht.add_edges([(newNeighbor,G_ht.es[bcEdge].source),
                                (newNeighbor,G_ht.es[bcEdge].target)])
                edges_to_be_deleted.append(bcEdge)                                
                numEdges = G_ht.ecount()                
                G_ht.es[numEdges-2]['diameters'] = eDiameters[:nnPoint+1]
                G_ht.es[numEdges-2]['diameter'] = \
                                             np.mean(eDiameters[:nnPoint+1])                
                G_ht.es[numEdges-2]['points'] = ePoints[:nnPoint+1]
                # source always < target and is < newNeighbor. 
                # No change required.
                
                
                G_ht.es[numEdges-1]['diameters'] = eDiameters[nnPoint:]
                G_ht.es[numEdges-1]['diameter'] = \
                                             np.mean(eDiameters[nnPoint:])
                G_ht.es[numEdges-1]['points'] = ePoints[nnPoint:]
                # target always > source but < newNeighbor. 
                # Need to flip points and diameters:
                G_ht.es[numEdges-1]['diameters'] = \
                  G_ht.es[numEdges-1]['diameters'][::-1]
                G_ht.es[numEdges-1]['points'] = \
                  G_ht.es[numEdges-1]['points'][::-1]
                stopPoint = bestCandidate[1] + subEdge_offset
                continue
        
        
        # Connect the current endpoint with the bestCandidate:
        nConnections += 1
        newEdge = G_ht.ecount()
        G_ht.add_edges((ep.index, newNeighbor)) 
        start, stop = np.sort([startPoint, stopPoint])        
        newPoints = G_lt.es[edge]['points'][start:stop+1]
        newDiameters = G_lt.es[edge]['diameters'][start:stop+1]
        newDiameter = np.mean(newDiameters)
        if ep.index > newNeighbor:
            source = newNeighbor
            target = ep.index
            if startPoint < stopPoint:
                newPoints = newPoints[::-1]
                newDiameters = newDiameters[::-1]
        else:
            source = ep.index
            target = newNeighbor
            if startPoint > stopPoint:
                newPoints = newPoints[::-1]
                newDiameters = newDiameters[::-1]
                
                
        G_ht.es[newEdge]['points'] = newPoints
        G_ht.es[newEdge]['points'][0] = G_ht.vs[source]['r']
        G_ht.es[newEdge]['points'][-1] = G_ht.vs[target]['r']        
        G_ht.es[newEdge]['diameters'] = newDiameters
        G_ht.es[newEdge]['diameter'] = newDiameter

    G_ht.delete_edges(edges_to_be_deleted) # keep lut valid till the end   
    G_ht.vs['degree'] = G_ht.degree()      # update all at once
    log.info("Process completed. %i new connections made." % (nConnections))


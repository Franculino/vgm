from __future__ import division                   
import numpy as np
from scipy.spatial import kdtree, ConvexHull
from copy import deepcopy

__all__ = ['find_vessels_in_barrel','define_branchingOrder_fromMainDA',
           'paths_between_barrelVessels_and_mainDA']

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def find_vessels_in_barrel(G,barrelG,barrelIndex):
    """ Finds all vessels which have at least one point located inside the specified barrel.
    Computes the total length of the vessels in the barrel.
    INPUT: G:  Vascular graph in iGraph format.
            barrelG: Outer coordinates of the barrels (set of points in igraph format)
    OUTPUT: list of edeges in barrel;
            barrelVolume;
            total length of vessels in the barrel; 
            total length of capillaries in the barrel
    """

    coordsActivation = barrelG.vs(barrelIndex_eq=barrelIndex,center_eq=1)['r'][0]
    maxBarrelRadius = np.max([np.linalg.norm(coords[0:2]-coordsActivation[0:2]) for coords in barrelG.vs(barrelIndex_eq=barrelIndex)['r']])
    meanBarrelRadius = np.mean([np.linalg.norm(coords[0:2]-coordsActivation[0:2]) for coords in barrelG.vs(barrelIndex_eq=barrelIndex)['r']])

    allPoints = np.concatenate(G.es['points'],axis=0)
    inBarrelBool = [0]*len(allPoints)
    allEdgeIndices = np.concatenate([[i]*len(G.es[i]['points']) for i in range(G.ecount())], axis=0)
    Kdt = kdtree.KDTree(allPoints, leafsize=10)
    allPointsInRadius = Kdt.query_ball_point(coordsActivation,maxBarrelRadius)

    barrel_xCoords = []; barrel_yCoords = []; barrel_zCoords = []
    for coords in barrelG.vs(barrelIndex_eq=barrelIndex)['r']:
        barrel_xCoords.append(coords[0])
        barrel_yCoords.append(coords[1])
        barrel_zCoords.append(coords[2])

    xMinBarrel = np.min(barrel_xCoords); xMaxBarrel = np.max(barrel_xCoords)
    yMinBarrel = np.min(barrel_yCoords); yMaxBarrel = np.max(barrel_yCoords)
    zMinBarrel = np.min(barrel_zCoords); zMaxBarrel = np.max(barrel_zCoords)

    barrelCoords_yx = zip(barrel_yCoords, barrel_xCoords)
    barrelCoords_yx.sort()
    barrel_yCoords, barrel_xCoords = zip(*barrelCoords_yx)

    allPointsInBoundingBox=[]
    for pI in allPointsInRadius:
        pI_coords = allPoints[pI]
        if pI_coords[0] >= xMinBarrel and pI_coords[0] <= xMaxBarrel and pI_coords[1] <= yMaxBarrel and pI_coords[1] >= yMinBarrel \
                and pI_coords[2] >= zMinBarrel and pI_coords[2] <= zMaxBarrel:
            allPointsInBoundingBox.append(pI)

    pointsInBarrel=[]
    edgesInBarrel=[]
    pointsInvestigated=[]
    for pI in allPointsInBoundingBox:
        pI_coords = allPoints[pI]
        higher_yValue_index_low = np.argmax(barrel_yCoords >= pI_coords[1])
        higher_yValue = barrel_yCoords[higher_yValue_index_low]
        if higher_yValue == barrel_yCoords[-1]:
            higher_yValue_index_high = len(barrel_yCoords)
        else:
            higher_yValue_index_high = np.argmax(barrel_yCoords > higher_yValue)-1
        if higher_yValue_index_low == 0:
            lower_yValue_index_high = 0
        else:
            lower_yValue_index_high = higher_yValue_index_low-1
        lower_yValue = barrel_yCoords[lower_yValue_index_high]
        lower_yValue_index_low = np.argmax(barrel_yCoords == lower_yValue)
        barrel_xCoords_ofInterest = np.array(barrel_xCoords[lower_yValue_index_low:higher_yValue_index_high+1])
        barrel_xCoords_ofInterest.sort()
        split_xValues_index = np.argmax(barrel_xCoords_ofInterest>np.mean(barrel_xCoords_ofInterest))
        xMin = np.mean(barrel_xCoords_ofInterest[0:split_xValues_index])
        xMax = np.mean(barrel_xCoords_ofInterest[split_xValues_index::])
        if pI_coords[0] >= xMin and pI_coords[0] <= xMax:
            pointsInBarrel.append(pI_coords)
            edgesInBarrel.append(allEdgeIndices[pI])
            inBarrelBool[pI] = 1
        pointsInvestigated.append(pI_coords)

    edgesInBarrel=np.unique(edgesInBarrel)

    totalVesselLength_inBarrel = 0
    totalCapillaryLength_inBarrel = 0

    for i in range(len(allPoints)-1): # inBarrelBool, allEdgeIndices):
        if inBarrelBool[i] == 1 and inBarrelBool[i+1] == 1 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            length = np.linalg.norm(allPoints[i]-allPoints[i+1])
            totalVesselLength_inBarrel += length
            if G.es[allEdgeIndices[i]]['nkind'] == 4:
                totalCapillaryLength_inBarrel += length

    hull = ConvexHull(barrelG.vs(barrelIndex_eq=barrelIndex)['r'])

    return edgesInBarrel, hull.volume, totalVesselLength_inBarrel, totalCapillaryLength_inBarrel

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def define_branchingOrder_fromMainDA(G,maxLength=150):
    """ Introduces the edge attribute branchingOrder. Branching order: branches apart from the mainDA. First branch 1.
    Only until branching order 3 because that's until where pericytes with SMA have been detected. All other vessels have the value -1.
    A length criterion is further introduced because otherwise too many vessel are possible. the length cirterion can be defined.
    The default is set to 150 um. This is based on grant et al. 2017 --> pericyte length * 3
    INPUT: G:  Vascular graph in iGraph format.
        maxLength: maximum distance from mainDA which is allowed to be considered as precapillary
    OUTPUT: G with the edge attribute branching order
    """

    G.es['branchingOrder']=[-1]*G.ecount()
    G.es['distanceFromMainDA']=[-1]*G.ecount()

    allMainDAedges = G.es(mainDA_eq=1).indices
    allMainDAvertices = np.concatenate([e.tuple for e in G.es[allMainDAedges]],axis=0)
    allMainDAvertices = np.unique(allMainDAvertices)

    edgesDone=[]
    for vDA in allMainDAvertices:
        vertexList = [vDA]
        vertexListNew=[]
        for i in range(3):
            for vBase in vertexList:
                for v,e in zip(G.neighbors(vBase),G.incident(vBase)):
                    if G.es[e]['mainDA'] != 1 and e not in edgesDone and G.es[e]['nkind'] > 1:
                        if i == 0:
                            G.es[e]['distanceFromMainDA'] = G.es[e]['length']
                            G.es[e]['branchingOrder']=i+1
                        else:
                            minLengthNeighbors = 9999999999999999
                            for v2,e2 in zip(G.neighbors(vBase),G.incident(vBase)):
                                if e2 != e and G.es[e2]['branchingOrder'] == i:
                                    if G.es[e2]['distanceFromMainDA'] < minLengthNeighbors:
                                        minLengthNeighbors = G.es[e2]['distanceFromMainDA']
                            if minLengthNeighbors + G.es[e]['length'] < maxLength:
                                G.es[e]['distanceFromMainDA'] = minLengthNeighbors + G.es[e]['length']
                                G.es[e]['branchingOrder']=i+1
                        edgesDone.append(e)
                        vertexListNew.append(v)
            vertexList = vertexListNew[:]
        vertexListNew = []

    return G


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def paths_between_barrelVessels_and_mainDA(G,edgesInBarrel):
    """ computes all paths between the capillaries in the barrel and the node of the first mainBranch of the upstream DAs.
    These can be used to assign the preCapillaries and the DAs which are to be dilated
    INPUT: G:  Vascular graph in iGraph format.
            edgesInBarrel: all vessels which are located in the barrel
    OUTPUT: pathsToDA_edges, pathsToDa_vertices and mainDAstarts, associated_edgeInNarrel
    """
    Gdummy=deepcopy(G)
    Gdummy.to_directed_flow_based()

    mainDAstarts=[]
    pathsToDA_edges=[]
    pathsToDA_vertices=[]
    associated_edgeInBarrel=[]
    for edge in edgesInBarrel:
        if G.es[edge]['mainDA'] != 1 and G.es[edge]['mainAV'] != 1 and G.es[edge]['diameter'] < 10:
            vertices=[Gdummy.es[edge].source]
            allPathsEdges=[[]]
            allPathsVertices=[vertices]
            finished=[0]
            mainDAbool=[0]
            boolChange=1
            while boolChange:
                boolChange=0
                allPathsEdgesNew=[]
                allPathsVerticesNew=[]
                mainDAboolNew=[]
                finishedNew=[]
                for currentPathV,currentPathE,boolFinished,mainDA in zip(allPathsVertices,allPathsEdges,finished,mainDAbool):
                    if not boolFinished:
                        if len(Gdummy.neighbors(currentPathV[-1],'in'))==0:
                            allPathsEdgesNew.append(currentPathE[:])
                            allPathsVerticesNew.append(currentPathV[:])
                            mainDAboolNew.append(0)
                            finishedNew.append(1)
                        for n,e in zip(Gdummy.neighbors(currentPathV[-1],'in'),Gdummy.incident(currentPathV[-1],'in')):
                            currentPathVNew=currentPathV[:]
                            currentPathENew=currentPathE[:]
                            if G.es[e]['mainDA'] != 1:
                                currentPathVNew.append(n)
                                currentPathENew.append(e)
                                mainDAboolNew.append(0)
                                finishedNew.append(0)
                                boolChange=1
                            else:
                                mainDAboolNew.append(1)
                                mainDAstarts.append(e)
                                finishedNew.append(1)
                            allPathsEdgesNew.append(currentPathENew)
                            allPathsVerticesNew.append(currentPathVNew)
                    else:
                        allPathsEdgesNew.append(currentPathE[:])
                        allPathsVerticesNew.append(currentPathV[:])
                        mainDAboolNew.append(mainDA)
                        finishedNew.append(1)
                allPathsVertices=deepcopy(allPathsVerticesNew)
                allPathsEdges=deepcopy(allPathsEdgesNew)
                finished=finishedNew[:]
                mainDAbool=mainDAboolNew[:]
            pathsToDA_edges =pathsToDA_edges + allPathsEdges
            pathsToDA_vertices =pathsToDA_vertices + allPathsVertices
            associated_edgeInBarrel = associated_edgeInBarrel + [edge]*len(allPathsEdges)
    #Check for similar paths
    duplicates=[]
    for i,path in enumerate(pathsToDA_edges):
        if i not in duplicates:
            for j in range(i+1,len(pathsToDA_edges)):
                path2=pathsToDA_edges[j]
                if path == path2:
                    duplicates.append(j)

    duplicates.sort()
    for j in duplicates[::-1]:
        del(pathsToDA_edges[j])
        del(pathsToDA_vertices[j])
        del(associated_edgeInBarrel[j])
    mainDAstarts=np.unique(mainDAstarts)

    return pathsToDA_edges, pathsToDA_vertices, mainDAstarts, associated_edgeInBarrel
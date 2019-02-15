from __future__ import division                   
import numpy as np
from scipy.spatial import kdtree, ConvexHull
from copy import deepcopy

__all__ = ['find_vessels_in_barrel','find_vessels_in_barrel_with_coordinateLimits','define_branchingOrder_fromMainDA',
           'find_vessels_in_slice','paths_between_barrelVessels_and_mainDA','compute_inflow_into_barrel',
           'compute_nRBC_in_barrel','compute_connectivity_in_barrel']

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def find_vessels_in_barrel(G,barrelG,barrelIndex):
    """ Finds all vessels which have at least one point located inside the specified barrel.
    Computes the total length of the vessels in the barrel.
    INPUT: G:  Vascular graph in iGraph format.
            barrelG: Outer coordinates of the barrels (set of points in igraph format)
            barrelIndex: barrel that is under investigation
    OUTPUT: list of edeges in barrel;
            barrelVolume;
            [total length of vessels in the barrel, total volume of vessels in the barrel]; 
            [total length of capillaries in the barrel, total volume of capillaries in the barrel];
            [total length of pre-capillaries in the barrel, total volume of pre-capillaries in the barrel];
            [total length of mainDA in the barrel, total volume of mainDA in the barrel];
            [total length of mainAV in the barrel, total volume of mainAV in the barrel];
    """

    coordsActivation = barrelG.vs(barrelIndex_eq=barrelIndex,center_eq=1)['r'][0]
    maxBarrelRadius = np.max([np.linalg.norm(coords[0:2]-coordsActivation[0:2]) for coords in barrelG.vs(barrelIndex_eq=barrelIndex)['r']])

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

    edgesInBarrel=[]
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
            edgesInBarrel.append(allEdgeIndices[pI])
            inBarrelBool[pI] = 1

    edgesInBarrel=np.unique(edgesInBarrel)

    totalVesselLength_inBarrel = 0
    totalCapillaryLength_inBarrel = 0
    totalPreCapillaryLength_inBarrel = 0
    totalVesselVolume_inBarrel = 0
    totalCapillaryVolume_inBarrel = 0
    totalPreCapillaryVolume_inBarrel = 0
    totalMainDALength_inBarrel = 0
    totalMainAVLength_inBarrel = 0
    totalMainDAVolume_inBarrel = 0
    totalMainAVVolume_inBarrel = 0

    allEdgeIndicesDummy=allEdgeIndices[:]
    allEdgeIndicesDummy.sort()
    if np.any(allEdgeIndices != allEdgeIndicesDummy):
        print('ERROR')

    for i in range(len(allPoints)-1): # inBarrelBool, allEdgeIndices):
        if inBarrelBool[i] == 1 and inBarrelBool[i+1] == 1 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            length = np.linalg.norm(allPoints[i]-allPoints[i+1])
            totalVesselLength_inBarrel += length
            totalVesselVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                totalCapillaryLength_inBarrel += length
                totalCapillaryVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] == 1:
                totalMainDALength_inBarrel += length
                totalMainDAVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainAV'] == 1:
                totalMainAVLength_inBarrel += length
                totalMainAVVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and \
                    G.es[allEdgeIndices[i]]['diameter'] < 14 and G.es[allEdgeIndices[i]]['branchingOrder'] > 0:
                totalPreCapillaryLength_inBarrel += length
                totalPreCapillaryVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
        elif inBarrelBool[i] == 1 and inBarrelBool[i+1] == 0 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            length = np.linalg.norm(allPoints[i]-allPoints[i+1])
            totalVesselLength_inBarrel += length
            totalVesselVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                totalCapillaryLength_inBarrel += length
                totalCapillaryVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] == 1:
                totalMainDALength_inBarrel += length
                totalMainDAVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainAV'] == 1:
                totalMainAVLength_inBarrel += length
                totalMainAVVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and \
                    G.es[allEdgeIndices[i]]['diameter'] < 14 and G.es[allEdgeIndices[i]]['branchingOrder'] > 0:
                totalPreCapillaryLength_inBarrel += length
                totalPreCapillaryVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
        elif inBarrelBool[i] == 0 and inBarrelBool[i+1] == 1 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            length = np.linalg.norm(allPoints[i]-allPoints[i+1])
            totalVesselLength_inBarrel += length
            totalVesselVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                totalCapillaryLength_inBarrel += length
                totalCapillaryVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] == 1:
                totalMainDALength_inBarrel += length
                totalMainDAVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainAV'] == 1:
                totalMainAVLength_inBarrel += length
                totalMainAVVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and \
                    G.es[allEdgeIndices[i]]['diameter'] < 14 and G.es[allEdgeIndices[i]]['branchingOrder'] > 0:
                totalPreCapillaryLength_inBarrel += length
                totalPreCapillaryVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2

    hull = ConvexHull(barrelG.vs(barrelIndex_eq=barrelIndex)['r'])

    return edgesInBarrel, hull.volume, [totalVesselLength_inBarrel, totalVesselVolume_inBarrel], \
            [totalCapillaryLength_inBarrel, totalCapillaryVolume_inBarrel], [totalPreCapillaryLength_inBarrel, totalPreCapillaryVolume_inBarrel], \
            [totalMainDALength_inBarrel, totalMainDAVolume_inBarrel], [totalMainAVLength_inBarrel, totalMainAVVolume_inBarrel]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def find_vessels_in_barrel_with_coordinateLimits(G,barrelG,barrelIndex,xLimit,yLimit,zLimit):
    """ Finds all vessels which have at least one point located inside the specified barrel.
    Computes the total length of the vessels in the barrel.
    NOTE: capillaries are defined as everything that is not 'mainDA' nor 'mainAV' and has a diameter < 10
    INPUT: G:  Vascular graph in iGraph format.
            barrelG: Outer coordinates of the barrels (set of points in igraph format)
            barrelIndex: barrel that is under investigation
            xLimit: [xMin,xMax] only vessels in those limits should be considered
            yLimit: [yMin,yMax] only vessels in those limits should be considered
            zLimit: [zMin,zMax] only vessels in those limits should be considered
    OUTPUT: list of edeges in barrel;
            barrelVolume;
            [total length of vessels in the barrel, total volume of vessels in the barrel]; 
            [total length of capillaries in the barrel, total volume of capillaries in the barrel];
            [total length of pre-capillaries in the barrel, total volume of pre-capillaries in the barrel];
            [total length of mainDA in the barrel, total volume of mainDA in the barrel];
            [total length of mainAV in the barrel, total volume of mainAV in the barrel];
    """

    coordsActivation = barrelG.vs(barrelIndex_eq=barrelIndex,center_eq=1)['r'][0]
    maxBarrelRadius = np.max([np.linalg.norm(coords[0:2]-coordsActivation[0:2]) for coords in barrelG.vs(barrelIndex_eq=barrelIndex)['r']])

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
    if zMinBarrel < zLimit[0]:
        zMinBarrel = zLimit[0]
    if zMaxBarrel > zLimit[1]:
        zMaxBarrel = zLimit[1]

    barrelCoords_yx = zip(barrel_yCoords, barrel_xCoords)
    barrelCoords_yx.sort()
    barrel_yCoords, barrel_xCoords = zip(*barrelCoords_yx)

    allPointsInBoundingBox=[]
    for pI in allPointsInRadius:
        pI_coords = allPoints[pI]
        if pI_coords[0] >= xMinBarrel and pI_coords[0] <= xMaxBarrel and pI_coords[1] <= yMaxBarrel and pI_coords[1] >= yMinBarrel \
                and pI_coords[2] >= zMinBarrel and pI_coords[2] <= zMaxBarrel:
            allPointsInBoundingBox.append(pI)

    edgesInBarrel=[]
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
        if pI_coords[0] >= xMin and pI_coords[0] <= xMax and pI_coords[0] > xLimit[0] and pI_coords[0] < xLimit[1]\
                and pI_coords[1] > yLimit[0] and pI_coords[1] < yLimit[1]:
            edgesInBarrel.append(allEdgeIndices[pI])
            inBarrelBool[pI] = 1

    edgesInBarrel=np.unique(edgesInBarrel)

    totalVesselLength_inBarrel = 0
    totalCapillaryLength_inBarrel = 0
    totalPreCapillaryLength_inBarrel = 0
    totalVesselVolume_inBarrel = 0
    totalCapillaryVolume_inBarrel = 0
    totalPreCapillaryVolume_inBarrel = 0
    totalMainDALength_inBarrel = 0
    totalMainAVLength_inBarrel = 0
    totalMainDAVolume_inBarrel = 0
    totalMainAVVolume_inBarrel = 0

    allEdgeIndicesDummy=allEdgeIndices[:]
    allEdgeIndicesDummy.sort()
    if np.any(allEdgeIndices != allEdgeIndicesDummy):
        print('ERROR')

    for i in range(len(allPoints)-1): # inBarrelBool, allEdgeIndices):
        if inBarrelBool[i] == 1 and inBarrelBool[i+1] == 1 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            length = np.linalg.norm(allPoints[i]-allPoints[i+1])
            totalVesselLength_inBarrel += length
            totalVesselVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                totalCapillaryLength_inBarrel += length
                totalCapillaryVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] == 1:
                totalMainDALength_inBarrel += length
                totalMainDAVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainAV'] == 1:
                totalMainAVLength_inBarrel += length
                totalMainAVVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and \
                    G.es[allEdgeIndices[i]]['diameter'] < 14 and G.es[allEdgeIndices[i]]['branchingOrder'] > 0:
                totalPreCapillaryLength_inBarrel += length
                totalPreCapillaryVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
        elif inBarrelBool[i] == 1 and inBarrelBool[i+1] == 0 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            length = np.linalg.norm(allPoints[i]-allPoints[i+1])
            totalVesselLength_inBarrel += length
            totalVesselVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                totalCapillaryLength_inBarrel += length
                totalCapillaryVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] == 1:
                totalMainDALength_inBarrel += length
                totalMainDAVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainAV'] == 1:
                totalMainAVLength_inBarrel += length
                totalMainAVVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and \
                    G.es[allEdgeIndices[i]]['diameter'] < 14 and G.es[allEdgeIndices[i]]['branchingOrder'] > 0:
                totalPreCapillaryLength_inBarrel += length
                totalPreCapillaryVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
        elif inBarrelBool[i] == 0 and inBarrelBool[i+1] == 1 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            length = np.linalg.norm(allPoints[i]-allPoints[i+1])
            totalVesselLength_inBarrel += length
            totalVesselVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                totalCapillaryLength_inBarrel += length
                totalCapillaryVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] == 1:
                totalMainDALength_inBarrel += length
                totalMainDAVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainAV'] == 1:
                totalMainAVLength_inBarrel += length
                totalMainAVVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and \
                    G.es[allEdgeIndices[i]]['diameter'] < 14 and G.es[allEdgeIndices[i]]['branchingOrder'] > 0:
                totalPreCapillaryLength_inBarrel += length
                totalPreCapillaryVolume_inBarrel += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2

    barrelPoints=[]
    countOutside = 0
    #barrelPoints_x=[]
    #barrelPoints_y=[]
    #barrelPoints_z=[]
    for coords in barrelG.vs(barrelIndex_eq=barrelIndex)['r']:
        boolChanged = 0
        coordsCurrent = deepcopy(coords)
        if coords[0] <= xLimit[0]:
            boolChanged = 1
            coordsCurrent[0]=xLimit[0]
        elif coords[0] >= xLimit[1]:
            boolChanged = 1
            coordsCurrent[0]=xLimit[1]
        if coords[1] <= yLimit[0]:
            boolChanged = 1
            coordsCurrent[1]=yLimit[0]
        elif coords[1] >= yLimit[1]:
            boolChanged = 1
            coordsCurrent[1]=yLimit[1]
        if coords[2] <= zLimit[0]:
            boolChanged = 1
            coordsCurrent[2]=zLimit[0]
        elif coords[2] >= zLimit[1]:
            boolChanged = 1
            coordsCurrent[2]=zLimit[1]
        barrelPoints.append(coordsCurrent)
        if boolChanged:
            countOutside += 1
        #boolAlreadyInList = 0
        #if coordsCurrent[0] in barrelPoints_x:
        #    for i in np.where(barrelPoints_x == barrelPoints_x[0])[0]:
        #        if coordsCurrent[1] == barrelPoints_y[i] and coordsCurrent[2] == barrelPoints_z[i]:
        #            boolAlreadyInList = 1
        #            break
        #if not boolAlreadyInList: 
        #    barrelPoints.append(coordsCurrent)
        #    barrelPoints_x.append(coordsCurrent[0])
        #    barrelPoints_y.append(coordsCurrent[1])
        #    barrelPoints_z.append(coordsCurrent[2])

    if countOutside == len(barrelG.vs(barrelIndex_eq=barrelIndex)):
        volume = 0
    else:
        hull = ConvexHull(barrelPoints)
        volume = hull.volume

    return edgesInBarrel, volume, [totalVesselLength_inBarrel, totalVesselVolume_inBarrel], \
            [totalCapillaryLength_inBarrel, totalCapillaryVolume_inBarrel], [totalPreCapillaryLength_inBarrel, totalPreCapillaryVolume_inBarrel],\
            [totalMainDALength_inBarrel, totalMainDAVolume_inBarrel], [totalMainAVLength_inBarrel, totalMainAVVolume_inBarrel]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def find_vessels_in_slice(G,xLimit,yLimit,zLimit):
    """ Finds all vessels which have at least one point located inside the specified barrel.
    Computes the total length of the vessels in the barrel.
    INPUT: G:  Vascular graph in iGraph format.
            xLimit: [xMin,xMax] only vessels in those limits should be considered
            yLimit: [yMin,yMax] only vessels in those limits should be considered
            zLimit: [zMin,zMax] only vessels in those limits should be considered
    OUTPUT: list of edeges in barrel;
            barrelVolume;
            [total length of vessels in the barrel, total volume of vessels in the barrel]; 
            [total length of capillaries in the barrel, total volume of capillaries in the barrel];
            [total length of pre-capillaries in the barrel, total volume of pre-capillaries in the barrel];
            [total length of mainDA in the barrel, total volume of mainDA in the barrel];
            [total length of mainAV in the barrel, total volume of mainAV in the barrel];
    """

    allPoints = np.concatenate(G.es['points'],axis=0)
    allEdgeIndices = np.concatenate([[i]*len(G.es[i]['points']) for i in range(G.ecount())], axis=0)
    Kdt = kdtree.KDTree(allPoints, leafsize=10)

    sliceCenter=np.array([np.mean(xLimit),np.mean(yLimit),np.mean(zLimit)])
    inPlaneRadius=np.sqrt((sliceCenter[0]-xLimit[0])**2 +(sliceCenter[1]-yLimit[0])**2)
    maxPlaneRadius=np.sqrt(inPlaneRadius**2+(sliceCenter[2]-zLimit[0])**2)
    inPlaneBool = [0]*len(allPoints)
    allPointsInRadius = Kdt.query_ball_point(sliceCenter,maxPlaneRadius)

    allPointsInPlane=[]
    edgesInPlane=[]
    inPlaneBool = [0]*len(allPoints)
    for pI in allPointsInRadius:
        pI_coords = allPoints[pI]
        if pI_coords[0] >= xLimit[0] and pI_coords[0] <= xLimit[1] and pI_coords[1] >= yLimit[0] and pI_coords[1] <= yLimit[1] \
                and pI_coords[2] >= zLimit[0] and pI_coords[2] <= zLimit[1]:
            edgesInPlane.append(allEdgeIndices[pI])
            inPlaneBool[pI] = 1

    edgesInPlane=np.unique(edgesInPlane)

    totalVesselLength_inPlane = 0
    totalCapillaryLength_inPlane = 0
    totalPreCapillaryLength_inPlane = 0
    totalVesselVolume_inPlane = 0
    totalCapillaryVolume_inPlane = 0
    totalPreCapillaryVolume_inPlane = 0
    totalMainDALength_inPlane = 0
    totalMainAVLength_inPlane = 0
    totalMainDAVolume_inPlane = 0
    totalMainAVVolume_inPlane = 0

    allEdgeIndicesDummy=allEdgeIndices[:]
    allEdgeIndicesDummy.sort()
    if np.any(allEdgeIndices != allEdgeIndicesDummy):
        print('ERROR')

    for i in range(len(allPoints)-1): # inBarrelBool, allEdgeIndices):
        if inPlaneBool[i] == 1 and inPlaneBool[i+1] == 1 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            length = np.linalg.norm(allPoints[i]-allPoints[i+1])
            totalVesselLength_inPlane += length
            totalVesselVolume_inPlane += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                totalCapillaryLength_inPlane += length
                totalCapillaryVolume_inPlane += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] == 1:
                totalMainDALength_inPlane += length
                totalMainDAVolume_inPlane += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainAV'] == 1:
                totalMainAVLength_inPlane += length
                totalMainAVVolume_inPlane += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2
            if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and \
                    G.es[allEdgeIndices[i]]['diameter'] < 14 and G.es[allEdgeIndices[i]]['branchingOrder'] > 0:
                totalPreCapillaryLength_inPlane += length
                totalPreCapillaryVolume_inPlane += length*0.25*np.pi*G.es[allEdgeIndices[i]]['diameter']**2

    volume = (xLimit[1]-xLimit[0])*(yLimit[1]-yLimit[0])*(zLimit[1]-zLimit[0])

    return edgesInPlane, volume, [totalVesselLength_inPlane, totalVesselVolume_inPlane], \
            [totalCapillaryLength_inPlane, totalCapillaryVolume_inPlane], [totalPreCapillaryLength_inPlane, totalPreCapillaryVolume_inPlane], \
            [totalMainDALength_inPlane, totalMainDAVolume_inPlane], [totalMainAVLength_inPlane, totalMainAVVolume_inPlane]

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
        vertices=[Gdummy.es[edge].source]
        allPathsEdges=[[]]
        allPathsVertices=[vertices]
        if G.es[edge]['mainDA'] != 1:
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
        pathsToDA_edges = pathsToDA_edges + allPathsEdges
        pathsToDA_vertices = pathsToDA_vertices + allPathsVertices
        associated_edgeInBarrel = associated_edgeInBarrel + [edge]*len(allPathsEdges)

    mainDAstarts=np.unique(mainDAstarts)

    return pathsToDA_edges, pathsToDA_vertices, mainDAstarts, associated_edgeInBarrel
# -----------------------------------------------------------------------------
def paths_between_barrelVessels_and_mainAV(G,edgesInBarrel):
    """ computes all paths between the capillaries in the barrel and the node of the first mainBranch of the upstream DAs.
    These can be used to assign the preCapillaries and the DAs which are to be dilated
    INPUT: G:  Vascular graph in iGraph format.
            edgesInBarrel: all vessels which are located in the barrel
    OUTPUT: pathsToAV_edges, pathsToAV_vertices and mainAVstarts, associated_edgeInNarrel
    """
    Gdummy=deepcopy(G)
    Gdummy.to_directed_flow_based()

    mainAVstarts=[]
    pathsToAV_edges=[]
    pathsToAV_vertices=[]
    associated_edgeInBarrel=[]
    for edge in edgesInBarrel:
        vertices=[Gdummy.es[edge].target]
        allPathsEdges=[[]]
        allPathsVertices=[vertices]
        if G.es[edge]['mainAV'] != 1:
            finished=[0]
            mainAVbool=[0]
            boolChange=1
            while boolChange:
                boolChange=0
                allPathsEdgesNew=[]
                allPathsVerticesNew=[]
                mainAVboolNew=[]
                finishedNew=[]
                for currentPathV,currentPathE,boolFinished,mainAV in zip(allPathsVertices,allPathsEdges,finished,mainAVbool):
                    if not boolFinished:
                        if len(Gdummy.neighbors(currentPathV[-1],'out'))==0:
                            allPathsEdgesNew.append(currentPathE[:])
                            allPathsVerticesNew.append(currentPathV[:])
                            mainAVboolNew.append(0)
                            finishedNew.append(1)
                        for n,e in zip(Gdummy.neighbors(currentPathV[-1],'out'),Gdummy.incident(currentPathV[-1],'out')):
                            currentPathVNew=currentPathV[:]
                            currentPathENew=currentPathE[:]
                            if G.es[e]['mainAV'] != 1:
                                currentPathVNew.append(n)
                                currentPathENew.append(e)
                                mainAVboolNew.append(0)
                                finishedNew.append(0)
                                boolChange=1
                            else:
                                mainAVboolNew.append(1)
                                mainAVstarts.append(e)
                                finishedNew.append(1)
                            allPathsEdgesNew.append(currentPathENew)
                            allPathsVerticesNew.append(currentPathVNew)
                    else:
                        allPathsEdgesNew.append(currentPathE[:])
                        allPathsVerticesNew.append(currentPathV[:])
                        mainAVboolNew.append(mainAV)
                        finishedNew.append(1)
                allPathsVertices=deepcopy(allPathsVerticesNew)
                allPathsEdges=deepcopy(allPathsEdgesNew)
                finished=finishedNew[:]
                mainAVbool=mainAVboolNew[:]
        pathsToAV_edges =pathsToAV_edges + allPathsEdges
        pathsToAV_vertices =pathsToAV_vertices + allPathsVertices
        associated_edgeInBarrel = associated_edgeInBarrel + [edge]*len(allPathsEdges)

    mainAVstarts=np.unique(mainAVstarts)

    return pathsToAV_edges, pathsToAV_vertices, mainAVstarts, associated_edgeInBarrel
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def paths_between_barrelCapillares_and_mainDA(G,edgesInBarrel):
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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def compute_inflow_into_barrel(G,barrelG,barrelIndex):
    """ Finds all vessels which have at least one point located inside the specified barrel.
    Computes the total length of the vessels in the barrel.
    INPUT: G:  Vascular graph in iGraph format.
            barrelG: Outer coordinates of the barrels (set of points in igraph format)
            barrelIndex: barrel that is under investigation
    OUTPUT: list of edeges in barrel;
            barrelVolume;
            [total inflow through all border vessels, total outflow through all border vessels]; 
            [total inflow through all capillary border vessels, total outflow through all capillary border vessels]; 
    """

    coordsActivation = barrelG.vs(barrelIndex_eq=barrelIndex,center_eq=1)['r'][0]
    maxBarrelRadius = np.max([np.linalg.norm(coords[0:2]-coordsActivation[0:2]) for coords in barrelG.vs(barrelIndex_eq=barrelIndex)['r']])

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

    edgesInBarrel=[]
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
            edgesInBarrel.append(allEdgeIndices[pI])
            inBarrelBool[pI] = 1

    edgesInBarrel=np.unique(edgesInBarrel)

    flow_into_Barrel = 0
    flow_outof_Barrel = 0
    capillaryFlow_into_Barrel = 0
    capillaryFlow_outof_Barrel = 0

    for i in range(len(allPoints)-1): # inBarrelBool, allEdgeIndices):
        if inBarrelBool[i] == 0 and inBarrelBool[i+1] == 1 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            if G.vs[G.es[allEdgeIndices[i]].source]['pressure'] <= G.vs[G.es[allEdgeIndices[i]].target]['pressure']:
                flow_outof_Barrel += G.es[allEdgeIndices[i]]['flow']
                if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                    capillaryFlow_outof_Barrel += G.es[allEdgeIndices[i]]['flow']
            else:
                flow_into_Barrel += G.es[allEdgeIndices[i]]['flow']
                if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                    capillaryFlow_into_Barrel += G.es[allEdgeIndices[i]]['flow']
        elif inBarrelBool[i] == 1 and inBarrelBool[i+1] == 0 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            if G.vs[G.es[allEdgeIndices[i]].source]['pressure'] > G.vs[G.es[allEdgeIndices[i]].target]['pressure']:
                flow_outof_Barrel += G.es[allEdgeIndices[i]]['flow']
                if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                    capillaryFlow_outof_Barrel += G.es[allEdgeIndices[i]]['flow']
            else:
                flow_into_Barrel += G.es[allEdgeIndices[i]]['flow']
                if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                    capillaryFlow_into_Barrel += G.es[allEdgeIndices[i]]['flow']

    hull = ConvexHull(barrelG.vs(barrelIndex_eq=barrelIndex)['r'])

    return edgesInBarrel, hull.volume,[flow_into_Barrel,flow_outof_Barrel], [capillaryFlow_into_Barrel, capillaryFlow_outof_Barrel] 

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def compute_nRBC_in_barrel(G,barrelG,barrelIndex):
    """ Finds all vessels which have at least one point located inside the specified barrel.
    Computes the total length of the vessels in the barrel.
    INPUT: G:  Vascular graph in iGraph format.
            barrelG: Outer coordinates of the barrels (set of points in igraph format)
            barrelIndex: barrel that is under investigation
    OUTPUT: list of edeges in barrel;
            barrelVolume;
            nRBC in Barrel
    """

    coordsActivation = barrelG.vs(barrelIndex_eq=barrelIndex,center_eq=1)['r'][0]
    maxBarrelRadius = np.max([np.linalg.norm(coords[0:2]-coordsActivation[0:2]) for coords in barrelG.vs(barrelIndex_eq=barrelIndex)['r']])

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

    edgesInBarrel=[]
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
            edgesInBarrel.append(allEdgeIndices[pI])
            inBarrelBool[pI] = 1

    edgesInBarrel=np.unique(edgesInBarrel)

    nRBC_perBarrel=0
    for i in range(len(allPoints)-1): # inBarrelBool, allEdgeIndices):
        length = np.linalg.norm(allPoints[i]-allPoints[i+1])
        #if inBarrelBool[i] != 0 or inBarrelBool[i+1] != 0:
        #    print('')
        #    print(i)
        #    print('inBarrelBool')
        #    print(inBarrelBool[i])
        #    print(inBarrelBool[i+1])
        #    print('allEdgeIndices')
        #    print(allEdgeIndices[i])
        #    print(allEdgeIndices[i+1])
        #    print('vesselData')
        #    print(currentEdge)
        #    print(lengthCurrentEdge)
        if inBarrelBool[i] == 0 and inBarrelBool[i+1] == 1 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            currentEdge = allEdgeIndices[i]
            lengthCurrentEdge = length
        elif inBarrelBool[i] == 0 and inBarrelBool[i+1] == 1 and allEdgeIndices[i] != allEdgeIndices[i+1]:
            currentEdge = allEdgeIndices[i+1]
            lengthCurrentEdge = 0
        elif inBarrelBool[i] == 1 and inBarrelBool[i+1] == 1 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            lengthCurrentEdge += length
        elif inBarrelBool[i] == 1 and inBarrelBool[i+1] == 1 and allEdgeIndices[i] != allEdgeIndices[i+1]:
            nRBC_perBarrel += G.es[currentEdge]['nRBC']*lengthCurrentEdge/G.es[currentEdge]['length']
            currentEdge = allEdgeIndices[i+1]
            lengthCurrentEdge = 0
        elif inBarrelBool[i] == 1 and inBarrelBool[i+1] == 0 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            lengthCurrentEdge += length
            nRBC_perBarrel += G.es[currentEdge]['nRBC']*lengthCurrentEdge/G.es[currentEdge]['length']
        elif inBarrelBool[i] == 1 and inBarrelBool[i+1] == 0 and allEdgeIndices[i] != allEdgeIndices[i+1]:
            nRBC_perBarrel += G.es[currentEdge]['nRBC']*lengthCurrentEdge/G.es[currentEdge]['length']

    hull = ConvexHull(barrelG.vs(barrelIndex_eq=barrelIndex)['r'])

    return edgesInBarrel, hull.volume, nRBC_perBarrel

# -----------------------------------------------------------------------------
def compute_connectivity_in_barrel(G,barrelG,barrelIndex):
    """ Finds all vessels which have at least one point located inside the specified barrel.
    Computes the total length of the vessels in the barrel.
    INPUT: G:  Vascular graph in iGraph format.
            barrelG: Outer coordinates of the barrels (set of points in igraph format)
            barrelIndex: barrel that is under investigation
    OUTPUT: list of edeges in barrel;
            barrelVolume;
            nRBC in Barrel
    """

    coordsActivation = barrelG.vs(barrelIndex_eq=barrelIndex,center_eq=1)['r'][0]
    maxBarrelRadius = np.max([np.linalg.norm(coords[0:2]-coordsActivation[0:2]) for coords in barrelG.vs(barrelIndex_eq=barrelIndex)['r']])

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

    edgesInBarrel=[]
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
            edgesInBarrel.append(allEdgeIndices[pI])
            inBarrelBool[pI] = 1

    edgesInBarrel=np.unique(edgesInBarrel)
    allEdges=range(G.ecount())
    for e in edgesInBarrel:
        allEdges.remove(e)

    Gtest=deepcopy(G)
    Gtest.delete_edges(allEdges) 
    Gtest.vs['degree']=Gtest.degree()
    Gtest.delete_vertices(Gtest.vs(degree_eq=0).indices)

    componentSizes=[]
    for i in range(len(Gtest.components())):
        componentSizes.append(len(Gtest.components()[i]))
    
    return componentSizes
# -----------------------------------------------------------------------------
def distance_vertex_DA(G,vertex):
    """ computes all paths between the capillaries in the barrel and the node of the first mainBranch of the upstream DAs.
    These can be used to assign the preCapillaries and the DAs which are to be dilated
    INPUT: G:  Vascular graph in iGraph format.
            vertex: vertex from which the distance to DA/AV will be computed
    OUTPUT: allDistancesToDA
    """
    Gdummy=deepcopy(G)
    Gdummy.to_directed_flow_based()

    pathsToDA_edges=[]
    pathsToDA_vertices=[]
    mainDA_allBools=[]
    edgesUpstream = Gdummy.incident(vertex,'in')
    for edge in edgesUpstream:
        if G.es[edge]['mainDA'] != 1 and G.es[edge]['mainAV'] != 1:
            vertices=[Gdummy.es[edge].source]
            allPathsEdges=[[edge]]
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
            mainDA_allBools = mainDA_allBools + mainDAbool

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
        del(mainDA_allBools[j])

    allDistancesToDA = []
    for path, mainDAbool in zip(pathsToDA_edges,mainDA_allBools):
        if mainDAbool:
            allDistancesToDA.append(np.sum(G.es[path]['length']))

    return allDistancesToDA
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def distance_vertex_AV(G,vertex):
    """ computes all paths between the capillaries in the barrel and the node of the first mainBranch of the upstream DAs.
    These can be used to assign the preCapillaries and the DAs which are to be dilated
    INPUT: G:  Vascular graph in iGraph format.
            vertex: vertex from which the distance to DA/AV will be computed
    OUTPUT: 
    """
    Gdummy=deepcopy(G)
    Gdummy.to_directed_flow_based()

    pathsToAV_edges=[]
    pathsToAV_vertices=[]
    mainAV_allBools=[]
    edgesDownstream = Gdummy.incident(vertex,'out')
    for edge in edgesDownstream:
        if G.es[edge]['mainDA'] != 1 and G.es[edge]['mainAV'] != 1:
            vertices=[Gdummy.es[edge].target]
            allPathsEdges=[[edge]]
            allPathsVertices=[vertices]
            finished=[0]
            mainAVbool=[0]
            boolChange=1
            while boolChange:
                boolChange=0
                allPathsEdgesNew=[]
                allPathsVerticesNew=[]
                mainAVboolNew=[]
                finishedNew=[]
                for currentPathV,currentPathE,boolFinished,mainAV in zip(allPathsVertices,allPathsEdges,finished,mainAVbool):
                    if not boolFinished:
                        if len(Gdummy.neighbors(currentPathV[-1],'out'))==0:
                            allPathsEdgesNew.append(currentPathE[:])
                            allPathsVerticesNew.append(currentPathV[:])
                            mainAVboolNew.append(0)
                            finishedNew.append(1)
                        for n,e in zip(Gdummy.neighbors(currentPathV[-1],'out'),Gdummy.incident(currentPathV[-1],'out')):
                            currentPathVNew=currentPathV[:]
                            currentPathENew=currentPathE[:]
                            if G.es[e]['mainAV'] != 1:
                                currentPathVNew.append(n)
                                currentPathENew.append(e)
                                mainAVboolNew.append(0)
                                finishedNew.append(0)
                                boolChange=1
                            else:
                                mainAVboolNew.append(1)
                                finishedNew.append(1)
                            allPathsEdgesNew.append(currentPathENew)
                            allPathsVerticesNew.append(currentPathVNew)
                    else:
                        allPathsEdgesNew.append(currentPathE[:])
                        allPathsVerticesNew.append(currentPathV[:])
                        mainAVboolNew.append(mainAV)
                        finishedNew.append(1)
                allPathsVertices=deepcopy(allPathsVerticesNew)
                allPathsEdges=deepcopy(allPathsEdgesNew)
                finished=finishedNew[:]
                mainAVbool=mainAVboolNew[:]
            pathsToAV_edges =pathsToAV_edges + allPathsEdges
            pathsToAV_vertices =pathsToAV_vertices + allPathsVertices
            mainAV_allBools = mainAV_allBools + mainAVbool

    #Check for similar paths
    duplicates=[]
    for i,path in enumerate(pathsToAV_edges):
        if i not in duplicates:
            for j in range(i+1,len(pathsToAV_edges)):
                path2=pathsToAV_edges[j]
                if path == path2:
                    duplicates.append(j)

    duplicates.sort()
    for j in duplicates[::-1]:
        del(pathsToAV_edges[j])
        del(pathsToAV_vertices[j])
        del(mainAV_allBools[j])

    allDistancesToAV = []
    for path, mainAVbool in zip(pathsToAV_edges,mainAV_allBools):
        if mainAVbool:
            allDistancesToAV.append(np.sum(G.es[path]['length']))

    return allDistancesToAV
# -----------------------------------------------------------------------------
def compute_inflow_into_barrel(G,barrelG,barrelIndex):
    """ Finds all vessels which have at least one point located inside the specified barrel.
    Computes the total length of the vessels in the barrel.
    INPUT: G:  Vascular graph in iGraph format.
            barrelG: Outer coordinates of the barrels (set of points in igraph format)
            barrelIndex: barrel that is under investigation
    OUTPUT: list of edeges in barrel;
            barrelVolume;
            [total inflow through all border vessels, total outflow through all border vessels]; 
            [total inflow through all capillary border vessels, total outflow through all capillary border vessels]; 
    """

    coordsActivation = barrelG.vs(barrelIndex_eq=barrelIndex,center_eq=1)['r'][0]
    maxBarrelRadius = np.max([np.linalg.norm(coords[0:2]-coordsActivation[0:2]) for coords in barrelG.vs(barrelIndex_eq=barrelIndex)['r']])

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

    edgesInBarrel=[]
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
            edgesInBarrel.append(allEdgeIndices[pI])
            inBarrelBool[pI] = 1

    edgesInBarrel=np.unique(edgesInBarrel)

    flow_into_Barrel = 0
    flow_outof_Barrel = 0
    capillaryFlow_into_Barrel = 0
    capillaryFlow_outof_Barrel = 0

    for i in range(len(allPoints)-1): # inBarrelBool, allEdgeIndices):
        if inBarrelBool[i] == 0 and inBarrelBool[i+1] == 1 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            if G.vs[G.es[allEdgeIndices[i]].source]['pressure'] <= G.vs[G.es[allEdgeIndices[i]].target]['pressure']:
                flow_outof_Barrel += G.es[allEdgeIndices[i]]['flow']
                if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                    capillaryFlow_outof_Barrel += G.es[allEdgeIndices[i]]['flow']
            else:
                flow_into_Barrel += G.es[allEdgeIndices[i]]['flow']
                if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                    capillaryFlow_into_Barrel += G.es[allEdgeIndices[i]]['flow']
        elif inBarrelBool[i] == 1 and inBarrelBool[i+1] == 0 and allEdgeIndices[i] == allEdgeIndices[i+1]:
            if G.vs[G.es[allEdgeIndices[i]].source]['pressure'] > G.vs[G.es[allEdgeIndices[i]].target]['pressure']:
                flow_outof_Barrel += G.es[allEdgeIndices[i]]['flow']
                if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                    capillaryFlow_outof_Barrel += G.es[allEdgeIndices[i]]['flow']
            else:
                flow_into_Barrel += G.es[allEdgeIndices[i]]['flow']
                if G.es[allEdgeIndices[i]]['mainDA'] != 1 and G.es[allEdgeIndices[i]]['mainAV'] != 1 and G.es[allEdgeIndices[i]]['diameter'] < 10:
                    capillaryFlow_into_Barrel += G.es[allEdgeIndices[i]]['flow']

    hull = ConvexHull(barrelG.vs(barrelIndex_eq=barrelIndex)['r'])

    return edgesInBarrel, hull.volume,[flow_into_Barrel,flow_outof_Barrel], [capillaryFlow_into_Barrel, capillaryFlow_outof_Barrel] 

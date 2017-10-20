from __future__ import division

import g_output
import pylab as pl
import vgm
import numpy as np
from sys import stdout
import time as ttime

    
__all__ = ['all_paths_between_two_vertices', 'all_paths_of_given_length',
           'shortest_path_between_two_vertices', 'paths_to_subgraph',
           'path_between_a_and_v_for_vertexList']
log = vgm.LogDispatcher.create_logger(__name__)
           
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def all_paths_between_two_vertices(G,v1,v2,max_path_length=25,direction='out'):
    """Finds all possible paths between two vertices of a graph. The paths can
    be directed or not directed, depending on the type of graph.
    INPUT: G: Vascular graph in iGraph format.
           v1: First vertex.
           v2: Second vertex.
           max_path_length: Maximum number of vertices (including v1 and v2) 
                            that a path may contain in order to be considered.
           direction: Way to traverse a directed graph. Can be either 'out', 
                      'in', or 'all'. Is ignored in undirected graphs.
    OUTPUT: paths: The possible paths between v1 and v2.
    """
    
    oldPaths = [[v1]]; newPaths = []; paths = []
    while oldPaths != []:
        log.info("Current number of search paths: %i" % (len(oldPaths)))
        for path in oldPaths:
            for neighbor in G.neighbors(path[-1],type=direction):
                if neighbor == v2:
                    paths.append(path + [v2])
                elif neighbor not in path:
                    if len(path)+1 < max_path_length:
                        newPaths.append(path + [neighbor])
        oldPaths = newPaths; newPaths = []
    
    return paths


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def all_paths_of_given_length(G,v,max_path_length=25,direction='out'):
    """Finds all posible paths from a given vertex that have a specific length.
    The paths can be directed or not directed, depending on the type of graph.
    INPUT: G: Vascular graph in iGraph format.
           v: Vertex from which the search is to be started. Alternatively, a
              tuple of two vertices can be supplied that indicates a starting
              edge and direction, rather than a single vertex.
           max_path_length: Maximum number of vertices (including v) that a path 
                            may contain in order to be considered.
           direction: Way to traverse a directed graph. Can be either 'out', 
                      'in', or 'all'. Is ignored in undirected graphs.                 
    OUTPUT: paths: The possible paths from v.
    """

    if type(v) == type([]):
        paths = [v]
    else:    
        paths = [[v]] 
    newPaths = []
    path_length = len(paths[0])
    while path_length < max_path_length:
        path_length += 1
        print('')
        print(path_length)
        print(len(paths))
        countDone = 0
        for path in paths:
            if G.neighbors(path[-1],type=direction) == []:
                newPaths.append(path)
                countDone += 1
            else:
                for neighbor in G.neighbors(path[-1],type=direction):
                    #if neighbor not in path:
                    newPaths.append(path + [neighbor])
        paths = newPaths
        newPaths = []
        log.info("Iteration %i: %i paths found" % (path_length, len(paths)))
        print(paths)
        if countDone == len(paths):
            break

    return paths
    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def path_between_a_and_v_for_vertexList(G,v,direction='out'):
    """Finds all posible paths from a given vertex that have a specific length.
    The paths can be directed or not directed, depending on the type of graph.
    INPUT: G: Vascular graph in iGraph format.
           v: Vertex List from which the search is to be started. 
           direction: Way to traverse a directed graph. Can be either 'out', 
                      'in', 'out' = from a to v and 'in' = from v to a
    OUTPUT: paths: The possible paths from v.
    """

    pathDict={}
    if direction == 'out':
        stopKind='v'
    else:
        stopKind='a'

    stopPaths=1e5
    stopPaths2=1e6
    indexHalf=np.floor(len(v)/9.)

    #for j in range(9):    
    for j in range(1):    
        pathDict={}
        print('ROUND')
        print(j)
        stdout.flush()
        if j == 9:
            indexStart=indexHalf*j
            indexEnd=len(v)
        else:
            indexStart=indexHalf*j
            indexEnd=indexHalf*(j+1)
        print(indexStart)
        print(indexEnd)
        #for i in v[int(indexStart):int(indexEnd)]:
        for i in v:
            print('')
            print(i)
            paths=[[i]]
            pathsEdges=[[]]
            boolContinue = 1
            newPaths = []
            newPathsEdges = []
            countLoop=0
            stdout.flush()
            while boolContinue:
                print(len(paths))
                stdout.flush()
                countDone=0
                countLoop += 1
                if countLoop > stopPaths:
                    print('Path length is getting too long')
                    break
                if len(paths) > stopPaths2:
                    print('Number of  Path is getting too large')
                    break
                for path,pathEdges in zip(paths,pathsEdges):
                    if G.neighbors(path[-1],type=direction) == []:
                        newPaths.append(path)
                        newPathsEdges.append(pathEdges)
                        countDone += 1
                    else:
                        for neighbor,adjacent in zip(G.neighbors(path[-1],type=direction),G.adjacent(path[-1],type=direction)):
                            if neighbor in path:
                                print('WARNING already in path')
                            if G.vs[neighbor]['kind'] == 'c':
                                newPaths.append(path + [neighbor])
                                newPathsEdges.append(pathEdges + [adjacent])
                            elif G.vs[neighbor]['kind'] == stopKind:
                                newPaths.append(path)
                                newPathsEdges.append(pathEdges)
                                countDone += 1
                            else:
                                pass
                paths = newPaths
                pathsEdges = newPathsEdges
                newPaths = []
                newPathsEdges = []
                if countDone == len(paths):
                    print('Add final vertex')
                    for path in paths:
                        if G.neighbors(path[-1],type=direction) == []:
                            newPaths.append(path)
                            newPathsEdges.append(pathEdges)
                        else:
                            for neighbor,adjacent in zip(G.neighbors(path[-1],type=direction),G.adjacent(path[-1],type=direction)):
                                newPaths.append(path + [neighbor])
                                newPathsEdges.append(pathEdges + [adjacent])
                    paths = newPaths
                    pathsEdges = newPathsEdges
                    boolContinue = 0
            if countLoop > stopPaths:
                dictDummy={}
                dictDummy['vertices']=[]
                dictDummy['edges']=[]
                dictDummy['finished']=0.0
            elif len(paths) > stopPaths2:
                dictDummy={}
                dictDummy['vertices']=[]
                dictDummy['edges']=[]
                dictDummy['finished']=0.5
            else:
                dictDummy={}
                dictDummy['vertices']=paths
                dictDummy['edges']=pathsEdges
                dictDummy['finished']=1.0
            pathDict[i]=dictDummy
        vgm.write_pkl(pathDict,'pathDict'+str(j+1)+'.pkl')

    #return pathDict
    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def path_between_a_and_v_for_vertexList_2(G,v,direction='out'):
    """Finds all posible paths from a given vertex that have a specific length.
    The paths can be directed or not directed, depending on the type of graph.
    INPUT: G: Vascular graph in iGraph format.
           v: Vertex List from which the search is to be started. 
           direction: Way to traverse a directed graph. Can be either 'out', 
                      'in', 'out' = from a to v and 'in' = from v to a
    OUTPUT: paths: The possible paths from v.
    """

    pathDict={}
    if direction == 'out':
        stopKind='v'
    else:
        stopKind='a'

    stopPaths=1e7
    stopPaths2=1e8
    indexHalf=np.floor(len(v)/9.)

    timeProbs=[]
    lengthsProbs=[]
    for j in range(9):    
        pathDict={}
        print('ROUND')
        print(j)
        stdout.flush()
        if j == 8:
            indexStart=indexHalf*j
            indexEnd=len(v)
        else:
            indexStart=indexHalf*j
            indexEnd=indexHalf*(j+1)
        for i in v[indexStart:indexEnd]:
            paths={}
            pathsEdges={}
            reconnection={}
            reconnected={}
            stopReached={}
            paths[0]=[i]
            pathsEdges[0]=[]
            reconnected[0]=0
            reconnection[0]=[]
            stopReached[0]=0
            boolContinue = 1
            countLoop=0
            countReconnected = 0
            stdout.flush()
            allVerts=[]
            boolInfo1=1
            boolInfo2=1
            boolInfo3=1
            boolInfo4=1
            tstart=ttime.time()
            tstep=ttime.time()
            while boolContinue:
                tdiff = tstep-tstart
                if tdiff > 43200:
                    print('stoppend because running longer than 14h')
                    timeProbs.append(i)
                    vgm.write_pkl(timeProbs,'timeProbs.pkl')
                    break
                else:
                    countDone=0
                    countLoop += 1
                    pathsCount=len(paths.keys())
                    if countLoop > 1e4 and boolInfo1 == 1:
                        print('Path length > 1e4')
                        boolInfo1 = 0
                    if countLoop > 1e5 and boolInfo2 == 1:
                        print('Path length > 1e5')
                        boolInfo2 = 0
                    if countLoop > 1e6 and boolInfo3 == 1:
                        print('Path length > 1e6')
                        boolInfo3 = 0
                    if countLoop > 5e6 and boolInfo4 == 1:
                        print('Path length > 5e6')
                        boolInfo4 = 0
                    if countLoop > stopPaths:
                        print('Path length is getting too long')
                        lengthsProbs.append([i,0])
                        break
                    if pathsCount - countReconnected > stopPaths2:
                        print('Number of  Path is getting too large')
                        lengthsProbs.append([i,1])
                        break
                    countReconnected = 0
                    for k in paths.keys():
                        if G.neighbors(paths[k][-1],type=direction) == [] or reconnected[k]== 1:
                            if G.neighbors(paths[k][-1],type=direction) == []:
                                countDone += 1
                            elif reconnected[k]== 1:
                                countReconnected += 1
                        else:
                            countNeighbor = 0
                            pathsOld=paths[k]
                            pathsEdgesOld=pathsEdges[k]
                            neighbors2=[]
                            adjacents2=[]
                            for neighbor,adjacent in zip(G.neighbors(paths[k][-1],type=direction),G.adjacent(paths[k][-1],type=direction)):
                                if neighbor not in neighbors2:
                                    neighbors2.append(neighbor)
                                    adjacents2.append(adjacent)
                            for neighbor,adjacent in zip(neighbors2,adjacents2):
                                if neighbor in paths[k]:
                                    print('WARNING already in path')
                                if G.vs[neighbor]['kind'] == 'c':
                                    if countNeighbor == 0:
                                        if neighbor in allVerts:
                                            for l in paths.keys():
                                                if neighbor in paths[l]:
                                                    reconnection[k] = [l,paths[l].index(neighbor)]
                                                    reconnected[k] = 1
                                                    countReconnected += 1
                                                    break
                                        else:
                                            reconnected[k] = 0
                                        paths[k]=pathsOld + [neighbor]
                                        pathsEdges[k]=pathsEdgesOld + [adjacent]
                                    else:
                                        if neighbor in allVerts:
                                            for l in paths.keys():
                                                if neighbor in paths[l]:
                                                    reconnection[pathsCount] = [l,paths[l].index(neighbor)]
                                                    reconnected[pathsCount] = 1
                                                    countReconnected += 1
                                                    break
                                        else:
                                            reconnected[pathsCount] = 0
                                        paths[pathsCount] = pathsOld + [neighbor]
                                        pathsEdges[pathsCount]=pathsEdgesOld + [adjacent]
                                        reconnected[pathsCount]=0
                                        stopReached[pathsCount]=0
                                        reconnection[pathsCount]=[]
                                        pathsCount = len(paths.keys())
                                    allVerts.append(neighbor)
                                elif G.vs[neighbor]['kind'] == stopKind:
                                    countDone += 1
                                    stopReached[k]=1
                                else:
                                    pass
                                countNeighbor = 1
                        allVerts = np.unique(allVerts)
                        allVerts = allVerts.tolist()
                    if countDone + countReconnected == pathsCount:
                        print('Add final vertex')
                        for k in paths.keys():
                            pathsOld=paths[k]
                            pathsEdgesOld=pathsEdges[k]
                            if reconnected[k] != 1:
                                if G.neighbors(paths[k][-1],type=direction) == []:
                                    pass
                                else:
                                    countNeighbor = 0
                                    neighbors2=[]
                                    adjacents2=[]
                                    for neighbor,adjacent in zip(G.neighbors(paths[k][-1],type=direction),G.adjacent(paths[k][-1],type=direction)):
                                        if neighbor not in neighbors2:
                                            neighbors2.append(neighbor)
                                            adjacents2.append(adjacent)
                                    for neighbor,adjacent in zip(neighbors2,adjacents2):
                                        if countNeighbor == 0:
                                            paths[k]=paths[k] + [neighbor]
                                            pathsEdges[k]=pathsEdges[k] + [adjacent]
                                        else:
                                           paths[pathsCount] = pathsOld + [neighbor]
                                           pathsEdges[pathsCount]=pathsEdgesOld + [adjacent]
                                           pathsCount = len(paths.keys())
                                           reconnected[pathsCount]=0
                                           stopReached[pathsCount]=0
                                           reconnection[pathsCount]=[]
                                        countNeighbor = 1
                        boolContinue = 0
                tstep=ttime.time()
            if countLoop > stopPaths:
                dictDummy={}
                dictDummy['vertices']=[]
                dictDummy['edges']=[]
                dictDummy['finished']=0.0
                dictDummy['reconnectionPath']=[]
                dictDummy['reconnectionPos']=[]
            elif len(paths) > stopPaths2:
                dictDummy={}
                dictDummy['vertices']=[]
                dictDummy['edges']=[]
                dictDummy['finished']=0.5
                dictDummy['reconnectionPath']=[]
                dictDummy['reconnectionPos']=[]
            else:
                dictDummy={}
                dictDummy['vertices']=paths
                dictDummy['edges']=pathsEdges
                reconnectedList=[] 
                reconnectionPath=[] 
                reconnectionPos=[]
                stopReachedList=[]
                keys = reconnected.keys()
                keys.sort()
                for k in keys:
                    reconnectedList.append(reconnected[k])
                    stopReachedList.append(stopReached[k])
                    if len(reconnection[k]) > 0:
                        reconnectionPath.append(reconnection[k][0])
                        reconnectionPos.append(reconnection[k][1])
                    else:
                        reconnectionPath.append(None)
                        reconnectionPos.append(None)
                dictDummy['reconnected']=reconnected
                dictDummy['stopReached']=stopReachedList
                dictDummy['reconnectionPath']=reconnectionPath
                dictDummy['reconnectionPos']=reconnectionPos
                dictDummy['finished']=[1]*len(keys)
            pathDict[i]=dictDummy
        vgm.write_pkl(pathDict,'pathDict'+str(j+1)+'.pkl')

    #return pathDict
    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

    
def shortest_path_between_two_vertices(G,v1,v2,weights=None):
    """Computes the shortest path between two vertices 
    INPUT: G:  Vascular graph in iGraph format.
           v1: Source vertex
           v2: Target vertex           
           weights: Edge weights in a list or the name of an edge attribute
                    holding edge weights. If C{None}, all edges are assumed to 
                    have equal weight.
    OUTPUT: path: At most one shortest path between the two given vertices.
                  Note that for directed graphs, the order of v1 and v2 
                  matters!
    """    
    
    path = filter(lambda x: x[0] == v2 or x[-1] == v2,
                  G.get_shortest_paths(v1,weights,'OUT'))[0]
    return path


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def paths_to_subgraph(G,paths,filename=None):
    """Constructs a subgraph of the input-graph, based on the given paths. That
    is, only nodes that are listed in the variable 'paths' will be included in
    the subgraph. Optionally, the subgraph can be saved to disk in form of a
    vtp-file.
    INPUT: G: Vascular graph in iGraph format.
           paths: Paths in the vascular graph in form of a list of vertex-
                  lists. Note that this list can be produced with
                  'all_paths_between_two vertices' or
                  'all_paths_of_a_given_length'.
           filename: (Optional.) Name of the vtp-file to which the subgraph
                     should be saved.
    OUTPUT: sg: Subgraph containing only those nodes in the paths variable.       
    """

    
    sg = g.subgraph(pl.flatten(paths))

    if filename != None:
        g_output.write_vgm(sg,filename)
    
    return sg


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


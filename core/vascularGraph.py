from __future__ import division

from copy import deepcopy
from igraph import Graph
import quantities as pq
import numpy as np
import pylab
from scipy.spatial import kdtree
import time as ttime
from scipy import finfo
import scipy as sp
from sys import stdout
from scipy import interpolate

import g_math
import units
import misc
import vgm

__all__ = ['VascularGraph']
#log = vgm.LogDispatcher.create_logger(__name__)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#Graph(num_of_vertices, edgelist, directed, graph_attrs, vertex_attrs, edge_attrs)

class VascularGraph(Graph):
    
    def __init__(self, *args, **kwargs):
        super(VascularGraph, self).__init__(*args, **kwargs)
        # Graph.__init__(self, *args, **kwargs)        
        if 'defaultUnits' not in kwargs.keys():
            defaultUnits = {'length': 'um', 'mass': 'ug', 'time': 'ms'}
        else:
            defaultUnits = kwargs['defaultUnits']
        self['defaultUnits'] = defaultUnits
            
        pq.set_default_units(**defaultUnits)
        
        
        #if 'defaultUnits' in self.attributes():
        #    return ('defaultUnits', self['defaultUnits'])
    
    #--------------------------------------------------------------------------
    # miscellaneous methods
    #--------------------------------------------------------------------------
    
    def neighbors2(self, vertex, mode='all'):
        """Returns the neighbors of a vertex. This method takes care that the
        order of the neighbors returned matches the order of the corresponding
        edges, as provided by the adjacent method.
        INPUT: vertex: The vertex index.
               mode: This can be 'all' (default), 'in', 'out' - ignored for 
                     undirected graphs.
        OUTPUT: Neighboring vertices.
        WARNING: This method is much slower than 'neighbors()', especially for
                 large graphs!
        """
        es = self.es[self.adjacent(vertex, mode)]
        return [e.source if e.source != vertex else e.target for e in es]
        
    def copy_graph_attributes(self, sourceGraph, mode='add'):
        """Adds the properties of the sourceGraph to the current VascularGraph.
        INPUT: sourceGraph: VascularGraph whose attributes are to be copied.
               mode: The copy-mode, this can be either 'add' or 'overwrite'
        OUTPUT: None
        """
        for attribute in sourceGraph.attributes():
            if mode == 'add':
                if attribute not in self.attributes():
                    self[attribute] = deepcopy(sourceGraph[attribute])
            elif mode == 'overwrite':
                self[attribute] = deepcopy(sourceGraph[attribute])        
 
    def adjacentEdges(self, edge):
        """Returns the adjacent Edges of a edge. This method takes care that the
        order of the neighbors returned matches the order of the corresponding
        edges, as provided by the adjacent method.
        INPUT: vertex: The vertex index.
               mode: This can be 'all' (default), 'in', 'out' - ignored for 
                     undirected graphs.
        OUTPUT: Neighboring vertices.
        WARNING: This method is much slower than 'neighbors()', especially for
                 large graphs!
        """
        edgeDo=self.es[edge]
        source=edgeDo.source
        target=edgeDo.target
        edges1=self.adjacent(source)
        edges2=self.adjacent(target)
        edges=edges1+edges2
        edges=np.unique(edges)
        edges=edges.tolist()
        edges.remove(edge)
        return edges

    #--------------------------------------------------------------------------           

    def copy_vertex_attributes(self, source, target, G=None):
        """Copies the attributes of vertex source to vertex target. Source and
        target can be a list of vertices, in which case the n-th source is 
        copied to the n-th target for n=0..len(source)-1.
        INPUT: source: The index of the source vertex
               target: The index of the target vertex
               G: The VascularGraph to which the source vertex belongs. If not
                  provided, it is assumed it belongs to the current instance.
        OUTPUT: None
        """
        if G is None:
            G = self
        if type(source) == int:
            source = [source]
            target = [target]
        for attribute in G.vs.attribute_names():
            self.vs[target][attribute] = deepcopy(G.vs[source][attribute]) 
#        for s,t in zip(source, target):    
#            for attribute in G.vs.attribute_names():
#                self.vs[t][attribute] = deepcopy(G.vs[s][attribute])
            
    #--------------------------------------------------------------------------           
            
    def copy_edge_attributes(self, source, target, G=None, assertPoints=True):
        """Copies the attributes of edge source to edge target. Source and
        target can be a list of edges, in which case the n-th source is copied 
        to the n-th target for n=0..len(source)-1.
        INPUT: source: The index of the source edge
               target: The index of the target edge
               G: The VascularGraph to which the source edge belongs. If not
                  provided, it is assumed it belongs to the current instance.
               assertPoints: Whether to check the integrety of the attribute
                             'points'. Amira expects to find the first and last
                             point of an edge to be identical to the edge
                             source and target respectively, with source_index 
                             < target_index.                                               
        OUTPUT: None
        """
        if G is None:
            G = self
        if type(source) == int:
            source = [source]
            target = [target]
        for attribute in G.es.attribute_names():
            self.es[target][attribute] = deepcopy(G.es[source][attribute])
        boolLengths2=0
        if 'lengths2' in  self.es.attribute_names():
            boolLengths2=1
        if assertPoints:
            for e in self.es[target]:
                if any(e['points'][0] != self.vs[e.source]['r']):
                    e['points']=e['points'][::-1]
                    e['diameters']=e['diameters'][::-1]
                    e['lengths']=e['lengths'][::-1]
                    if boolLengths2:
                        e['diameters2']=e['diameters2'][::-1]
                        e['lengths2']=e['lengths2'][::-1]
            #self.es[target]['points'] = [e['points'][::-1]
            #        if any(e['points'][0] != self.vs[e.source]['r'])
            #        else e['points'] for e in self.es[target]]
#            if attribute == 'points' and assertPoints:
#                for s,t in zip(source, target):
#                    points = deepcopy(G.es[s][attribute])
#                    if any(points[0] != self.vs[self.es[t].source]['r']):
#                        points = points[::-1]
#                    self.es[t]['points'] = points
#            else:
#                for s,t in zip(source, target):                    
#                    self.es[t][attribute] = deepcopy(G.es[s][attribute])    
    
    #--------------------------------------------------------------------------

    def union_attribute_preserving(self, other, gaCopyMode='add',
                                   assertPoints=True,kind='n'):
        """Creates a union of two VascularGraphs while preserving vertex and 
        edge attributes. Vertices that have identical coordinates (property 
        'r') are considered to be the same vertex.
        INPUT: other: The VascularGraph to be added to the instance.
               gaCopyMode: The mode with which graph attributes are copied from
                           other to self. This can be either 'omit', 'add' or 
                           'overwrite'.
               assertPoints: Whether to check the integrety of the attribute
                             'points'. Amira expects to find the first and last
                             point of an edge to be identical to the edge
                             source and target respectively, with source_index 
                             < target_index.                
               kind: kind preferences can be given, such that it is differed between vertices
                     with identical coordinates. if 'n' is the kind, vertices with same coordinates
                     are treated to be the same vertex. If the prefered kind is not available a 
                     random vertex is chosen
        OUTPUT: None, the VascularGraph is modified in-place.
        """

        Kdt = kdtree.KDTree(self.vs['r'], leafsize=10)
        vStartingIndex = self.vcount()
        eStartingIndex = self.ecount()
        self.add_vertices(other.vcount())
        self.add_edges([(e[0] + vStartingIndex, e[1] + vStartingIndex)
                        for e in other.get_edgelist()])        
        self.copy_vertex_attributes(xrange(other.vcount()),
                                    xrange(vStartingIndex, self.vcount()), 
                                    other)
        self.copy_edge_attributes(xrange(other.ecount()),
                                  xrange(eStartingIndex, self.ecount()), 
                                  other, assertPoints)


        #distance of the newly added vertices to the closest node of the old graph and the vertex number of the vertex in the old graph
        if kind =='n':
            distances, vertices = Kdt.query(self.vs(xrange(vStartingIndex, 
                                                       self.vcount()))['r'])
        else:
            distances=[]
            vertices=[]
            for i in self.vs[vStartingIndex:self.vcount()]['r']:
                nearest=Kdt.query(i,k=5)
                boolKind=0
                for j,k in zip(nearest[1],nearest[0]):
                    if self.vs[int(j)]['kind']==kind:
                        distances.append(k)
                        vertices.append(j)
                        boolKind = 1
                        break
                if not boolKind:
                    distances.append(nearest[0][0])
                    vertice.append(nearest[1][0])
  
        eps = finfo(float).eps * 1e4
        distances=np.array(distances)
        vertices=np.array(vertices)
        #Original vertices where distance to newly added graph == 0.0
        originals = vertices[np.nonzero(distances <= eps)[0]].tolist()
        #Vertex number of duplicate vertices
        duplicates = (np.nonzero(distances <= eps)[0] + vStartingIndex).tolist()
        
        for o, d in zip(originals, duplicates):
            for j, n in enumerate(self.neighbors(d)):
                if n not in duplicates:
                    self.add_edges((o, n))
                    self.copy_edge_attributes(self.adjacent(d, 'all')[j], 
                                              self.ecount()-1, None,
                                              assertPoints)
        self.delete_vertices(duplicates)
        if gaCopyMode != 'omit':
            self.copy_graph_attributes(other, gaCopyMode)        

    #--------------------------------------------------------------------------

    def disjoint_union_attribute_preserving(self, other, gaCopyMode,
                                            assertPoints=True):
        """Creates a disjoint union of two VascularGraphs while preserving 
        vertex and edge attributes.
        INPUT: other: The VascularGraph to be added to the instance.
               gaCopyMode: The mode with which graph attributes are copied from
                           other to self. This can be either 'omit', 'add' or 
                           'overwrite'.        
               assertPoints: Whether to check the integrety of the attribute
                             'points'. Amira expects to find the first and last
                             point of an edge to be identical to the edge
                             source and target respectively, with source_index 
                             < target_index.                                               
        OUTPUT: None, the VascularGraph is modified in-place.
        """
        vStartingIndex = self.vcount()
        eStartingIndex = self.ecount()
        self.add_vertices(other.vcount())
        self.add_edges([(e[0] + vStartingIndex, e[1] + vStartingIndex)
                        for e in other.get_edgelist()])        
        self.copy_vertex_attributes(xrange(other.vcount()),
                                    xrange(vStartingIndex, self.vcount()), 
                                    other)
        self.copy_edge_attributes(xrange(other.ecount()),
                                  xrange(eStartingIndex, self.ecount()), 
                                  other, assertPoints)
        if gaCopyMode != 'omit':
            self.copy_graph_attributes(other, gaCopyMode)        
                
    #--------------------------------------------------------------------------
    
    def vertices_away_from_border(self, distance, shape=None):
        """Returns the vertex indices of those vertices that are a given 
        distance away from the border of the VascularGraph.
        INPUT: distance: The desired distance from the border.
               shape: The shape of the VascularGraph domain (e.g. cylindrical).
                      If not provided, attempts will be made to determine it.
        OUTPUT: The indices of the 'inside vertices' as a list.              
        """
        if shape == None:
            shape = self.shape()

        r = np.vstack(self.es['points'])
        minima = np.amin(r,0)
        maxima = np.amax(r,0)    
        if shape == 'cuboid':
            insideVertices = [v.index for v in self.vs if 
                              min(min(maxima-v['r']),
                              min(v['r']-minima)) > distance]
        elif shape == 'cylinder':
            radius, center = self.radius_and_center()
            insideVertices = [v.index for v in self.vs if
                              radius - np.linalg.norm(v['r'][:-1]-center) > 
                              distance and
                              maxima[2] - v['r'][2] > distance and
                              v['r'][2] - minima[2] > distance]
        else:
            raise KeyError('Only domains of shape cuboid or cylinder are \
                            currently implemented')
        return insideVertices                    
    
    #--------------------------------------------------------------------------
    
    def to_directed_flow_based(self):
        """Converts an undirected vascular graph to a directed graph. The 
        direction of the edges corresponds to the direction of flow (i.e. from 
        higher to lower pressure.
        Note that this method does not change the order of the edges, i.e. edge
        indices are preserved!
        INPUT: None (except self)
        OUTPUT: None - the graph is modified in place
        WARNING: Eventhogh the order of the edges is preserved, 'adjacent()' 
                 will return a different order.
        """
        
        self.to_directed(mutual=False)
        edgelist = self.get_edgelist()
        newEdgelist = []
        for edge in edgelist:
            if self.vs[edge[0]]['pressure'] < self.vs[edge[1]]['pressure']:
                newEdgelist.append((edge[1],edge[0]))
            else:
                newEdgelist.append(edge)
        es = {}        
        for attribute in self.es.attribute_names():
            es[attribute] = deepcopy(self.es[attribute])                

        self.delete_edges(xrange(self.ecount()))
        self.add_edges(newEdgelist)
        for attribute in es:
            self.es[attribute] = es[attribute]

    #--------------------------------------------------------------------------

    def get_edge_vertices(self, edgeIndices, flatten=True):
        """Finds the tupel of vertex indices associated to each of the given
        edge indices.
        INPUT: edgeIndices: The indices of the edges of interest.
               flatten: Boolean indicating whether or not the returned list is 
                        to be flattened (the flattened list will also only 
                        contain unique entries).
        OUTPUT: Vertex indices either as list of tuples, or as a flattened list
                of unique entries.
        """
        tuples = [e.tuple for e in self.es(edgeIndices)]
        if flatten:
            return np.unique(np.array(tuples)).tolist()
        else:
            return tuples    

    #--------------------------------------------------------------------------

    def get_vertex_edges(self, vertexIndices, type='all', flatten=True):
        """Finds the indices of the adjacent edges associated to each of the 
        given vertex indices.
        INPUT: vertexIndices: The indices of the vertices of interest.
               type: Whether to return only edges to predecessors ('in'), 
                     successors ('out') or both ('all').
               flatten: Boolean indicating whether or not the returned list is 
                        to be flattened (the flattened list will also only 
                        contain unique entries).
        OUTPUT: Edge indices either as list of tuples, or as a flattened list
                of unique entries.
        """
        adjacentEdges = [self.adjacent(v, type) for v in vertexIndices]
        if flatten:
            return np.unique(pylab.flatten(adjacentEdges)).tolist()
        else:
            return adjacentEdges    
        
    #--------------------------------------------------------------------------

    def get_capillary_vertices(self, dThreshold=None, mode='all'):
        """Finds the tupel of vertex indices associated to each of the given
        edge indices.
        INPUT: dThreshold: The diameter threshold that separates capillaries
                           from non-capillaries. If no value is supplied, 
                           8 microns is used as the default.
               mode: The search-mode to qualify a vertex as a capillary vertex.
                     If set to 'all', all adjacent edges need to be 
                     capillaries. If set to 'some', one or more edges suffice.             
        OUTPUT: Edge indices either as list of tuples, or as a flattened list
                of unique entries.
        """
        if dThreshold is None:
            dThreshold = 7.0 * \
                         units.scaling_factor_du('um', self['defaultUnits'])
        if mode == 'all':                 
            ci = [v for v in xrange(self.vcount())
                  if max(self.es(self.adjacent(v,'all'))['diameter']) <= 
                  dThreshold]
        elif mode == 'some':
            ci = [v for v in xrange(self.vcount())
                  if min(self.es(self.adjacent(v,'all'))['diameter']) <= 
                  dThreshold]
        return ci    

    #--------------------------------------------------------------------------

    def get_tree_subcomponent(self, startingVertex, dThreshold):
        """Finds the indices of vertices that make up a vascular tree, given
        the index of a vertex, which is part of that tree. The search stops at
        edges which have a diameter below a certain threshold value. This is 
        typically used to exclude the capillaries, so that a purely arterial or
        venous tree is extracted.
        INPUT: startingVertex: The index of the vertex at which the search is 
                               started.
               dThreshold: The diameter threshold below or equal to which edges 
                           are to be ignored.
        OUTPUT: The indices of the vertices that make up the vascular tree.
        """
        GC = VascularGraph(self.get_edgelist())
        GC.es['diameter'] = self.es['diameter']
        GC.delete_edges(GC.es(diameter_le=dThreshold))
        return GC.subcomponent(startingVertex,'all')

    #-------------------------------------------------------------------------- 
 
    def get_tree_subgraph(self, startingVertex, dC, dNC=None):
        """Extracts a vascular tree given the starting vertex of that tree. The
        search stops at edges which have a diameter below a certain threshold 
        value. This is typically used to exclude the capillaries, so that a 
        purely arterial or venous tree is extracted. If the tree has pre-
        capillary connections to other trees, these trees can be excluded by 
        providing a suitable value for the dNC parameter.
        INPUT: startingVertex: The index of the vertex at which the search is 
                               started.
               dC: The diameter threshold below or equal to which edges are to 
                   be ignored.
               dNC: The diameter above which ending-edges with vertices not 
                    including the starting vertex are considered as the 
                    starting edge of a different, individual tree. The default
                    value is none, in which case no search for other trees is
                    performed.
        OUTPUT: The vascular tree in VascularGraph format.
        """
        
        GC = VascularGraph(self.get_edgelist())
        GC.es['diameter'] = self.es['diameter']
        GC.delete_edges(GC.es(diameter_le=dThreshold))
        return GC.subcomponent(startingVertex,'all')

    #--------------------------------------------------------------------------

    def get_backbone(self, bbVertices=None, mode='edges'):
        """Iteratively removes hanging nodes until the (fluid-dynamically 
        functional) backbone is left.
        INPUT: bbVertices: Vertices that are part of the backbone. If none are
                           provided, the routine looks for pressure boundary 
                           nodes in terms of the graph attributes 'av' and 
                           'vv'. If these are not available either, the 
                           original graph is returned.
               mode: This may be either 'edges' or 'components', signifying the
                     degree to which pruning takes place. The default mode
                     'edges' removes all edges not belonging to the backbone,
                     whereas the mode 'components' simply removes all non-
                     backbone components (which is coarser). 
        OUTPUT: Backbone of the vascular network in VascularGraph format.
        """
        #if mode not in ['edges', 'components']:
        #    log.warning('Mode %s not recognized. Setting mode == "edges"' %
        #                (mode))
        GC = deepcopy(self)
        if bbVertices is None:
            bbVertices = []
            for prop in ['av', 'vv']:
                try:
                    GC.vs[GC[prop]]['avvv'] = [prop for p in GC[prop]]
                    bbVertices.extend(self[prop])
                except:
                    pass
        if bbVertices == []:
            return GC
                
        GC.vs['isBBVertex'] = [1 if v in bbVertices else 0 
                               for v in xrange(self.vcount())]
        
        # Delete all components not connected to a BC vertex:
        components = GC.components(mode='weak')
        deletionList = []
        for component in components:
            if not any([x in component for x in bbVertices]):
                deletionList.extend(component)
        GC.delete_vertices(deletionList)
        if mode == 'edges':
            # Consecutively remove degree 1 vertices:
            while True:
                GC.vs['degree'] = GC.degree()
                deletionList = GC.vs(degree_le=1, isBBVertex_eq=0).indices
                if len(deletionList) == 0:
                    break
                else:
                    GC.delete_vertices(deletionList)
        # Remove island boundary vertices:
        GC.vs['degree'] = GC.degree()
        GC.delete_vertices(GC.vs(degree_eq=0).indices)
        
        # Update av and vv indices:
        if 'avvv' in GC.vs.attribute_names():
            for prop in ['av', 'vv']:
                try:
                    GC[prop] = GC.vs(avvv_eq=prop).indices
                except:
                    pass
            del GC.vs['avvv']
        del GC.vs['isBBVertex']
        del GC.vs['degree']
        return GC        

    #--------------------------------------------------------------------------   
    def get_isolated_edges(self):
        """Finds the indices of the isolated edges in the graph.
        INPUT: None (except self)
        OUTPUT: Indices of the isolated edges.
        """
        ie = []
        for e in self.es:
            if self.degree(e.source) + self.degree(e.target) == 2:
                ie.append(e.index)
        return ie

    #--------------------------------------------------------------------------
    
    def get_unconnected_vertices(self):
        """Finds the indices of the unconnected vertices in the graph.
        INPUT: None (except self)
        OUTPUT: Indices of the unconnected vertices.
        """
        return sp.nonzero([x == 0 for x in self.strength(weights=
                          [1 for i in xrange(self.ecount())])])[0].tolist()
                          
    #--------------------------------------------------------------------------
    
    def get_endpoints(self):
        """Finds all endpoints in the graph, i.e. those vertices that connect
        to exactly one other vertex.
        INPUT: None (except self)
        OUTPUT: Indices of the endpoint vertices.
        """
        return np.nonzero(np.array(self.degree()) == 1)[0].tolist()                        

    #--------------------------------------------------------------------------

    def closest_edge(self, coordinate):
        """Finds the edge of the VascularGraph closest to a given coordinate
        and returns its index, as well as the distance between coordinate and
        nearest edge point.
        If coordinate is a list of coordinates, the search is performed for
        each coordinate triple.
        INPUT: coordinate: The (x,y,z) search coordinate. This can also be a
                           list of coordinate triples. 
        OUTPUT: The index of the edge closest to the given coordinate.
        """
        Kdt = kdtree.KDTree(np.concatenate(self.es['points'], 0), leafsize=10)
        if np.shape(coordinate) == (3,):
            coordinate = [coordinate]
        searchResult = Kdt.query(coordinate)
        
        cumsum = np.cumsum([len(p) for p in self.es['points']])
        eIndex = []
        for cIndex in xrange(len(coordinate)):    
            pIndex = searchResult[1][cIndex]        
            eIndex.append(np.nonzero(cumsum > pIndex)[0][0])          
        return eIndex, searchResult[0]
                                
    #--------------------------------------------------------------------------

    def closest_vertex(self, coordinate):
        """Finds the vertex of the VascularGraph closest to a given coordinate
        and returns its index, as well as the distance between coordinate and
        nearest vertex.
        If coordinate is a list of coordinates, the search is performed for
        each coordinate triple.        
        INPUT: coordinate: The (x,y,z) search coordinate. This can also be a
                           list of coordinate triples.  
        OUTPUT: The index of the vertex closest to the given coordinate.
        """
        Kdt = kdtree.KDTree(self.vs['r'], leafsize=10)
        if np.shape(coordinate) == (3,):
            coordinate = [coordinate]        
        searchResult = Kdt.query(coordinate)
        return searchResult[1].tolist(), searchResult[0]          
                                        
    #--------------------------------------------------------------------------                         
    def assign_tissue_volume_to_edge(self,cubeSize=3):
        """
        """

        cubeSizeFactor=5
        x=[]; y=[]; z=[]
        for coords in self.vs['r']:
            x.append(coords[0])
            y.append(coords[1])
            z.append(coords[2])

        self.vs['x']=x; self.vs['y']=y; self.vs['z']=z

        zmin1=np.min(self.vs[self.vs(nkind_eq=0).indices+self.vs(nkind_eq=1).indices]['z'])
        zmin2=np.max(self.vs[self.vs(nkind_eq=0).indices+self.vs(nkind_eq=1).indices]['z'])
        zmin=np.mean(self.vs[self.vs(nkind_eq=0).indices+self.vs(nkind_eq=1).indices]['z'])
        xDist=np.max(x)-np.min(x)
        yDist=np.max(y)-np.min(y)
        zDist=np.max(z)-zmin
        
        zmax=np.mean(self.vs[self.vs(degree_eq=1,z_gt=zmin+0.8*zDist).indices]['z'])
        xmin=np.mean(self.vs[self.vs(degree_eq=1,x_lt=np.max(x)-0.75*xDist).indices]['x'])
        ymin=np.mean(self.vs[self.vs(degree_eq=1,y_lt=np.max(y)-0.75*yDist).indices]['y'])
        xmax=np.mean(self.vs[self.vs(degree_eq=1,x_gt=np.min(x)+0.75*xDist).indices]['x'])
        ymax=np.mean(self.vs[self.vs(degree_eq=1,y_gt=np.min(y)+0.75*yDist).indices]['y'])
        xDist=xmax-xmin
        yDist=ymax-ymin
        zDist=zmax-zmin1

        xSplits=int(np.ceil(xDist/cubeSize))
        ySplits=int(np.ceil(yDist/cubeSize))
        zSplits=int(np.ceil(zDist/cubeSize))
        xStart=xmin; yStart=ymin; zStart=zmin1
        xCurrent=xmin; yCurrent=ymin; zCurrent=zmin1
        diag=np.sqrt(3)*cubeSize
        diag2=np.sqrt(2)*cubeSize*cubeSizeFactor
        countVolumes=[0]*self.ecount()
        
        #Kdt = kdtree.KDTree(np.concatenate(self.es['points'], 0), leafsize=10)
        Kdt = kdtree.KDTree(np.concatenate(self.es['points'], 0))
        cumsum = np.cumsum([len(p) for p in self.es['points']])

        #Find zValues where the tissue shall start
        xSplits2=int(np.ceil(xDist/(cubeSize*cubeSizeFactor)))
        ySplits2=int(np.ceil(yDist/(cubeSize*cubeSizeFactor)))
        zStartMesh=[]
        for i in range(ySplits2):
            zAll=[]
            for j in range(xSplits2):
                x1=xCurrent
                y1=yCurrent
                zVals=[]
                for k in range(zSplits):
                    searchResult = Kdt.query([xCurrent+0.5*cubeSize*cubeSizeFactor,yCurrent+0.5*cubeSize*cubeSizeFactor,zCurrent+0.5*cubeSize])
                    print('Cube Center')
                    print([xCurrent+0.5*cubeSize*cubeSizeFactor,yCurrent+0.5*cubeSize*cubeSizeFactor,zCurrent+0.5*cubeSize])
                    print(searchResult[0])
                    if zCurrent >= zmin1 and zCurrent <= zmin2: #check if a vessel is lying in that cube, otherwise the cube 
                        #should not be considered for analyzing the tissue volume supplied
                        if searchResult[0] < 0.5*diag2:
                            zVals.append(zCurrent)
                    else:
                        break
                    zCurrent += cubeSize
                if zVals == []:
                    zVals.append(zmin2)
                zAll.append(np.min(zVals))
                zCurrent=zStart
                xCurrent += cubeSize*cubeSizeFactor
            zStartMesh.append(np.array(zAll))
            xCurrent=xStart
            yCurrent += cubeSize*cubeSizeFactor

        xx,yy=np.meshgrid(np.arange(xmin,xmax,cubeSize*cubeSizeFactor),np.arange(ymin,ymax,cubeSize*cubeSizeFactor))
        f=interpolate.interp2d(xx,yy,zStartMesh,kind='linear')
        xValsFine=np.arange(xmin,xmax,cubeSize)
        yValsFine=np.arange(ymin,ymax,cubeSize)
        zStartNew=f(xValsFine,yValsFine)
        zStartNewList=np.concatenate(zStartNew)

        print('Number of cubes')
        print(xSplits*ySplits*zSplits)
        print('Number of xSplits')
        print(xSplits)
        print('Number of Values in zStartList')
        print(len(zStartNewList))
        xStart=xmin; yStart=ymin; zStart=zmin1
        xCurrent=xmin; yCurrent=ymin; zCurrent=zmin1
        count = 0
        count2 = 0
        count3 = 0
        count4 = 0
        r = []
        associatedEdge = []
        for i in range(ySplits):
            count += 1
            print(count)
            for j in range(xSplits):
                for k in range(zSplits):
                    searchResult = Kdt.query([xCurrent+0.5*cubeSize,yCurrent+0.5*cubeSize,zCurrent+0.5*cubeSize])
                    if zCurrent >= zmin1 and zCurrent <= zmin2: #check if a vessel is lying in that cube, otherwise the cube 
                        if zCurrent > zStartNewList[count4]:
                            EdgeIndex=np.nonzero(cumsum > searchResult[1])[0][0]
                            countVolumes[EdgeIndex] = countVolumes[EdgeIndex]+1
                            r.append(np.array([xCurrent+0.5*cubeSize,yCurrent+0.5*cubeSize,zCurrent+0.5*cubeSize]))
                            count3 += 1
                            associatedEdge.append(EdgeIndex)
                    else:
                        EdgeIndex=np.nonzero(cumsum > searchResult[1])[0][0]
                        countVolumes[EdgeIndex] = countVolumes[EdgeIndex]+1
                        r.append(np.array([xCurrent+0.5*cubeSize,yCurrent+0.5*cubeSize,zCurrent+0.5*cubeSize]))
                        count3 += 1
                        associatedEdge.append(EdgeIndex)
                    zCurrent += cubeSize
                    count2 += 1
                count4 += 1
                zCurrent=zStart
                xCurrent += cubeSize
            xCurrent=xStart
            yCurrent += cubeSize
            if count >= 15:
                print('Cubes done')
                print(count2)
                self.es['countVolumes']=countVolumes
                vgm.write_pkl(self,'G_averaged_withMainDA_125_withTissueVol_'+str(count2)+'.pkl')
                count = 0

        print('Total number of cubes')
        print(count3)

        self.es['tissueVolume']=np.array(self.es['countVolumes'])*cubeSize**3
        self.es['tissueVolumePerLength']=np.array(self.es['tissueVolume'])/np.array(self.es['length'])
        self.es['tissueRadiusPerLength']=np.sqrt(np.array(self.es['tissueVolumePerLength'])/np.pi)
        vgm.write_pkl(self,'G_averaged_withMainDA_125_withTissueVol.pkl')

        tissueG=vgm.VascularGraph(len(r))
        tissueG.vs['r'] = r
        tissueG.vs['associatedEdge'] = associatedEdge
        vgm.write_pkl(tissueG,'GTissue_averaged_withMainDA_125.pkl')
        vgm.write_vtp(tissueG,'GTissue_averaged_withMainDA_125.vtp',False)

    #--------------------------------------------------------------------------                         
                         
    def split_edge(self, eIndex, pIndex, deleteOldEdge=True):
        """Splits an edge in two by introducing an additional vertex.
        If RBCs are in the edge, the RBCs are distributed as well and nRBC is updated.
        
        
        INPUT: eIndex: The index of the edge to be split.
               pIndex: The index of the edge-point which will become the new 
                       vertex.
               deleteOldEdge: Boolean that defines whether or not to delete the
                              old edge after it has been split into two new 
                              ones, default is 'True'. 
                              If splitting multiple edges, it may be useful not
                              to delete the old edge as this changes the edge
                              indices of all edges > eIndex. One would delete
                              all old edges in one go after each one has been 
                              split instead.
        OUTPUT: None
        """
        newVertex = self.vcount()
        source = self.es[eIndex].source
        target = self.es[eIndex].target
        self.add_vertices(1)
        self.vs[newVertex]['r'] = self.es[eIndex]['points'][pIndex]
        
        newEdges = [self.ecount(), self.ecount()+1]
        self.add_edges([(source, newVertex), (target, newVertex)])
        self.es[newEdges[0]]['diameters'] = self.es[eIndex]['diameters'][:pIndex+1]
        self.es[newEdges[0]]['lengths'] = self.es[eIndex]['lengths'][:pIndex+1]
        self.es[newEdges[0]]['points'] = self.es[eIndex]['points'][:pIndex+1]
        self.es[newEdges[0]]['diameter'] = np.sqrt(np.average(
                                        self.es[eIndex]['diameters'][:pIndex]**2.0,
                                        weights=self.es[eIndex]['lengths'][:pIndex]))
        self.es[newEdges[0]]['length'] = sum(self.es[eIndex]['lengths'][:pIndex])
        if 'rRBC' in self.es.attribute_names():
            if 'nRBC' in self.es.attribute_names():
                for i in range(len(self.es['rRBC'][eIndex])):
                    if self.es[eIndex]['rRBC'][i] > self.es[newEdges[0]]['length']:
                        self.es[newEdges[0]]['rRBC']=self.es[eIndex]['rRBC'][0:i-1]
                        break
                if i == len(self.es['rRBC'][eIndex])-1:
                    self.es[newEdges[0]]['rRBC']=self.es[eIndex]['rRBC']
                    self.es[newEdges[1]]['rRBC']=[]
                else:
                    self.es[newEdges[1]]['rRBC']=np.array(self.es[eIndex]['rRBC'][i::])-np.array([self.es[newEdges[0]]['length']]*len(self.es[eIndex]['rRBC'][i::]))
                self.es[newEdges[0]]['nRBC']=len(self.es[newEdges[0]]['rRBC'])
                self.es[newEdges[1]]['nRBC']=len(self.es[newEdges[1]]['rRBC'])
                self.es[newEdges[0]]['nRBC_avg']=len(self.es[newEdges[0]]['rRBC'])
                self.es[newEdges[1]]['nRBC_avg']=len(self.es[newEdges[1]]['rRBC'])
        self.es[newEdges[1]]['diameters'] = self.es[eIndex]['diameters'][pIndex:][::-1]
        self.es[newEdges[1]]['lengths'] = self.es[eIndex]['lengths'][pIndex:][::-1]
        self.es[newEdges[1]]['points'] = self.es[eIndex]['points'][pIndex:][::-1]
        self.es[newEdges[1]]['diameter'] = np.sqrt(np.average(
                                        self.es[eIndex]['diameters'][pIndex:-1]**2.0,
                                        weights=self.es[eIndex]['lengths'][pIndex:-1]))
        self.es[newEdges[1]]['length'] = sum(self.es[eIndex]['lengths'][pIndex:-1])
            
        attr = self.es.attribute_names()
        for a in attr:
            if a not in ('length', 'lengths', 'diameter', 'diameters', 'points','nRBC','nRBC_avg','rRBC'):
                self.es[newEdges[0]][a] = self.es[eIndex][a]
                self.es[newEdges[1]][a] = self.es[eIndex][a]

        attr = self.vs.attribute_names()
        for a in attr:
            if a not in ('r',):
                self.vs[newVertex][a] = self.vs[source][a]

        if deleteOldEdge:
            self.delete_edges(eIndex)        
            
    #--------------------------------------------------------------------------                         
                         
    def split_edge_relDist(self, eIndex, relDist, species, deleteOldEdge=True):
        """Splits an edge in two by introducing an additional vertex.
        Points need to be present in the graph. Introduces further points if necessary.
        If pressure values are present, pressure value assigned at new vertex. If RBCs are present
        RBCs are distributed to new edges
        INPUT: eIndex: The index of the edge to be split.
               relDist: relativeDistance along the edge
               deleteOldEdge: Boolean that defines whether or not to delete the
                              old edge after it has been split into two new 
                              ones, default is 'True'. 
                              If splitting multiple edges, it may be useful not
                              to delete the old edge as this changes the edge
                              indices of all edges > eIndex. One would delete
                              all old edges in one go after each one has been 
                              split instead.
        OUTPUT: None
        """
        P=vgm.Physiology(self['defaultUnits'])
        vrbc=P.rbc_volume(species)
        eps = finfo(float).eps * 1e4
        newVertex = self.vcount()
        source = self.es[eIndex].source
        target = self.es[eIndex].target
        self.add_vertices(1)
        if 'pressure' in self.vs.attribute_names():
            pressureBool = 1
        else:
            pressureBool = 0

        if 'points' not in self.es.attribute_names():
            self.add_points(self.es['length'][eIndex]/3.,edgeList=[eIndex])
        elif self.es['points'][eIndex] == None:
            self.add_points(self.es['length'][eIndex]/3.,edgeList=[eIndex])

        #Calculate relDistance for points
        relDistPoints=[]
        lengthSum=0
        smallerBool=0
        largerBool=0
        noNewPointNeededBool=0
        index=0
        posNewVert=self.es[eIndex]['length']*relDist
        for i in range(len(self.es[eIndex]['lengths2'])):
            lengthSum += self.es[eIndex]['lengths2'][i]
            relDistCurrent=lengthSum/self.es[eIndex]['length']
            relDistPoints.append(relDistCurrent)
            #This ensures that in all the new edges we have at least 3 points in every edge
            if relDistCurrent > relDist and i == 0:
                smallerBool = 1 
            elif relDistCurrent < relDist and i == len(self.es[eIndex]['lengths2'])-2:
                largerBool = 1
            if np.abs(relDistCurrent - relDist) < eps and index == 0:
                noNewPointNeededBool=1
                index = i
                coordsNewVertex=self.es['points'][eIndex][i+1]
                self.vs[newVertex]['r'] = coordsNewVertex
                break
            elif relDistCurrent > relDist and index == 0:
                index = i
                vect=self.es[eIndex]['points'][i+1]-self.es[eIndex]['points'][i]
                diff = posNewVert-(lengthSum-self.es[eIndex]['lengths2'][i])
                relDistLocal= diff / self.es[eIndex]['lengths2'][i]
                coordsNewVertex=self.es['points'][eIndex][i] + relDistLocal*vect
                self.vs[newVertex]['r'] = coordsNewVertex
                break

        if smallerBool == 1:
            coordsNewPoint = self.es[eIndex]['points'][i] + 0.5*relDistLocal*vect
            points=self.es[eIndex]['points']
            diameters=self.es[eIndex]['diameters']
            pointsNew=[]
            diametersNew=[]
            pointsNew.append(points[0])
            diametersNew.append(diameters[0])
            #calculate diameter of new points
            length01=np.linalg.norm(points[0] - points[1])
            length0_newPoint=np.linalg.norm(points[0] - coordsNewPoint)
            length0_newVertex=np.linalg.norm(points[0] - coordsNewVertex)
            dnewPoint=diameters[0]+(diameters[1]-diameters[0])*length0_newPoint/length01
            dnewVertex=diameters[0]+(diameters[1]-diameters[0])*length0_newVertex/length01
            if pressureBool:
                pressurenewVertex=self.vs['pressure'][source]+(self.vs['pressure'][target]-self.vs['pressure'][source])*length0_newVertex/self.es[eIndex]['length']
            diametersNew.append(dnewPoint)
            diametersNew.append(dnewVertex)
            pointsNew.append(coordsNewPoint)
            pointsNew.append(coordsNewVertex)
            index = index + 1
            for j in range(1,len(points)):
                pointsNew.append(points[j])
                diametersNew.append(diameters[j])

        if largerBool == 1:
            coordsNewPoint = self.es[eIndex]['points'][i] + (0.5*(1-relDistLocal)+relDistLocal)*vect
            points=self.es[eIndex]['points']
            diameters=self.es[eIndex]['diameters']
            pointsNew=[]
            diametersNew=[]
            for j in range(0,len(points)-1):
                pointsNew.append(points[j])
                diametersNew.append(diameters[j])
            #calculate diameter of new points
            length01=np.linalg.norm(points[-2] - points[-1])
            length0_newPoint=np.linalg.norm(points[-2] - coordsNewPoint)
            length0_newVertex=np.linalg.norm(points[-2] - coordsNewVertex)
            dnewPoint=diameters[-2]+(diameters[-1]-diameters[-2])*length0_newPoint/length01
            dnewVertex=diameters[-2]+(diameters[-1]-diameters[-2])*length0_newVertex/length01
            if pressureBool:
                pressurenewVertex=self.vs['pressure'][source]+(self.vs['pressure'][target]-self.vs['pressure'][source])*(self.es[eIndex]['length']-(length01-length0_newVertex))/self.es[eIndex]['length']
            diametersNew.append(dnewVertex)
            diametersNew.append(dnewPoint)
            pointsNew.append(coordsNewVertex)
            pointsNew.append(coordsNewPoint)
            pointsNew.append(points[-1])
            diametersNew.append(diameters[-1])

        if smallerBool == 0 and largerBool == 0 and noNewPointNeededBool==0:
            pointsNew=[]
            diametersNew=[]
            points=self.es[eIndex]['points']
            diameters=self.es[eIndex]['diameters']
            for i in range(0,index+1): #all points until the new vertex
                pointsNew.append(points[i])
                diametersNew.append(diameters[i])
            #calculate diameter of new points
            length01=np.linalg.norm(points[i] - points[i+1])  #length of the edgeSegment in which the new vertex will be lying
            length0_newVertex=np.linalg.norm(points[i] - coordsNewVertex)
            length0_newVertex2=np.sum(self.es[eIndex]['lengths2'][0:index])
            dnewVertex=(diameters[i]*length0_newVertex+diameters[i+1]*(length01-length0_newVertex))/length01
            if pressureBool:
                pressurenewVertex=self.vs['pressure'][source]+(self.vs['pressure'][target]-self.vs['pressure'][source])*(length0_newVertex+length0_newVertex2)/self.es[eIndex]['length']
            diametersNew.append(dnewVertex)
            pointsNew.append(coordsNewVertex)
            for i in range(index+1,len(points)):
                pointsNew.append(points[i])
                diametersNew.append(diameters[i])
        elif smallerBool == 0 and largerBool == 0 and noNewPointNeededBool==1:
            pointsNew=self.es[eIndex]['points']
            diametersNew=self.es[eIndex]['diameters']
            if pressureBool:
                pressurenewVertex=self.vs['pressure'][source]+(self.vs['pressure'][target]-self.vs['pressure'][source])*relDist

        #List of all points and diameters, the new vertex has been introduced as a point in the list, still for the whole edge
        self.es[eIndex]['points']=pointsNew
        self.es[eIndex]['diameters']=diametersNew
        if pressureBool:
            self.vs[newVertex]['pressure']=pressurenewVertex

        #points,diameters should be well arranged now and can be split at the newly introduced vertex
        #pIndex is the vertex where the edges are split
        pIndex = index + 1
        newEdges = [self.ecount(), self.ecount()+1]
        self.add_edges([(source, newVertex), (target, newVertex)])
        #print deal with first newEdge
        self.es[newEdges[0]]['diameters'] = self.es[eIndex]['diameters'][:pIndex+1]
        self.es[newEdges[0]]['points'] = self.es[eIndex]['points'][:pIndex+1]
        #update lengths for first new Edge
        lengths2=[]
        for j in range(len(self.es[newEdges[0]]['points'])-1):
            lengthsNew = np.linalg.norm(self.es[newEdges[0]]['points'][j] - self.es[newEdges[0]]['points'][j+1])
            lengths2.append(lengthsNew)
        self.es[newEdges[0]]['lengths2']=lengths2
        self.es[newEdges[0]]['length'] = np.sum(lengths2)
        lengths=[]
        for j in range(len(self.es[newEdges[0]]['points'])):
            if j == 0:
                lengths.append(self.es[newEdges[0]]['lengths2'][j]/2.)
            elif j == len(self.es[newEdges[0]]['points'])-1:
                lengths.append(self.es[newEdges[0]]['lengths2'][j-1]/2.)
            else:
                 lengths.append(0.5*self.es[newEdges[0]]['lengths2'][j]+0.5*self.es[newEdges[0]]['lengths2'][j-1])
        self.es[newEdges[0]]['lengths']=lengths
        diameters2=[]
        resistances=[]
        for j in range(len(self.es[newEdges[0]]['points'])-1):
            if j == 0 or j == len(self.es[newEdges[0]]['points'])-2:
                diameters2.append((self.es[newEdges[0]]['diameters'][j]+self.es[newEdges[0]]['diameters'][j+1])/2)
            else:
                diameters2.append((self.es[newEdges[0]]['diameters'][j]*self.es[newEdges[0]]['lengths2'][j]+ \
                    self.es[newEdges[0]]['diameters'][j+1]*self.es[newEdges[0]]['lengths2'][j+1])/(self.es[newEdges[0]]['lengths2'][j]+self.es[newEdges[0]]['lengths2'][j+1]))
            resistances.append(self.es[newEdges[0]]['lengths2'][j]/(diameters2[-1]**4))
        resistanceTot=np.sum(resistances)
        self.es[newEdges[0]]['effDiam']=(self.es[newEdges[0]]['length']/resistanceTot)**(0.25)
        self.es[newEdges[0]]['diameter']=self.es[newEdges[0]]['effDiam']
        self.es[newEdges[0]]['diameters2']=diameters2

        #deal with second newEdge
        self.es[newEdges[1]]['diameters'] = self.es[eIndex]['diameters'][pIndex:][::-1]
        self.es[newEdges[1]]['points'] = self.es[eIndex]['points'][pIndex:][::-1]
        #update lengths for first new Edge
        lengths2=[]
        for j in range(len(self.es[newEdges[1]]['points'])-1):
            lengthsNew = np.linalg.norm(self.es[newEdges[1]]['points'][j] - self.es[newEdges[1]]['points'][j+1])
            lengths2.append(lengthsNew)
        self.es[newEdges[1]]['lengths2']=lengths2
        self.es[newEdges[1]]['length'] = np.sum(lengths2)
        diameters2=[]
        resistances=[]
        for j in range(len(self.es[newEdges[1]]['points'])-1):
            if j == 0 or j == len(self.es[newEdges[1]]['points'])-2:
                diameters2.append((self.es[newEdges[1]]['diameters'][j]+self.es[newEdges[1]]['diameters'][j+1])/2)
            else:
                diameters2.append((self.es[newEdges[1]]['diameters'][j]*self.es[newEdges[1]]['lengths2'][j]+ \
                    self.es[newEdges[1]]['diameters'][j+1]*self.es[newEdges[1]]['lengths2'][j+1])/(self.es[newEdges[1]]['lengths2'][j]+self.es[newEdges[1]]['lengths2'][j+1]))
            resistances.append(self.es[newEdges[1]]['lengths2'][j]/(diameters2[-1]**4))
        resistanceTot=np.sum(resistances)
        self.es[newEdges[1]]['effDiam']=(self.es[newEdges[1]]['length']/resistanceTot)**(0.25)
        self.es[newEdges[1]]['diameter']=self.es[newEdges[1]]['effDiam']
        self.es[newEdges[1]]['diameters2']=diameters2
        lengths=[]
        for j in range(len(self.es[newEdges[1]]['points'])):
            if j == 0:
                lengths.append(self.es[newEdges[1]]['lengths2'][j]/2.)
            elif j == len(self.es[newEdges[1]]['points'])-1:
                lengths.append(self.es[newEdges[1]]['lengths2'][j-1]/2.)
            else:
                 lengths.append(0.5*self.es[newEdges[1]]['lengths2'][j]+0.5*self.es[newEdges[1]]['lengths2'][j-1])
        self.es[newEdges[1]]['lengths']=lengths

        #Deal with RBCs if present
        boolBreak=0
        if 'rRBC' in self.es.attribute_names():
            if 'nRBC' in self.es.attribute_names():
                if len(self.es['rRBC'][eIndex]) > 0:
                    for i in range(len(self.es['rRBC'][eIndex])):
                        if self.es[eIndex]['rRBC'][i] > self.es[newEdges[0]]['length']:
                            self.es[newEdges[0]]['rRBC']=self.es[eIndex]['rRBC'][0:i]
                            boolBreak = 1
                            break
                    if i == len(self.es['rRBC'][eIndex])-1 and boolBreak == 0:
                        self.es[newEdges[0]]['rRBC']=self.es[eIndex]['rRBC']
                        self.es[newEdges[1]]['rRBC']=[]
                    else:
                        self.es[newEdges[1]]['rRBC']=np.array(self.es[eIndex]['rRBC'][i::])-np.array([self.es[newEdges[0]]['length']]*len(self.es[eIndex]['rRBC'][i::]))
                        self.es[newEdges[1]]['rRBC']=self.es[newEdges[1]]['length']-self.es[newEdges[1]]['rRBC'][::-1]
                else:
                    self.es[newEdges[0]]['rRBC']=[]
                    self.es[newEdges[1]]['rRBC']=[]
                self.es[newEdges[0]]['nRBC']=len(self.es[newEdges[0]]['rRBC'])
                self.es[newEdges[1]]['nRBC']=len(self.es[newEdges[1]]['rRBC'])
                self.es[newEdges[0]]['nRBC_avg']=len(self.es[newEdges[0]]['rRBC'])
                self.es[newEdges[1]]['nRBC_avg']=len(self.es[newEdges[1]]['rRBC'])
           
        if 'minDist' in self.es.attribute_names():
            self.es[newEdges[0]]['minDist']=vrbc / (0.25*np.pi * self.es[newEdges[0]]['diameter']**2)
            self.es[newEdges[1]]['minDist']=vrbc / (0.25*np.pi * self.es[newEdges[1]]['diameter']**2)

        if 'nMax' in self.es.attribute_names():
            self.es[newEdges[0]]['nMax'] = np.floor(self.es[newEdges[0]]['length']/ self.es[newEdges[0]]['minDist'])
            self.es[newEdges[1]]['nMax'] = np.floor(self.es[newEdges[1]]['length']/ self.es[newEdges[1]]['minDist'])

        attr = self.es.attribute_names()
        for a in attr:
            if a not in ('length', 'lengths', 'diameter', 'diameters', 'points','nRBC','nRBC_avg','rRBC','lengths2','diameters2','minDist','nMax'):
                self.es[newEdges[0]][a] = self.es[eIndex][a]
                if a == 'sign':
                    self.es[newEdges[1]][a] = -1*self.es[eIndex][a]
                else:
                    self.es[newEdges[1]][a] = self.es[eIndex][a]

        attr = self.vs.attribute_names()
        for a in attr:
            if a not in ('r','pBC','rBC','av','vv','kind','nkind','pressure'):
                self.vs[newVertex][a] = self.vs[source][a]

        if deleteOldEdge:
            self.delete_edges(eIndex)        
            
    #--------------------------------------------------------------------------                                                   

    def join_edges(self, orderTwoVertex, assertOrder=True):
        """Joins edges adjacent to an order two vertex.
        Note that this code is designed to work for VascularGraphs imported 
        from AmiraMesh files (i.e. 'diameter', 'diameters', 'length', 'lengths', 
        'points','lengths2','diameters2','kind' and 'nkind' are the only edge properties expected).
        INPUT: orderTwoVertex: The index of the order two vertex.
               assertOrder: Boolean determining whether to assert that the
                            vertex in question is indeed of order 2.
        OUTPUT: None
        """
        #if self.degree()[orderTwoVertex] != 2:
        #    log.error('Vertex %i is not of order 2. Aborting.' %
        #              orderTwoVertex)
        #    return
        
        # Convert from numpy integer:
        orderTwoVertex = int(orderTwoVertex)
                
        newEdge = self.ecount()
        neighbors = self.neighbors(orderTwoVertex)
        adjacent = self.adjacent(orderTwoVertex)
        self.add_edges([(neighbors[0], neighbors[1])])
        
        if 'diameters' in self.es.attribute_names() and \
           'lengths' in self.es.attribute_names():
            properties = ['diameters', 'lengths', 'points']
            for property in properties:
                data = []
                data.extend(self.es[adjacent[0]][property])
                if neighbors[0] > orderTwoVertex:
                    data = data[::-1]
                data = data[:-1] # Both edges contain this datapoint    
                if neighbors[1] < orderTwoVertex:
                    data.extend(self.es[adjacent[1]][property][::-1])
                else:
                    data.extend(self.es[adjacent[1]][property])
                self.es[newEdge][property] = np.array(data)

        if 'diameters2' in self.es.attribute_names() and \
           'lengths2' in self.es.attribute_names():
            properties = ['diameters2', 'lengths2']
            for property in properties:
                data = []
                data.extend(self.es[adjacent[0]][property])
                if neighbors[0] > orderTwoVertex:
                    data = data[::-1]
                if neighbors[1] < orderTwoVertex:
                    data.extend(self.es[adjacent[1]][property][::-1])
                else:
                    data.extend(self.es[adjacent[1]][property])
                self.es[newEdge][property] = np.array(data)

        self.es[newEdge]['length'] = np.sum(self.es[newEdge]['lengths2'])
        resistanceTotSimplified=np.sum(np.array(self.es[newEdge]['lengths2'])/(np.array(self.es[newEdge]['diameters2'])**4))
        self.es[newEdge]['diameter']=(self.es[newEdge]['length']/resistanceTotSimplified)**(0.25)
        if 'nkind' in self.es.attribute_names():
            nkinds=self.es[adjacent]['nkind']
            if len(np.unique(nkinds)) > 1:
                print('WARNING different vessel types are combined')
                print(nkinds)
                if 2 in nkinds:
                    nkinds.remove(2)
                    self.es[newEdge]['nkind']=nkinds[0]
                elif 3 in nkinds:
                    nkinds.remove(3)
                    self.es[newEdge]['nkind']=nkinds[0]
                elif 0 in nkinds:
                    nkinds.remove(0)
                    self.es[newEdge]['nkind']=nkinds[0]
                elif 1 in nkinds:
                    nkinds.remove(1)
                    self.es[newEdge]['nkind']=nkinds[0]
                elif 5 in nkinds:
                    nkinds.remove(5)
                    self.es[newEdge]['nkind']=nkinds[0]
            else:
                self.es[newEdge]['nkind']=nkinds[0]
        
        mapNkindToKind={0:'pa',1:'pv',2:'a',3:'v',4:'c',5:'n'}
        self.es[newEdge]['kind']=mapNkindToKind[self.es[newEdge]['nkind']]

        self.delete_vertices(orderTwoVertex)                      
        
    #--------------------------------------------------------------------------                                                   

    def delete_order_two_vertices(self, **kwargs):
        """Joins all edges adjacent to order two vertices in the VascularGraph.
        INPUT: **kwargs
               whitelist: A list of order two vertices that should be deleted
                          (others, if they exist, are kept).
               blacklist: A list of order two vertices that should be kept, all
                          others are deleted.
        OUTPUT: None
        """
        degree = self.degree()
        if 'whitelist' in kwargs.keys():
            self.vs['deleteme'] = [1 if v in kwargs['whitelist'] else 0
                                   for v in xrange(self.vcount())]
        elif 'blacklist' in kwargs.keys():
            self.vs['deleteme'] = [1 if (degree[v] == 2) and 
                                   (not v in kwargs['blacklist']) else 0
                                   for v in xrange(self.vcount())]
        else:
            self.vs['deleteme'] = [1 if degree[v] == 2 else 0
                                   for v in xrange(self.vcount())]
        nDelete = len(self.vs(deleteme_eq=1))
        for i in xrange(nDelete):
            orderTwoVertex = self.vs(deleteme_eq=1)[0].index
            self.join_edges(orderTwoVertex)

        del self.vs['deleteme']
        
    #--------------------------------------------------------------------------                                                   

    def delete_selfloops(self):
        """Removes all selfloops from the VascularGraph.
        INPUT: None
        OUTPUT: None
        """
        self.delete_edges(np.nonzero(self.is_loop())[0].tolist())

                                      
    #--------------------------------------------------------------------------
    # geometry methods
    #--------------------------------------------------------------------------    
    
    def central_dilation(self, eid, factor, cf):
        """Splits an edge into a central and two distal parts. The central part
        of the edge is dilated by a given factor.
        INPUT: eid: Index of the edge that is to be dilated.
               factor: Multiplicative factor of dilation.
               cf: Center fraction. E.g. cf=2/3 would split the edge into 1/6,
               4/6, 1/6. HOWEVER, the resulting splitting position depends on the number
               of available points and the spacing between the points.
        OUTPUT: Vertex and edge indices are returned as a tuple and triple
                respectively.
        """
        npoints = len(self.es[eid]['points'])
        # An edge consisting of less than 4 points cannot be centrally-dilated.
        # Return error code -1 and leave VascularGraph unchanged:
        if npoints < 4:
            print('Edge has less than four points and thus cannot be centrally dilated!')
            return -1
        ec = self.ecount()
        vids = self.es[eid].tuple
        new_eids = (ec-1, ec, ec+1)
        delete_eids = [eid, ec+1]
        pIndex = int(max(1, np.floor(npoints * ((1-cf)/2))))

        self.split_edge(eid, pIndex, False)
        self.split_edge(self.ecount()-1, pIndex, False)
        self.delete_edges(delete_eids) 
        dilated_eid=ec+1

        self.es[dilated_eid]['diameter'] *= factor
        self.es[dilated_eid]['diameters'] *= factor
        return vids, new_eids, dilated_eid

    def add_points(self, spacing,edgeList=None):
        """Adds intermediate points between vertices as edge-property.
        Diameters and lengths are added as well.
        INPUT: spacing: The space between points.
        OUTPUT: None, edge property 'points', 'diameters', and 'lengths' are
                added.
        """
       
        if edgeList==None:
            edges=range(self.ecount())
        else:
            edges=edgeList

        for eI in edges: 
            e=self.es[eI]
            rs = self.vs[e.source]['r']
            rt = self.vs[e.target]['r']
            dist = np.linalg.norm(rs-rt)
            if dist == 0:
                e['points'] = [rs, rs, rs]
                d = e['diameter']
                e['diameters'] = np.array([d] * 3)
                e['diameters2'] = np.array([d] * 2)
                l = e['length']
                e['lengths'] = np.array([l/3.] * 3)
                e['lengths2'] = np.array([l/2.] * 2)
            else:
                n = max(int(round(dist / spacing)), 1)
                v = (rt - rs) / n
                e['points'] = [rs + x*v for x in xrange(n+1)]
                d = e['diameter']
                e['diameters'] = np.array([d] * (n+1))
                e['diameters2'] = np.array([d] * (n))
                lengths2=[]
                for j in range(len(e['points'])-1):
                    lengthsNew = np.linalg.norm(e['points'][j] - e['points'][j+1])
                    lengths2.append(lengthsNew)
                e['lengths2']=lengths2
                lengths=[]
                for j in range(len(e['points'])):
                    if j == 0:
                        lengths.append(e['lengths2'][j]/2.)
                    elif j == len(e['points'])-1:
                        lengths.append(e['lengths2'][j-1]/2.)
                    else:
                        lengths.append(0.5*e['lengths2'][j]+0.5*e['lengths2'][j-1])
                e['lengths']=lengths

    def add_lengths2(self, edgeList=None):
        """Adds lengths2 and diameters2 as edge attriute. Lengths2 and diameters2
        are properties of the segments of an edge defined by points. (lengths, diameters = attributes
        for the points, lengths2, diameters2 = attributes for the segements defined by the points)
        INPUT: edgeList
        OUTPUT: None, edge property 'diameters2', and 'lengths2' are
                added.
        """

        if edgeList==None:
            edges=range(self.ecount())
        else:
            edges=edgeList

        for eI in edges: 
            e=self.es[eI]
            lengths2=[]
            for j in range(len(e['points'])-1):
                lengthsNew = np.linalg.norm(e['points'][j] - e['points'][j+1])
                lengths2.append(lengthsNew)
            e['lengths2']=lengths2
            lengths=[]
            for j in range(len(e['points'])):
                if j == 0:
                    lengths.append(e['lengths2'][j]/2.)
                elif j == len(e['points'])-1:
                    lengths.append(e['lengths2'][j-1]/2.)
                else:
                    lengths.append(0.5*e['lengths2'][j]+0.5*e['lengths2'][j-1])
            e['lengths']=lengths
            diameters2=[]
            resistances=[]
            for j in range(len(e['points'])-1):
                if j == 0 or j == len(e['points'])-2:
                    diameters2.append((e['diameters'][j]+e['diameters'][j+1])/2)
                else:
                    diameters2.append((e['diameters'][j]*e['lengths2'][j]+e['diameters'][j+1]*e['lengths2'][j+1])/(e['lengths2'][j]+e['lengths2'][j+1]))
                resistances.append(e['lengths2'][j]/(diameters2[-1]**4))
            resistanceTot=np.sum(resistances)
            e['length']=np.sum(e['lengths'])
            e['effDiam']=(e['length']/resistanceTot)**(0.25)



    def radius_and_center(self,shape='cylinder'):
        """Computes the approximate radius and center of the circular cross-
        section of a cylindrical graph (the rotational axis is assumed to lie 
        in z-direction). If the shape is cubic the center is computer and the 
        radius is the radius in which all of the nodes can be found
        INPUT: shape 'cylinder' or 'cube'.
        OUTPUT: radius: The approximate radius of the circular cross-section.
                center: The approximate center of the circular cross-section.
        """
    
        if shape == 'cylinder':
            radius = max(((sp.amax(self.vs['r'],0) - 
                           sp.amin(self.vs['r'],0)) / 2.0)[:-1])
            center = sp.amin(self.vs['r'],0)[:-1] + radius
        elif shape == 'cube':
            center = np.mean(self.vs['r'],axis=0)[0:2]
            radius = np.max([np.max(np.max(self.vs['r'],axis=0)[0:2]-center),np.max(center - np.min(self.vs['r'],axis=0)[0:2])])
    
        return radius, center

    #--------------------------------------------------------------------------
        
    def dimension_extrema(self):
        """Retrieves the minimum and maximum x,y,z values of the points making
        up the graph, as well as the corresponding lengths.
        INPUT: None (except self).
        OUTPUT: minima, maxima, lengths as numpy arrays.
        """
        r = np.vstack(self.es['points'])
        minima = np.amin(r,0)
        maxima = np.amax(r,0)
        lengths = maxima - minima
        return minima, maxima, lengths
    
    #--------------------------------------------------------------------------
    
    def shape(self):
        """Determines the shape of a vascular graph (this may be either 
        'cylinder' or 'cuboid'). This is done on the basis of determining 
        whether the lower left corner of the vascular graph, looked at in 
        z-direction, is devoid of points. If so, it is cylindrical.
        INPUT: None (except self)
        OUTPUT: Either 'cylinder' or 'cuboid' as string.
        """
        r = sp.vstack(self.es['points'])
        minima = np.amin(r,0)
        maxima = np.amax(r,0)
        
        length = np.mean((maxima[0]-minima[0], maxima[1]-minima[1])) / 2.0
        factor = 0.25 # A circle enscribed by a square touches the square at 
                      # four points. The small square one can draw at the 
                      # bottom left corner is sin(pi/4) * (sqrt(2)-1) the size 
                      # of the big square. The factor 0.25 is a (conservative) 
                      # approximation of this scaling factor.
        in_small_sq = filter(lambda x: (x[0] <= minima[0]+length*factor) and \
                                       (x[1] <= minima[1]+length*factor),r)
        if len(in_small_sq) == 0:
            return 'cylinder'
        else:
            return 'cuboid'    
    
    #--------------------------------------------------------------------------
    
    def cross_sectional_area_and_length(self,**kwargs):
        """Computes the approximate cross-sectional area and length of the 
        vascular graph.
        INPUT: **kwargs               
                 shape: The shape of the vascular graph. This may be either 
                        'cuboid' or 'cylinder'. If not provided, the shape is 
                        determined from the data.
        OUTPUT: The cross-sectional area and length of the vascular graph.
        """
        if kwargs.has_key('shape'):
            gShape = kwargs['shape']
        else:
            gShape = self.shape()
    
        r = sp.vstack(self.es['points'])
        minima = sp.amin(r,0)
        maxima = sp.amax(r,0)
        length   = maxima[2]-minima[2]
        
        if gShape == 'cuboid':
            return (sp.prod(maxima[:-1]-minima[:-1]), length)
        elif gShape == 'cylinder':
            diameter = sp.mean((maxima[0]-minima[0],maxima[1]-minima[1]))
            return (sp.pi * diameter**2 / 4., length)    

    #--------------------------------------------------------------------------
    
    
    def vertex_volume(self, vertices=None):
        """Computes the volume associated with graph vertices from the volumes
        of their respective adjacent edges. If a vertex is an upscaled vertex,
        also its upscaled volume is considered.
        INPUT: vertices: List of vertices for which to compute the volume. If
                         not provided, all vertices are considered.
        OUTPUT: List of vertex volumes.                  
        """
        if vertices is None:
            vertices = xrange(self.vcount())

#        if 'volume' not in self.es.attribute_names():
#            self.es['volume'] = [np.pi * e['diameter']**2.0 / 4.0 * e['length'] 
#                                 for e in self.es]
            
        volumes = []                
        for vertex in vertices:
            volumes.append(
                sum(self.es(self.adjacent(vertex, 'all'))['volume']) / 2.0)                    
            
        if 'uVolume' in self.vs.attribute_names():
            for i, vertex in enumerate(vertices):
                if self.vs[vertex]['uVolume'] is not None:
                    volumes[i] += self.vs['uVolume']
        
        return volumes


    #--------------------------------------------------------------------------
    
        
    def total_volume(self,**kwargs):
        """Computes the approximate total volume of the vascular graph (i.e. 
        the combined volumina of tissue and vasculature).
        INPUT: **kwargs
                 shape: The shape of the vascular graph. This may be either 
                        'cuboid' or 'cylinder'. If not provided, the shape is 
                        determined from the data.
                 zRange: The range of cortical depths to consider (as list).       
        OUTPUT: The total volume of the vascular graph.
        WARNING: zRange is not tested for consistency (i.e. values that lie 
                 outside of the actual domain are quietly accepted).
        """
        area, length = self.cross_sectional_area_and_length(**kwargs)
        if kwargs.has_key('zRange'):
            if not 'depth' in self.es.attribute_names():
                misc.update_depth(self)
            length = kwargs['zRange'][1] - kwargs['zRange'][0]
        return area * length
        
    #--------------------------------------------------------------------------    
    
    def vascular_volume(self, **kwargs):
        """Computes the vascular volume of the vascular graph. This makes use 
        of the vessel volume property, if defined, otherwise it is computed 
        using the vessel length and mean diameter. (Note that the latter may 
        be either exact or approximate, depending on how the mean diameter is
        defined).
        Note that this is typically using the *inner* diameter of the vessels.
        INPUT: **kwargs
                 zRange: The range of cortical depths to consider (as list).
                 wallThickness: The thickness of the vessel wall to be added
                                to the inner diameter (assumed equal for all
                                vessels). This parameter is only taken into 
                                account when the edge property 'volume' is not
                                defined.
        OUTPUT: The vascular volume of the vascular graph.
        """
        if kwargs.has_key('zRange'):
            if not 'depth' in self.es.attribute_names():
                misc.update_depth(self)
            edges = self.es(depth_ge=kwargs['zRange'][0],
                         depth_le=kwargs['zRange'][1])
        else:
            edges = self.es    
                
        if 'volume' in self.es.attribute_names():        
            return sum(edges['volume'])
        else:
            #log.warning("Using V = l * d_mean")
            if kwargs.has_key('wallThickness'):
                wt = kwargs['wallThickness']
            else:
                wt = 0.0
            return sum([x[0] * sp.pi * (x[1]+wt*2.0)**2 / 4. for x in
                       zip(edges['length'], edges['diameter'])])    
            
    #--------------------------------------------------------------------------

    def vascular_volume_fraction(self,**kwargs):
        """Computes the approximate vascular volume fraction of the vascular 
        graph.
        Note that this is typically using the *inner* diameter of the vessels.
        INPUT: **kwargs
                 shape: The shape of the vascular graph. This may be either 
                        'cuboid' or 'cylinder'. If not provided, the shape is 
                        determined from the data.
                 zRange: The range of cortical depths to consider (as list).       
                 wallThickness: The thickness of the vessel wall to be added
                                to the inner diameter (assumed equal for all
                                vessels). This parameter is only taken into 
                                account when the edge property 'volume' is not
                                defined.
        OUTPUT: The approximate vascular volume fraction of the vascular graph.
        WARNING: zRange is not tested for consistency (i.e. values that lie 
                 outside of the actual domain are quietly accepted).    
        """
        return self.vascular_volume(**kwargs) / self.total_volume(**kwargs)    

    #--------------------------------------------------------------------------
    
    def length_density(self, dThreshold = 7.0, **kwargs):
        """Compute the length density of a sample - capillaries, 
        non-capillaries and combined. Either for the entire sample or a given 
        z-fraction of it. Published values in the macaque visual cortex with 
        dThreshold = 8.0 micron outer diameter(Weber 2008), approx. 7.0 micron
        inner diameter are
        capillaries: [300, 400] mm/mm^3
        non-capillaries: [100,150] mm/mm^3 
        INPUT: dThreshold: (Inner) diameter threshold below which vessels are 
                           considered as capillaries (optional, default is 7.0)             
               **kwargs:   
                 shape: The shape of the vascular graph. This may be either 
                        'cuboid' or 'cylinder'. If not provided, the shape is 
                        determined from the data.           
                 zRange: Range of zValues in which to consider the edges of the
                         graph (as list e.g. [0,1000]). If not supplied, all 
                         edges are considered.                     
        OUTPUT: Capillary length-density, non-capillary length-density, total
                length-density.
        WARNING: This function assumes that the vascular graph is either
                 cylindrical with rotational axis in z-direction or cuboid.
        """
        cEdges = self.es(diameter_le=dThreshold)
        ncEdges = self.es(diameter_gt=dThreshold)
        cLength = sum(cEdges['length'])
        ncLength = sum(ncEdges['length'])
        totalLength = cLength + ncLength
        vascularVolume = self.vascular_volume(**kwargs)
        totalVolume = self.total_volume(**kwargs)
        
        return cLength/totalVolume, \
               ncLength/totalVolume, \
               totalLength/totalVolume      
                        
        

    #--------------------------------------------------------------------------
    # substance related methods
    #--------------------------------------------------------------------------
    
    def add_substance(self,substance,concentration=0.0):
        """Adds a substance to the vascular graph. This is a scalar property.
        INPUT: substance: The name of the substance as string. 
               concentration: The initial concentration to be set at all 
                              vertices. The concentration is a volume fraction 
                              and hence a dimensionless number between 0 and 1.
                              (Optional parameter, default is 0.)
        OUTPUT: none, the instance is modified in place.                              
        """
        if 'substance' not in self.vs.attribute_names():                
            for v in self.vs:
                v['substance'] = {substance: concentration}
        else:
            for v in self.vs:
                v['substance'][substance] = concentration

    #--------------------------------------------------------------------------
    

    def add_exchange_coefficient(self, substance, value=0.0):
        """Adds a uniform exchange coefficient to the VascularGraph. The 
        exchange coefficient describes the transfer of substances between the 
        vasculature and the tissue. It is an edge property and depends on the 
        substance. Therefore, it is possible to assign individual exchange 
        coefficients for different substances. Upscaled vertices store an 
        effective upscaled exchange coefficient for its unresolved associated 
        upscaled edges.   
        INPUT: substance: The name of the substance as string.
               value: The value of the exchange coefficient.
        """
        if 'exchangeCoefficient' not in self.es.attribute_names():                
            for e in self.es:
                e['exchangeCoefficient'] = {substance: value}
        else:
            for e in self.es:
                e['exchangeCoefficient'][substance] = value
                
        # Upscaled vertices need to store a representative exchange coefficient
        # of their upscaled edges:
        if 'kind' in self.vs.attribute_names():                
            if 'uExchangeCoefficient' not in self.vs.attribute_names():
                for v in self.vs:
                    if v['kind'] == 'u':
                        v['uExchangeCoefficient'] = {substance: value}
                    else:
                        v['uExchangeCoefficient'] = None
            else:
                for v in self.vs:
                    if v['kind'] == 'u':
                        v['uExchangeCoefficient'][substance] = value
                                
                                
#--------------------------------------------------------------------------
    

    def add_diffusion_factor(self, substance, value=0.0):
        """Adds a uniform diffusion factor to the VascularGraph. The diffusion
        factor describes the diffusive flux of substances within the 
        VascularGraph. It is an edge property and depends on the substance. 
        Therefore, it is possible to assign individual diffusion factors 
        for different substances. Note that this diffusion factor is 
        defined differently from the diffusion coefficient of the literature: 
        its dimensions are [1 / time], as opposed to [length^2 / time]. The 
        advantage of this formulation is that the diffusion factor can be
        defined even for upscaled edges that have no straight-forward 
        cross-sectional surface area. 
        INPUT: substance: The name of the substance as string.
               value: The value of the exchange coefficient.
        """
        if 'diffusionFactor' not in self.es.attribute_names():                
            for e in self.es:
                e['diffusionFactor'] = {substance: value}
        else:
            for e in self.es:
                e['diffusionFactor'][substance] = value

    
    #--------------------------------------------------------------------------

    
    def add_sBCs(self,substance,vertices,concentrations):
        """Adds substance boundary conditions to the vascular graph.
        INPUT: substance: The name of the substance as string. 
               vertices: The indices of the vertices for which the substance
                         boundary conditions are to be set.
               concentrations: vector of boundary concentrations for the 
                               respective vertices (values required in the 
                               range [0,1]).          
        OUTPUT: none, the instance is modified in place.
        """
        if 'sBC' not in self.vs.attribute_names():      
            self.vs['sBC'] = [{} for v in xrange(self.vcount())]
        counter = 0
        for i,v in enumerate(self.vs):
            if i in vertices:
                v['sBC'][substance] = concentrations[counter]
                v['substance'][substance] = concentrations[counter]
                counter += 1
            else:
                v['sBC'][substance] = None
    
    #--------------------------------------------------------------------------

    def get_concentration(self,substance):
        """Returns a copy of the substance concentration as a list.
        INPUT: substance: The name of the substance as string.
        OUTPUT: c: The substance concentration at the vertices as list.
        """
        c = []
        for v in self.vs:
            c.append(v['substance'][substance])
        return c    
    
    #--------------------------------------------------------------------------
    
    def set_concentration(self,substance,c):
        """ Sets the substance concentration at the graph vertices.
        INPUT: substance: The name of the substance as string.
        c: The substance concentration at the vertices as list.
        OUTPUT: none, the instance is modified in place.
        """
        for i, v in enumerate(self.vs):
            v['substance'][substance] = c[i]

    #--------------------------------------------------------------------------
    def get_edges_in_boundingBox_point_based(self,xCoordsBox=[6750,7400],yCoordsBox=[6900,7500],zCoordsBox=[850,1100]):
        """ Outputs edges belonging to a given box. 
        INPUT: xCoords = xmin,xmax
               yCoords = ymin,ymax
               zCoords = zmin,zmax
        OUTPUT: edges_in_box: all edges 
        """

        coordsPoints = np.concatenate(self.es['points'], 0)
        cumsum = np.cumsum([len(p) for p in self.es['points']])
        edges=[[e.index]*len(e['points']) for e in self.es]
        associatedEdges = np.concatenate(edges, 0)
        
        edges_in_box=[]
        for coords,edge in zip(coordsPoints,associatedEdges):
            if edge not in edges_in_box:
                if coords[0] >= xCoordsBox[0] and coords[0] <= xCoordsBox[1]:
                    if coords[1] >= yCoordsBox[0] and coords[1] <= yCoordsBox[1]:
                        if coords[2] >= zCoordsBox[0] and coords[2] <= zCoordsBox[1]:
                            edges_in_box.append(edge)
         
        return  edges_in_box

    #--------------------------------------------------------------------------
    def get_edges_in_boundingBox_vertex_based(self,xCoordsBox=[6750,7400],yCoordsBox=[6900,7500],zCoordsBox=[850,1100]):
        """ Outputs edges belonging to a given box. 
        INPUT: xCoords = xmin,xmax
               yCoords = ymin,ymax
               zCoords = zmin,zmax
        OUTPUT: edges_in_box: all edges completely in edge
                edges_across_border: edges with one vertex in box and one outside
                border_vertices: vertices outside box
        """

        edges_in_box=[]
        edges_across_border=[]
        border_vertices=[]
        for e in self.es:
            vertices = [e.source,e.target]
            vertices_in_box = [0,0]
            for i,v in enumerate(vertices):
                coords = self.vs[v]['r']
                if coords[0] >= xCoordsBox[0] and coords[0] <= xCoordsBox[1]:
                    if coords[1] >= yCoordsBox[0] and coords[1] <= yCoordsBox[1]:
                        if coords[2] >= zCoordsBox[0] and coords[2] <= zCoordsBox[1]:
                            vertices_in_box[i] = 1
            if np.sum(vertices_in_box) == 2:
                edges_in_box.append(e.index)
            elif np.sum(vertices_in_box) == 1:
                edges_across_border.append(e.index)
                if vertices_in_box[0] == 0:
                    border_vertices.append(vertices[0])
                else:
                    border_vertices.append(vertices[1])
        
        edges_in_box = np.unique(edges_in_box)
        edges_across_border = np.unique(edges_across_border)
        border_vertices = np.unique(border_vertices)

        return  edges_in_box, edges_across_border, border_vertices

    #--------------------------------------------------------------------------
    def split_into_volumes(self,xSplits=20):
        """Splits the vascular graph into quads.
        INPUT: xSplit: the number of splits in x-direction.
               the splits in y and z direction are chosen such,
               that cubic volums are obtained
        OUTPUT: volumes: list of all the volumes [xmin,ymin,zmin] of
                the corresponding volume
                dxs: [dx,dy,dz] spacings of the volumes
        """
        origin = np.mean(self.vs['r'], axis=0)[:2]
        #origin = np.mean(G.vs['r'], axis=0)[:2]
        x=[]
        y=[]
        z=[]
        distanceFromOrigin=[]
        for v in self.vs:
        #for v in G.vs:
            r=v['r']
            x.append(r[0])
            y.append(r[1])
            z.append(r[2])
            distanceFromOrigin.append(r[:2] - origin)

        self.vs['x']=x
        self.vs['y']=y
        self.vs['z']=z
        xDist=np.max(x)-np.min(x)
        yDist=np.max(y)-np.min(y)
        zDist=np.max(z)-np.min(z)
        dx=xDist/xSplits
        dy=yDist/(np.ceil(yDist/dx))
        dz=zDist/(np.ceil(zDist/dx))
        radiusMax=np.max(distanceFromOrigin)

        volumes=[]
        xStart=np.min(x)
        yStart=np.min(y)
        zStart=np.min(z)
        xCurrent=np.min(x)
        yCurrent=np.min(y)
        zCurrent=np.min(z)
        for i in range(xSplits):
            print('x')
            print(i)
            print(xCurrent)
            for j in range(int(np.ceil(yDist/dx))):
                x1=xCurrent
                x2=xCurrent+dx
                y1=yCurrent
                y2=yCurrent+dy
                distFromO = [np.linalg.norm([x1,y1] - origin),np.linalg.norm([x1,y2] - origin),np.linalg.norm([x2,y1] - origin),
                    np.linalg.norm([x2,y2] - origin)]
                boolDist=0
                for k in range(4):
                    if distFromO[k] < radiusMax:
                        boolDist=1
                        break
                if boolDist:
                    for k in range(int(np.ceil(yDist/dx))):
                        volumes.append([xCurrent,yCurrent,zCurrent])
                        zCurrent += dz
                else:
                    print('Volume outside')
                zCurrent=zStart
                yCurrent += dy
            yCurrent=yStart
            xCurrent += dx

        dxs=[dx,dy,dz]

        return volumes,dxs

#--------------------------------------------------------------------------

    def edges_of_splitVolumes(self,volumes,dxs):
        """
        uses the function get_edges_in_boundingBox to obtain the edges belonging to one volume.
        For every volume the "allEdges" and the "borderEdges" are defined. For the borderEdges 
        the distance along the edge is defined where the vessel enters the volume
        INPUT: volumes: list of all the volumes [xmin,ymin,zmin] of
                the corresponding volume
                dxs: [dx,dy,dz] spacings of the volumes
        OUTPUT: allEdges list of allEdges belonging to the volumes
                borderEdges: list of all borderEdges belonging to each volume
                relDistBorderEdges: relDist of hte bording edges
        """

        allEdges=[]
        borderEdges=[]
        relDistBorderEdges=[]
        count = 0
        count2 = 0
        emptyVolumes=[]
        eps = finfo(float).eps * 1e4
        print('Total number of volumes')
        print(len(volumes))
        for i in volumes:
            print('Volume')
            print(count)
            xmin=i[0]
            xmax=i[0]+dxs[0]
            ymin=i[1]
            ymax=i[1]+dxs[1]
            zmin=i[2]
            zmax=i[2]+dxs[2]
            boolEmptyVol=0
            filename='Box/GDummy'
            allEdgeCurrent,borderEdgesCurrent, internalEdgesCurrent = self.get_edges_in_boundingBox(xCoords=[xmin,xmax],
                yCoords=[ymin,ymax],zCoords=[zmin,zmax],outputName=filename)
            allEdges.append(allEdgeCurrent)
            borderEdges.append(borderEdgesCurrent)
            if allEdgeCurrent==[]:
                emptyVolumes.append(count)
                print('Empty Volume')
                boolEmptyVol=1
            else:
                filename='Box/GBox'+str(count2)
                allEdgeCurrent,borderEdgesCurrent, internalEdgesCurrent = self.get_edges_in_boundingBox(xCoords=[xmin,xmax],
                    yCoords=[ymin,ymax],zCoords=[zmin,zmax],outputName=filename)
                count2 += 1
            pointIn=[]
            pointOut=[]
            boolInOutFound=0
            DistBorderEdgesDummy=[]
            #plane: 1: xmin, 2: xmax, 3: ymin, 4:ymax, 5: zmin, 6: zmax
            for j in borderEdgesCurrent:
                intersectionPlane = []
                sourceV=self.es[int(j)].source
                sourceCoords=self.vs[sourceV]['r']
                if sourceCoords[0] <= xmax and sourceCoords[0] >= xmin and sourceCoords[1] <= ymax and sourceCoords[1] >= ymin and sourceCoords[2] <= zmax and sourceCoords[2] >= zmin:
                    fromInToOut = 1
                else:
                    fromInToOut = 0
                for k,m in enumerate(self.es[int(j)]['points']):
                #for k,m in enumerate(G.es[int(j)]['points']):
                    if fromInToOut:
                        if m[0] < xmin or m[0] > xmax or m[1] < ymin or m[1] > ymax or m[2] < zmin or m[2] > zmax:
                            if m[0] < xmin:
                               intersectionPlane.append(1) 
                            elif m[0] > xmax:
                               intersectionPlane.append(2) 
                            if m[1] < ymin:
                               intersectionPlane.append(3) 
                            elif m[1] > ymax:
                               intersectionPlane.append(4) 
                            if m[2] < zmin:
                               intersectionPlane.append(5) 
                            elif m[2] > zmax:
                               intersectionPlane.append(6) 
                            pointOut.append(k)
                            pointIn.append(k-1)
                            boolInOutFound = 1
                            break
                    else:
                        if m[0] >= xmin and m[0] <= xmax and m[1] >= ymin and m[1] <= ymax and m[2] >= zmin and m[2] <= zmax:
                            n=self.es[int(j)]['points'][k-1]
                            #n=G.es[int(j)]['points'][k-1]
                            if n[0] < xmin:
                               intersectionPlane.append(1) 
                            elif n[0] > xmax:
                               intersectionPlane.append(2) 
                            if n[1] < ymin:
                               intersectionPlane.append(3) 
                            if n[1] > ymax:
                               intersectionPlane.append(4) 
                            if n[2] < zmin:
                               intersectionPlane.append(5) 
                            if n[2] > zmax:
                               intersectionPlane.append(6) 
                            pointOut.append(k-1)
                            pointIn.append(k)
                            boolInOutFound = 1
                            break
                if not boolInOutFound:
                    print('ERROR boording points not found')
                    break
                pLine=self.es[int(j)]['points'][pointOut[-1]]
                vLine=np.array(self.es[int(j)]['points'][pointOut[-1]])-np.array(self.es[int(j)]['points'][pointIn[-1]])
                iP=[]
                for m in intersectionPlane:
                    if m == 1 or m == 2:
                        nPlane = [1,0,0]
                        if m == 1:
                            pPlane=[xmin,0,0]
                        else:
                            pPlane=[xmax,0,0]
                        iP=misc.intersection_plane_line(pPlane,nPlane,pLine,vLine) 
                        if iP[1] > ymax or iP[1] < ymin or iP[2] > zmax or iP[2] < zmin:
                            print('Intersection point not in surface')
                        else:
                            break
                    elif m == 3 or m == 4:
                        nPlane = [0,1,0]
                        if m == 3:
                            pPlane=[0,ymin,0]
                        else:
                            pPlane=[0,ymax,0]
                        iP=misc.intersection_plane_line(pPlane,nPlane,pLine,vLine) 
                        if iP[0] > xmax or iP[0] < xmin or iP[2] > zmax or iP[2] < zmin:
                            print('Intersection point not in surface')
                        else:
                            break
                    elif m == 5 or m == 6:
                        nPlane = [0,0,1]
                        if m == 5:
                            pPlane=[0,0,zmin]
                        else:
                            pPlane=[0,0,zmax]
                        iP=misc.intersection_plane_line(pPlane,nPlane,pLine,vLine) 
                        if iP[1] > ymax or iP[1] < ymin or iP[0] > xmax or iP[0] < xmin:
                            print('Intersection point not in surface')
                        else:
                            break
                if iP == []:
                    print('ERROR no intersection point found')
                    print(j)
                    break
                else:
                    #compute distance to source point
                    iP=np.array(iP,dtype=np.float_)
                    if pointIn[-1] < pointOut[-1]:
                       distPoints=np.linalg.norm(iP-np.array(self.es[int(j)]['points'][pointIn[-1]])) 
                       DistBorderEdgesDummy.append(np.sum(self.es[int(j)]['lengths2'][:pointOut[-1]])+distPoints)
                    else:
                       distPoints=np.linalg.norm(iP-np.array(self.es[int(j)]['points'][pointOut[-1]])) 
                       DistBorderEdgesDummy.append(np.sum(self.es[int(j)]['lengths2'][:pointIn[-1]])+distPoints)
            relDistBorderEdges.append(DistBorderEdgesDummy)
            if len(borderEdgesCurrent) != len(DistBorderEdgesDummy):
                print('ERROR no intersection point found')
                print(j)
                vgm.write_pkl(volumes,'volumesBackUp.pkl')
                vgm.write_pkl(allEdges,'allEdgesBackUp.pkl')
                vgm.write_pkl(borderEdges,'borderEdgesBackUp.pkl')
                vgm.write_pkl(relDistBorderEdges,'relDistBorderEdgesBackUp.pkl')
                break
            count += 1
            if not boolInOutFound and not boolEmptyVol:
                print('ERROR boording points not found')
                break
                            
        if len(volumes) != len(allEdges) or len(volumes) != len(borderEdges) or len(volumes) != len(relDistBorderEdges) or len(allEdges) != len(borderEdges) \
            or len(allEdges) != len(relDistBorderEdges) or len(borderEdges) != len(relDistBorderEdges):
            print('ERROR')
            print(len(volumes))
            print(len(allEdges))
            print(len(borderEdges))
            print(len(relDistBorderEdges))
            vgm.write_pkl(volumes,'volumesBackUp.pkl')
            vgm.write_pkl(allEdges,'allEdgesBackUp.pkl')
            vgm.write_pkl(borderEdges,'borderEdgesBackUp.pkl')
            vgm.write_pkl(relDistBorderEdges,'relDistBorderEdgesBackUp.pkl')

        vgm.write_pkl(volumes,'volumesBackUp.pkl')
        vgm.write_pkl(allEdges,'allEdgesBackUp.pkl')
        vgm.write_pkl(borderEdges,'borderEdgesBackUp.pkl')
        vgm.write_pkl(relDistBorderEdges,'relDistBorderEdgesBackUp.pkl')

        volumes2=[]
        allEdges2=[]
        borderEdges2=[]
        relDistBorderEdges2=[]
        for i in range(len(volumes)):
            if i not in emptyVolumes:
                volumes2.append(volumes[i])
                allEdges2.append(allEdges[i])
                borderEdges2.append(borderEdges[i])
                relDistBorderEdges2.append(relDistBorderEdges[i])

        return volumes2, allEdges2, borderEdges2, relDistBorderEdges2
 
#---------------------------------------------------------
    def assign_splitToEdges(self,volumes,allEdges,borderEdges,relDistBorderEdges):
        """ Assigns vTypes of vertices base on current pressure distribution 
        INPUT: 
        OUTPUT: 
        """

        volsList=[]
        relDistList=[]
        absDistList=[]
        self.es['borderEdges']=[0]*self.ecount()
        for i in range(self.ecount()):
        #for i in range(G.ecount()):
            volsList.append([])
            relDistList.append([0])
            absDistList.append([0])

        for i in range(len(volumes)):
            for j in allEdges[i]:
                volsList[j].append(i)
                if j not in borderEdges[i]:
                    relDistList[j].append(None)
                    absDistList[j].append(None)
                else:
                    relDistList[j].append(relDistBorderEdges[i][borderEdges[i].tolist().index(j)])
                    relDistList[j].append((relDistBorderEdges[i][borderEdges[i].tolist().index(j)])*self.es[j]['length'])
                    #G.es[j]['borderEdges']=1
                    self.es[j]['borderEdges']=1

        for i in range(sielf.ecount()):
        #for i in range(G.ecount()):
            relDistList[i].append(1)
            absDistList[i].append(1)
            
        self.es['vols']=volsList
        self.es['relDistList']=relDistList
        self.es['absDistList']=absDistList
        self['numberOfVols']=len(volumes)
        
#---------------------------------------------------------
    def assign_vType(self):
        """ Assigns vTypes of vertices base on current pressure distribution 
        INPUT: 
        OUTPUT: 
        """
        G=self
        #Beginning   
        inEdges=[]
        outEdges=[]
        divergentV=[]
        convergentV=[]
        connectingV=[]
        doubleConnectingV=[]
        noFlowV=[]
        noFlowE=[]
        vertices=[]
        count=0
        if not 'sign' in G.es.attributes() or not 'signOld' in G.es.attributes():
            for v in G.vs:
                vI=v.index
                outE=[]
                inE=[]
                noFlowE=[]
                pressure = G.vs[vI]['pressure']
                adjacents=G.adjacent(vI)
                for j,nI in enumerate(G.neighbors(vI)):
                    #outEdge
                    if pressure > G.vs[nI]['pressure']:
                        outE.append(adjacents[j])
                    elif pressure == G.vs[nI]['pressure']:
                        noFlowE.append(adjacents[j])
                    #inflowEdge
                    else: #G.vs[vI]['pressure'] < G.vs[nI]['pressure']
                        inE.append(adjacents[j])
                #Group into divergent, convergent and connecting Vertices
                if len(outE) > len(inE) and len(inE) >= 1:
                    divergentV.append(vI)
                elif len(inE) > len(outE) and len(outE) >= 1:
                    convergentV.append(vI)
                elif len(inE) == len(outE) and len(inE) == 1:
                    connectingV.append(vI)
                elif len(inE) == len(outE) and len(inE) == 2:
                    doubleConnectingV.append(vI)
                elif vI in G['av']:
                    if len(inE) == 0 and len(outE) == 1:
                        pass
                    elif len(inE) == 1 and len(outE) == 0:
                        print('WARNING1 boundary condition changed: from av --> vv')
                        print(vI)
                        G.vs[vI]['av'] = 0
                        G.vs[vI]['vv'] = 1
                        G.vs[vI]['vType'] = 2
                        edgeVI=G.adjacent(vI)[0]
                        if 'httBC' in G.es.attribute_names():
                            G.es[edgeVI]['httBC']=None
                            G.es[edgeVI]['posFirst_last']=None
                            G.es[edgeVI]['v_last']=None
                            print(G.es[edgeVI]['v_last'])
                    elif len(inE) == 0 and len(outE) == 0:
                        print('WARNING changed to noFlow edge')
                        edgeVI=G.adjacent(vI)[0]
                        noFlowV.append(vI)
                        noFlowE.append(edgeVI)
                    else:
                        print('ERROR in defining in and outlets')
                        print(vI)
                elif vI in G['vv']:
                    edgeVI=G.adjacent(vI)[0]
                    if len(inE) == 1 and len(outE) == 0:
                        pass
                    elif len(inE) == 0 and len(outE) == 1:
                        print('WARNING1 boundary condition changed: from vv --> av')
                        print(vI)
                        G.vs[vI]['av'] = 1
                        G.vs[vI]['vv'] = 0
                        G.vs[vI]['vType'] = 1
                        if 'httBC' in G.es.attribute_names() and 'httBC_init' in G.es.attribute_names():
                            G.es[edgeVI]['httBC']=G.es[edgeVI]['httBC_init']
                            if 'rRBC' in G.es.attribute_names():
                                if len(G.es[edgeVI]['rRBC']) > 0:
                                    if G.es['sign'][edgeVI] == 1:
                                        G.es[edgeVI]['posFirst_last']=G.es['rRBC'][edgeVI][0]
                                    else:
                                        G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]-G.es['rRBC'][edgeVI][-1]
                                else:
                                    G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]
                            G.es[edgeVI]['v_last']=0
                            print(G.es[edgeVI]['v_last'])
                    elif len(inE) == 0 and len(outE) == 0:
                        print('WARNING changed to noFlow edge')
                        noFlowV.append(vI)
                        noFlowE.append(edgeVI)
                    else:
                        print('ERROR in defining in and outlets')
                        print(vI)
                else:
                    for i in G.adjacent(vI):
                        #if G.es['flow'][i] > 5.0e-08:
                        #    print('FLOWERROR')
                        #    print(vI)
                        #    print(inE)
                        #    print(outE)
                        #    print(noFlowE)
                        #    print(i)
                        #    print('Flow and diameter')
                        #    print(G.es['flow'][i])
                        #    print(G.es['diameter'][i])
                        noFlowE.append(i)
                    inE=[]
                    outE=[]
                    noFlowV.append(vI)
                inEdges.append(inE)
                outEdges.append(outE)
            G.vs['inflowE']=inEdges
            G.vs['outflowE']=outEdges
            G.es['noFlow']=[0]*G.ecount()
            noFlowE=np.unique(noFlowE)
            if len(noFlowE) > 0:
                G.es[noFlowE]['noFlow']=[1]*len(noFlowE)
            G['divV']=divergentV
            G['conV']=convergentV
            G['connectV']=connectingV
            G['dConnectV']=doubleConnectingV
            G['noFlowV']=noFlowV
            #vertex type av = 1, vv = 2,divV = 3, conV = 4, connectV = 5, dConnectV = 6, noFlowV = 7
            G.vs['vType']=[0]*G.vcount()
            G['av']=G.vs(av_eq=1).indices
            G['vv']=G.vs(vv_eq=1).indices
            for i in G['av']:
                G.vs[i]['vType']=1
            for i in G['vv']:
                G.vs[i]['vType']=2
            for i in G['divV']:
                G.vs[i]['vType']=3
            for i in G['conV']:
                G.vs[i]['vType']=4
            for i in G['connectV']:
                G.vs[i]['vType']=5
            for i in G['dConnectV']:
                G.vs[i]['vType']=6
            for i in G['noFlowV']:
                G.vs[i]['vType']=7
            if len(G.vs(vType_eq=0).indices) > 0:
                print('BIGERROR vertex type not assigned')
                print(len(G.vs(vType_eq=0).indices))
            del(G['divV'])
            del(G['conV'])
            del(G['connectV'])
            del(G['dConnectV'])
            print('Number of noFlow vertices')
            print(len(noFlowV))
        #Every Time Step
        else:
            if G.es['sign']!=G.es['signOld']:
                sign=np.array(G.es['sign'])
                signOld=np.array(G.es['signOld'])
                sumTes=abs(sign+signOld)
                #find edges where sign change took place
                edgeList=np.array(np.where(sumTes < abs(2))[0])
                edgeList=edgeList.tolist()
                sign0=G.es(sign_eq=0,signOld_eq=0).indices
                for e in sign0:
                    edgeList.remove(e)
                stdout.flush()
                vertices=[]
                for e in edgeList:
                    for vI in G.es[int(e)].tuple:
                        vertices.append(vI)
                vertices=np.unique(vertices)
                count = 0
                for vI in vertices:
                    #vI=v.index
                    count += 1
                    vI=int(vI)
                    outE=[]
                    inE=[]
                    noFlowE=[]
                    neighbors=G.neighbors(vI)
                    pressure = G.vs[vI]['pressure']
                    adjacents=G.adjacent(vI)
                    for j in range(len(neighbors)):
                        nI=neighbors[j]
                        #outEdge
                        if pressure > G.vs[nI]['pressure']:
                            outE.append(adjacents[j])
                        elif pressure == G.vs[nI]['pressure']:
                            noFlowE.append(adjacents[j])
                        #inflowEdge
                        else: #G.vs[vI]['pressure'] < G.vs[nI]['pressure']
                            inE.append(adjacents[j])
                    #Group into divergent, convergent, connecting, doubleConnecting and noFlow Vertices
                    #it is now a divergent Vertex
                    if len(outE) > len(inE) and len(inE) >= 1:
                        #Find history of vertex
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow']=[0]*len(inE)
                            G.es[outE]['noFlow']=[0]*len(outE)
                        G.vs[vI]['vType']=3
                        G.vs[vI]['inflowE']=inE
                        G.vs[vI]['outflowE']=outE
                    #it is now a convergent Vertex
                    elif len(inE) > len(outE) and len(outE) >= 1:
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow']=[0]*len(inE)
                            G.es[outE]['noFlow']=[0]*len(outE)
                        G.vs[vI]['vType']=4
                        G.vs[vI]['inflowE']=inE
                        G.vs[vI]['outflowE']=outE
                    #it is now a connecting Vertex
                    elif len(outE) == len(inE) and len(outE) == 1:
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow']=[0]*len(inE)
                            G.es[outE]['noFlow']=[0]*len(outE)
                        G.vs[vI]['vType']=5
                        G.vs[vI]['inflowE']=inE
                        G.vs[vI]['outflowE']=outE
                    #it is now a double connecting Vertex
                    elif len(outE) == len(inE) and len(outE) == 2:
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow']=[0]*len(inE)
                            G.es[outE]['noFlow']=[0]*len(outE)
                        G.vs[vI]['vType']=6
                        G.vs[vI]['inflowE']=inE
                        G.vs[vI]['outflowE']=outE
                    elif vI in G['av']:
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow']=[0]*len(inE)
                            G.es[outE]['noFlow']=[0]*len(outE)
                        if G.vs[vI]['rBC'] != None:
                            for j in G.adjacent(vI):
                                if G.es[j]['flow'] > 1e-6:
                                    print(' ')
                                    print(vI)
                                    print(len(G.vs[vI]['inflowE']))
                                    print(len(G.vs[vI]['outflowE']))
                                    print('ERROR flow direction of inlet vertex changed')
                                    print(G.es[G.adjacent(vI)]['flow'])
                                    print(G.vs[vI]['rBC'])
                                    print(G.vs[vI]['kind'])
                                    print(G.vs[vI]['isSrxtm'])
                                    print(G.es[G.adjacent(vI)]['sign'])
                                    print(G.es[G.adjacent(vI)]['signOld'])
                        else:
                            print('WARNING direction out vv changed to av')
                            print(vI)
                            G.vs[vI]['av'] = 1
                            G.vs[vI]['vv'] = 0
                            G.vs[vI]['vType'] = 1
                            edgeVI=G.adjacent(vI)[0]
                            if 'httBC' in G.es.attribute_names():
                                G.es[edgeVI]['httBC']=G.es[edgeVI]['httBC_init']
                                if len(G.es[edgeVI]['rRBC']) > 0:
                                    if G.es['sign'][edgeVI] == 1:
                                        G.es[edgeVI]['posFirst_last']=G.es['rRBC'][edgeVI][0]
                                    else:
                                        G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]-G.es['rRBC'][edgeVI][-1]
                                else:
                                    G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]
                                G.es[edgeVI]['v_last']=G.es[edgeVI]['v']
                            G.vs[vI]['inflowE']=inE
                            G.vs[vI]['outflowE']=outE
                    #it is now a noFlow Vertex
                    else:
                        if G.vs[vI]['degree']==1 and len(inE) == 1 and len(outE) == 0:
                            print('WARNING2 changed from noFlow to outflow')
                            print(vI)
                            G.vs[vI]['av'] = 0
                            G.vs[vI]['vv'] = 1
                            G.vs[vI]['vType'] = 2
                            edgeVI=G.adjacent(vI)[0]
                            if 'httBC' in G.es.attribute_names():
                                G.es[edgeVI]['httBC']=None
                                G.es[edgeVI]['posFirst_last']=None
                                G.es[edgeVI]['v_last']=None
                        elif G.vs[vI]['degree']==1 and len(inE) == 0 and len(outE) == 1:
                            print('WARNING2 changed from noFlow to inflow')
                            print(vI)
                            G.vs[vI]['av'] = 1
                            G.vs[vI]['vv'] = 0
                            G.vs[vI]['vType'] = 1
                            edgeVI=G.adjacent(vI)[0]
                            if 'httBC' in G.es.attribute_names():
                                G.es[edgeVI]['httBC']=G.es[edgeVI]['httBC_init']
                                if len(G.es[edgeVI]['rRBC']) > 0:
                                    if G.es['sign'][edgeVI] == 1:
                                        G.es[edgeVI]['posFirst_last']=G.es['rRBC'][edgeVI][0]
                                    else:
                                        G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]-G.es['rRBC'][edgeVI][-1]
                                else:
                                    G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]
                                G.es[edgeVI]['v_last']=G.es[edgeVI]['v']
                        else:
                            noFlowEdges=[]
                            for i in G.adjacent(vI):
                                #if G.es['flow'][i] > 5.0e-08:
                                #    print('FLOWERROR')
                                #    print(vI)
                                #    print(inE)
                                #    print(outE)
                                #    print(noFlowE)
                                #    print(i)
                                #    print('Flow and diameter')
                                #    print(G.es['flow'][i])
                                #    print(G.es['diameter'][i])
                                noFlowEdges.append(i)
                            G.vs[vI]['vType']=7
                            G.es[noFlowEdges]['noFlow']=[1]*len(noFlowEdges)
                            G.vs[vI]['inflowE']=[]
                            G.vs[vI]['outflowE']=[]
            print('Number of noFlow vertices')
            print(len(noFlowV))

            G['av']=G.vs(av_eq=1).indices
            G['vv']=G.vs(vv_eq=1).indices
 

    #--------------------------------------------------------------------------                                                   

    def strahler(self, nKind,startNKind,dir):
        """Computes the basic strahler order
        INPUT: nKind: the nkind of vertices which is to be investigated
               startNKind: the nkind of the neighboring starting points, to define order = 0
               dir: direction of strahler order analysis ('in' or 'out'). 'in' is used if upstream
               bifurcations have higher orders (usually used for arteriole trees)
        OUTPUT: edgeAttribute 'orderBasic'
        """

        Gdir=deepcopy(self)
        
        if 'orderBasic' not in self.es.attribute_names():
            self.es['orderBasic'] = [-1 for e in self.es]
        if 'orderBasic' not in self.vs.attribute_names():
            self.vs['orderBasic'] = [-1 for e in self.es]
        
        if dir == 'in':
            dir2='out'
        else:
            dir2='in'

        if nKind == 2:
            nKind2 = 0
        elif nKind == 3:
            nKind2 = 1
        
        capStarts=[]
        for i in self.vs(nkind_eq=nKind).indices + self.vs(nkind_eq=nKind2).indices:
            for j,k in zip(self.adjacent(i),self.neighbors(i)):
                if self.vs[k]['nkind'] == startNKind:
                    capStarts.append(i)

        capStarts=np.unique(capStarts)
        capStartsNew=[]
        for i in capStarts:
            for j,k in zip(self.adjacent(i),self.neighbors(i)):
                if self.vs[k]['nkind'] == nKind or self.vs[k]['nkind'] == nKind2: 
                    self.es[int(j)]['orderBasic'] = 0
                    self.vs[int(i)]['orderBasic'] = 0
                    capStartsNew.append(k)
        
        Gdir.to_directed_flow_based()
        for n in range(1000):
            changedOrder=[0,0]
            #boolInEdgesPresent=0
            boolInEdgesPresent=1
            while len(changedOrder) != 0:
                changedOrder=[]
                verts=self.vs(orderBasic_eq=n,nkind_eq=nKind).indices+self.vs(orderBasic_eq=n,nkind_eq=nKind2).indices
                verts2=[]
                for i in verts:
                    for k in Gdir.neighbors(i,dir):
                        if self.vs[k]['nkind'] == nKind or self.vs[k]['nkind'] == nKind2:
                            verts2.append(k)
                verts2=np.unique(verts2) 
                for i in verts2:
                    i=int(i)
                    orderBasics=[]
                    for j in Gdir.adjacent(i,dir2):
                        orderBasics.append(self.es[j]['orderBasic'])
                    #for j,k in zip(Gdir.adjacent(i,dir),Gdir.neighbors(i,dir)):
                    #    if self.vs[k]['nkind'] == nKind or self.vs[k]['nkind'] == nKind2:
                    #        boolInEdgesPresent=1
                    if orderBasics.count(n) >= 2:
                        if len(Gdir.neighbors(i,dir)) != 0:
                            for j,k in zip(Gdir.adjacent(i,dir),Gdir.neighbors(i,dir)):
                                if self.vs[k]['nkind'] == nKind or self.vs[k]['nkind'] == nKind2:
                                    if self.vs[i]['orderBasic'] != n+1:
                                        changedOrder.append(i)
                                    self.es[j]['orderBasic']=n+1
                                    self.vs[i]['orderBasic']=n+1
                        else:
                            if self.vs[i]['orderBasic'] != n+1:
                                changedOrder.append(i)
                            self.vs[i]['orderBasic']=n+1
                    elif orderBasics.count(n+1) >= 1 and orderBasics.count(n) >= 1:
                        if len(Gdir.neighbors(i,dir)) != 0:
                            for j,k in zip(Gdir.adjacent(i,dir),Gdir.neighbors(i,dir)):
                                if self.vs[k]['nkind'] == nKind or self.vs[k]['nkind'] == nKind2:
                                    if self.vs[i]['orderBasic'] != n+1:
                                        changedOrder.append(i)
                                    self.es[j]['orderBasic']=n+1
                                    self.vs[i]['orderBasic']=n+1
                        else:
                            if self.vs[i]['orderBasic'] != n+1:
                                changedOrder.append(i)
                            self.vs[i]['orderBasic']=n+1
                    else:
                        if len(Gdir.neighbors(i,dir)) != 0:
                            for j,k in zip(Gdir.adjacent(i,dir),Gdir.neighbors(i,dir)):
                                if self.vs[k]['nkind'] == nKind or self.vs[k]['nkind'] == nKind2:
                                    if self.vs[i]['orderBasic'] != n:
                                        changedOrder.append(i)
                                    self.es[j]['orderBasic']=n
                                    self.vs[i]['orderBasic']=n
                        else:
                            if self.vs[i]['orderBasic'] != n:
                                changedOrder.append(i)
                            self.vs[i]['orderBasic']=n
            if boolInEdgesPresent == 0:
                break

    #--------------------------------------------------------------------------                                                   

    def strahlerCapBed(self):
        """Computes the basic strahler order for the capillary bed
        INPUT: None
        OUTPUT: edgeAttribute 'orderBasic'
        """

        Gdir=deepcopy(self)
        
        if 'orderBasicCap' not in self.es.attribute_names():
            self.es['orderBasicCap'] = [-1 for e in self.es]
        if 'orderBasicCap' not in self.vs.attribute_names():
            self.vs['orderBasicCap'] = [-1 for e in self.es]
        
        dir='out'
        dir2='in'
        
        capStarts=[]
        for i in self.vs(nkind_eq=2).indices:
            for j,k in zip(self.neighbors(i),self.adjacent(i)):
                if self.vs[j]['nkind'] == 4:
                    capStarts.append(j)

        for j in capStarts:
            self.vs[j]['orderBasicCap'] = 0
            for j,k in zip(self.neighbors(i),self.adjacent(i)):
                if self.vs[j]['nkind'] == 4:
                    self.es[k]['orderBasicCap'] = 0
        
        #TODO abbruch kriterium fuer assigning orders
        Gdir.to_directed_flow_based()
        for n in range(100000):
            changedOrder=[0,0]
            #boolInEdgesPresent=0
            boolInEdgesPresent=1
            while len(changedOrder) != 0:
                changedOrder=[]
                verts=self.vs(orderBasicCap_eq=n).indices
                verts2=[]
                for i in verts:
                    for k in Gdir.neighbors(i,dir):
                        if self.vs[k]['nkind'] == 4:
                            verts2.append(k)
                verts2=np.unique(verts2) 
                for i in verts2:
                    i=int(i)
                    orderBasicsCap=[]
                    for j in Gdir.adjacent(i,'in'):
                        orderBasicsCap.append(self.es[j]['orderBasicCap'])
                    #for j,k in zip(Gdir.adjacent(i,'out'),Gdir.neighbors(i,'out')):
                    #    if self.vs[k]['nkind'] == 4:
                    #        boolInEdgesPresent=1
                    if orderBasicsCap.count(n) >= 2:
                        for j,k in zip(Gdir.adjacent(i,'out'),Gdir.neighbors(i,'out')):
                            if self.vs[k]['nkind'] == 4:
                                if self.vs[i]['orderBasicCap'] != n+1:
                                    changedOrder.append(i)
                                self.es[j]['orderBasicCap']=n+1
                                self.vs[i]['orderBasicCap']=n+1
                            else:
                                if self.vs[i]['orderBasicCap'] != n+1:
                                    changedOrder.append(i)
                                self.vs[i]['orderBasicCap']=n+1
                    elif orderBasicsCap.count(n+1) >= 1 and orderBasicsCap.count(n) >= 1:
                        for j,k in zip(Gdir.adjacent(i,'out'),Gdir.neighbors(i,'out')):
                            if self.vs[k]['nkind'] == 4:
                                if self.vs[i]['orderBasicCap'] != n+1:
                                    changedOrder.append(i)
                                self.es[j]['orderBasicCap']=n+1
                                self.vs[i]['orderBasicCap']=n+1
                            else:
                                if self.vs[i]['orderBasicCap'] != n+1:
                                    changedOrder.append(i)
                                self.vs[i]['orderBasicCap']=n+1
                    else:
                        for j,k in zip(Gdir.adjacent(i,'out'),Gdir.neighbors(i,'out')):
                            if self.vs[k]['nkind'] == 4:
                                if self.vs[i]['orderBasicCap'] != n:
                                    changedOrder.append(j)
                                self.es[j]['orderBasicCap']=n
                                self.vs[i]['orderBasicCap']=n
                            else:
                                if self.vs[i]['orderBasicCap'] != n:
                                    changedOrder.append(i)
                                self.vs[i]['orderBasicCap']=n
            if boolInEdgesPresent == 0:
                break

    #--------------------------------------------------------------------------
    
    def create_tortuous_structure(self,nkindKey=0,surfPlunKey=0):
        """Turns a graph which consists of individual data point into a graph with 
        the commonly used tortuous data structure, meaning the graph only consists of
        degree 3 and degree 4 vertices and the edge attributes 'points','diameters',
        'diameters2','lengths' and 'lengths2' are created.
        INPUT: graph itself.
               nkindKey: edge attribute which represents nkind (for kleinfeld NW 'labelAV'). works for up
                       to three different kinds per edge. if different nkinds per merged edge are found, the
                       one with the highest occurence is assigned. If similiar occurences are found the nkind
                       with the lower interger is assigned
               labelSurfPlunKey: edge attribute to differntiate between surface and plunging vessles specific
                                 for the kleinfeld NW, in Kleinfeld NW 'labelSurfPlun'
        OUTPUT: graph iteself is changed.
        """
        if nkindKey == 0:
            boolNkind=0
        else:
            boolNkind=1

        if surfPlunKey == 0:
            boolSurfPlun=0
        else:
            boolSurfPlun=1

        eps = finfo(float).eps*1000
        self.vs['degree']=self.degree()

        deg3=self.vs(degree_eq=3).indices
        deg4=self.vs(degree_eq=4).indices
        degs=deg3+deg4
        degs.sort()
        delVertices=[]
        self.es['points']=[None]*self.ecount()
        self.es['diameters']=[None]*self.ecount()
        self.es['lengths']=[None]*self.ecount()
        self.es['diameters2']=[None]*self.ecount()
        self.es['lengths2']=[None]*self.ecount()
        incidents=[]
        neighbors=[]
        for i in range(self.vcount()):
            incidents.append(self.incident(i))
            neighbors.append(self.neighbors(i))

        self.vs['incidents']=incidents
        self.vs['neighbors']=neighbors
        doneVertex=[]
        print('Vertices to analyze')
        print(len(degs))
        degs=self.vs(degs)
        count=0
        for i in degs:
            stdout.flush()
            count+= 1
            neighbors=i['neighbors']
            incidents=i['incidents']
            for k,j in zip(neighbors,incidents):
                incidentStart=j
                points=[]
                diams=[]
                lengths=[]
                diams2=[]
                lengths2=[]
                kinds=[]
                surfPlun=[]
                diams.append(np.mean(self.es[i['incidents']]['diameter']))
                points.append(i['r'])
                if boolNkind:
                    kinds.append(i[nkindKey])
                if k not in doneVertex:
                   vert=self.vs[k]
                   edge=self.es[j]
                   if vert['degree']==2:
                       boolDeg2=1
                       lengths.append(0.5*edge['length'])
                       diams.append(np.mean(self.es[vert['incidents']]['diameter']))
                       diams2.append(edge['diameter'])
                       lengths.append(0.5*np.sum(self.es[vert['incidents']]['length']))
                       lengths2.append(edge['length'])
                       if boolNkind:
                           kinds.append(vert[nkindKey])
                       if boolSurfPlun:
                           surfPlun.append(vert[surfPlunKey])
                       points.append(vert['r'])
                       lastVertex=i.index
                       delVertices.append(k)
                       noDeg2Edge=0
                   else:
                       boolDeg2=0
                       noDeg2Edge=1
                   m=k
                   while boolDeg2:
                       vert=self.vs[m]
                       if vert['degree'] == 2:
                           neighbors2 = vert['neighbors']
                           incidents2 = vert['incidents']
                           for k2,j2 in zip(neighbors2,incidents2):
                                if k2 != lastVertex:
                                    edge=self.es[j2]
                                    vert=self.vs[k2]
                                    diams.append(np.mean(self.es[vert['incidents']]['diameter']))
                                    diams2.append(edge['diameter'])
                                    lengths.append(0.5*np.sum(self.es[vert['incidents']]['length']))
                                    lengths2.append(edge['length'])
                                    points.append(vert['r'])
                                    if boolNkind:
                                        kinds.append(vert[nkindKey])
                                    if boolSurfPlun:
                                        surfPlun.append(vert[surfPlunKey])
                                    lastVertex=m
                                    delVertices.append(m)
                                    mOld=m
                                    m=k2
                                    break
                       else:
                           lengths=lengths[:-1]
                           lengths.append(0.5*edge['length'])
                           if len(diams) != len(diams2) + 1:
                               print('ERROR len diams')
                               print(i)
                           if np.abs(np.sum(lengths)-np.sum(lengths2)) > eps:
                               print('ERROR lengths')
                               print(i)
                               print(np.sum(lengths))
                               print(np.sum(lengths2))
                           if len(lengths) != len(lengths2) + 1:
                               print('ERROR len lengths')
                               print(i)
                           boolDeg2=0
                   if not noDeg2Edge:
                       self.add_edge(i,k2)
                       doneVertex.append(mOld)
                       stdout.flush()
                       if i.index > k2:
                           self.es[self.ecount()-1]['points']=np.array(points[::-1])
                           self.es[self.ecount()-1]['diameters']=np.array(diams[::-1])
                           self.es[self.ecount()-1]['diameters2']=np.array(diams2[::-1])
                           self.es[self.ecount()-1]['lengths']=np.array(lengths[::-1])
                           self.es[self.ecount()-1]['length']=np.sum(lengths[::-1])
                           self.es[self.ecount()-1]['lengths2']=np.array(lengths2[::-1])
                           self.es[self.ecount()-1]['diameter']=np.mean(diams[::-1])
                           self.es[self.ecount()-1]['endSeg']=self.es[incidentStart]['indexOrig']
                           self.es[self.ecount()-1]['startSeg']=self.es[j]['indexOrig']
                       else:
                           self.es[self.ecount()-1]['points']=np.array(points)
                           self.es[self.ecount()-1]['diameters']=np.array(diams)
                           self.es[self.ecount()-1]['diameters2']=np.array(diams2)
                           self.es[self.ecount()-1]['lengths']=np.array(lengths)
                           self.es[self.ecount()-1]['length']=np.sum(lengths)
                           self.es[self.ecount()-1]['lengths2']=np.array(lengths2)
                           self.es[self.ecount()-1]['diameter']=np.mean(diams)
                           self.es[self.ecount()-1]['startSeg']=self.es[incidentStart]['indexOrig']
                           self.es[self.ecount()-1]['endSeg']=self.es[j]['indexOrig']
                       #if boolNkind:
                       #    self.es[self.ecount()-1][nkindKey+'s']=np.array(kinds)
                       if boolNkind:
                           if len(np.unique(kinds)) != 1:
                               if len(np.unique(kinds)) == 3:
                                   if kinds.count(np.unique(kinds)[0]) > kinds.count(np.unique(kinds)[1]) and kinds.count(np.unique(kinds)[0]) > kinds.count(np.unique(kinds)[2]):
                                       self.es[self.ecount()-1][nkindKey]=np.unique(kinds)[0]
                                   elif kinds.count(np.unique(kinds)[1]) > kinds.count(np.unique(kinds)[0]) and kinds.count(np.unique(kinds)[1]) > kinds.count(np.unique(kinds)[2]):
                                       self.es[self.ecount()-1][nkindKey]=np.unique(kinds)[1]
                                   elif kinds.count(np.unique(kinds)[2]) > kinds.count(np.unique(kinds)[0]) and kinds.count(np.unique(kinds)[2]) > kinds.count(np.unique(kinds)[1]):
                                       self.es[self.ecount()-1][nkindKey]=np.unique(kinds)[2]
                                   else:
                                       if kinds.count(np.unique(kinds)[0]) == kinds.count(np.unique(kinds)[1]):
                                           self.es[self.ecount()-1][nkindKey]=np.unique(kinds)[0]
                                       elif kinds.count(np.unique(kinds)[0]) == kinds.count(np.unique(kinds)[2]):
                                           self.es[self.ecount()-1][nkindKey]=np.unique(kinds)[0]
                                       else:
                                           self.es[self.ecount()-1][nkindKey]=np.unique(kinds)[1]
                               else:
                                   if kinds.count(np.unique(kinds)[0]) > kinds.count(np.unique(kinds)[1]):
                                       self.es[self.ecount()-1][nkindKey]=np.unique(kinds)[0]
                                   elif kinds.count(np.unique(kinds)[1]) > kinds.count(np.unique(kinds)[0]):
                                       self.es[self.ecount()-1][nkindKey]=np.unique(kinds)[1]
                                   else:
                                       if kinds.count(np.unique(kinds)[0]) == kinds.count(np.unique(kinds)[1]):
                                           self.es[self.ecount()-1][nkindKey]=np.unique(kinds)[0]
                           else:
                               self.es[self.ecount()-1][nkindKey]=np.unique(kinds)[0]
                       if boolSurfPlun:
                           if len(np.unique(surfPlun)) != 1:
                               if len(np.unique(surfPlun)) == 3:
                                   if surfPlun.count(np.unique(surfPlun)[0]) > surfPlun.count(np.unique(surfPlun)[1]) and surfPlun.count(np.unique(surfPlun)[0]) > surfPlun.count(np.unique(surfPlun)[2]):
                                       self.es[self.ecount()-1][surfPlunKey]=np.unique(surfPlun)[0]
                                   elif surfPlun.count(np.unique(surfPlun)[1]) > surfPlun.count(np.unique(surfPlun)[0]) and surfPlun.count(np.unique(surfPlun)[1]) > surfPlun.count(np.unique(surfPlun)[2]):
                                       self.es[self.ecount()-1][surfPlunKey]=np.unique(surfPlun)[1]
                                   elif surfPlun.count(np.unique(surfPlun)[2]) > surfPlun.count(np.unique(surfPlun)[0]) and surfPlun.count(np.unique(surfPlun)[2]) > surfPlun.count(np.unique(surfPlun)[1]):
                                       self.es[self.ecount()-1][surfPlunKey]=np.unique(surfPlun)[2]
                                   else:
                                       if surfPlun.count(np.unique(surfPlun)[0]) == surfPlun.count(np.unique(surfPlun)[1]):
                                           self.es[self.ecount()-1][surfPlunKey]=np.unique(surfPlun)[1]
                                       elif surfPlun.count(np.unique(surfPlun)[0]) == surfPlun.count(np.unique(surfPlun)[2]):
                                           self.es[self.ecount()-1][surfPlunKey]=np.unique(surfPlun)[2]
                                       else:
                                           self.es[self.ecount()-1][surfPlunKey]=np.unique(surfPlun)[1]
                               else:
                                   if surfPlun.count(np.unique(surfPlun)[0]) > surfPlun.count(np.unique(surfPlun)[1]):
                                       self.es[self.ecount()-1][surfPlunKey]=np.unique(surfPlun)[0]
                                   elif surfPlun.count(np.unique(surfPlun)[1]) > surfPlun.count(np.unique(surfPlun)[0]):
                                       self.es[self.ecount()-1][surfPlunKey]=np.unique(surfPlun)[1]
                                   else:
                                       if surfPlun.count(np.unique(surfPlun)[0]) == surfPlun.count(np.unique(surfPlun)[1]):
                                           self.es[self.ecount()-1][surfPlunKey]=np.unique(surfPlun)[1]
                           else:
                               self.es[self.ecount()-1][surfPlunKey]=np.unique(surfPlun)[0]
                #else:
                #    print('Edge Already done')

        delVertices=np.unique(delVertices)
        self.delete_vertices(delVertices)
        del(self.vs['neighbors'])
        del(self.vs['incidents'])
        self.vs['degree']=self.degree()
        noConcatenatedEdges=self.es(startSeg_eq=None).indices
        for e in self.es[noConcatenatedEdges]:
            e['startSeg']=e['indexOrig']
            e['endSeg']=e['indexOrig']

        del(self.es['indexOrig'])

        #lookg for no points edges
        noPoints=self.es(points_eq=None).indices
        self.add_points(1.,noPoints)
        
        #Chose between effective and other diameter
        countAssignMedian = 0
        for i in range(self.ecount()):
            e=self.es[i]
            resistances=[]
            for j in range(len(e['lengths2'])):
                resistances.append(e['lengths2'][j]/(e['diameters2'][j]**4))
            resistanceTot=np.sum(resistances)
            e['effDiam']=(e['length']/resistanceTot)**(0.25)
            if np.abs(np.median(e['diameters2'])-e['effDiam'])/(np.mean([np.median(e['diameters2']),e['effDiam']])) > 0.1:
                e['effDiam']=np.median(e['diameters2'])
                countAssignMedian += 1
            if e['length'] == 0:
                print('Length problem')
                print(i)
                print(np.median(e['diameters2']))
                print(e['length'])
                print(e['lengths2'])
                print(resistanceTot)
                print(resistances)

        self['countAssignMedian']=countAssignMedian
        self.es['diameter']=deepcopy(self.es['effDiam'])
        del(self.es['effDiam'])
#--------------------------------------------------------------------
    def save_graph_as_dict(self, edgeAttr=['flow','length','diameter','nRBC','htt','httBC'], vertexAttr=['pressure','pBC']):
        """ Writes two pkl files (dictonaries) with all the attributes of the graph. file 1: vertices.pkl; file 2: edges.pkl
        INPUT: edgeAttr: list of edgeAttr to be outputed. if empty only the tuple list is written
                vertexAttr: list of vertexAttr to be outputed. if empty only coordinates are written
        OUTPUT: two .pkl files
        """

        verticesDict={}
        verticesDict['coords']=self.vs['r']
        for attr in vertexAttr:
            verticesDict[attr]=self.vs[attr]

        edgesDict={}
        tuples=[e.tuple for e in self.es]
        edgesDict['tuple']=tuples
        for attr in edgeAttr:
            edgesDict[attr]=self.es[attr]

        vgm.write_pkl(verticesDict,'verticesDict.pkl')
        vgm.write_pkl(edgesDict,'edgesDict.pkl')

#--------------------------------------------------------------------
    def remove_all_degree_1_vertices(self, keep_nkind=[0,1]):
        """ Generates a trimmed graph where all degree=1 vertices have been removed 
        INPUT: keep_nkind: degree 1 vertices of the listed kinds are not removed. default = [0,1] 
        OUTPUT: 
        """

        self.vs['degree'] = self.degree()

        keep_degree_1 = []
        for nkind in keep_nkind:
            keep_degree_1 = keep_degree_1 + self.vs(degree_eq=1,nkind_eq=nkind).indices 

        while len(self.vs(degree_eq=1)) > len(keep_degree_1):
            deg1 = []
            for v in self.vs(degree_eq=1).indices:
                if v not in keep_degree_1:
                    deg1.append(v)
            self.delete_vertices(deg1)

            loop_vertices = []
            self.vs['degree'] = self.degree()
            for v in self.vs(degree_ge=2).indices:
                if len(np.unique(self.neighbors(v))) == 1:
                    loop_vertices.append(v)
            self.delete_vertices(loop_vertices)
            self.vs['degree'] = self.degree()

            keep_degree_1 = []
            for nkind in keep_nkind:
                keep_degree_1 = keep_degree_1 + self.vs(degree_eq=1,nkind_eq=nkind).indices 

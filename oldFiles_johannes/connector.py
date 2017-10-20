from __future__ import division

import numpy as np
from scipy.spatial import kdtree

__all__ = ['Connector']


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class Connector(object):
    """Connects specific vertices of two graphs with each other on the basis 
    of a set of rules, without actually creating edges between these 
    vertices. These 'soft link' connections may have properties.
    (This concept could be extended to include connections between edges, as
    well as between edges and vertices.) 
    """    
    
    def __init__(self, G1, G2, strategy=None, **kwargs):
        """Initializes a Connector instance.
        INPUT: G1: First VascularGraph.
               G2: Second VascularGraph.
               strategy: The strategy to be applied when connecting G1 and G2.
                         Possible values include:
                         None: (The default) creates no connections.
                         vasculatureTissue: Connects the vertices of the 
                                            vasculature graph G1 with the
                                            respective nearest neighbors in the
                                            tissue graph G2.
               **kwargs: List of keyword arguments to be passed on to the 
                         connecting function.
        OUTPUT: None       
        """
        self._G1 = G1
        self._G2 = G2
        self._strategy = strategy
        if strategy is None:
            self.connect = self.connect_nothing
        elif strategy == 'vasculatureTissue':
            self.connect = self.connect_vasculature_with_tissue
        else:
            raise KeyError('Unknown connection strategy!')
        self.connect(**kwargs)


    # Helper class 'Connection' -----------------------------------------------
    class Connection(object):
        """Implements a single 'soft-link' connection between two graphs and
        serves as a means to store data associated with that connection.
        """
        
        def __init__(self, i1, i2):
            """Initializes the Connection.
            INPUT: i1: Index of the vertex (or edge) in G1
                   i2: Index of the vertex (or edge) in G2
            OUTPUT: None
            """
            self.i1 = i1
            self.i2 = i2        
    # -------------------------------------------------------------------------


    def connect_nothing(self, **kwargs):
        """Creates empty connection dictionaries / lists
        INPUT: None
        OUTPUT: None
        """
        self.vertexG1ToVerticesG2 = {}
        self.vertexG2ToVerticesG1 = {}
        self.connections = []                            
        

    def connect_vasculature_with_tissue(self, **kwargs):
        """Connects VascularGraphs G1 (vasculature) and G2 (tissue) with 
        traited 'soft links'. Each vascualar vertex connects to exactly one 
        tissue vertex (its nearest neighbor). A tissue vertex, however, may 
        connect to multiple vascular vertices. Each connection also stores 
        the sum of the products of surface area and exchange coefficient. This
        vertex property is required for the exchange of a given substance 
        between vasculature and tissue (see the Exchange class).
        INPUT: **kwargs
               substance: The name of the substance that is to be exchanged.              
        OUTPUT: None               
        """

        self.vertexG1ToVerticesG2 = {}
        self.vertexG2ToVerticesG1 = {}
        self.connections = []
        
        G1 = self._G1
        G2 = self._G2
        substance = kwargs['substance']
                
        Kdt = kdtree.KDTree(G2.vs['r'], leafsize=10)
        
        for vIndex in xrange(G1.vcount()):
            edgeIndices = G1.adjacent(vIndex, 'all')            
            exchangeFactor = 0.0
            for edge in G1.es(edgeIndices):
                exchangeFactor = exchangeFactor + edge['length'] * np.pi * \
                                                  edge['diameter'] / 2.0 * \
                                                  edge['exchangeCoefficient'][substance]
            if 'kind' in G1.vs.attribute_names():
                if G1.vs[vIndex]['kind'] == 'u':
                    exchangeFactor = exchangeFactor + \
                                     G1.vs[vIndex]['uSurfaceArea'] * \
                                     G1.vs[vIndex]['uExchangeCoefficient'][substance]                  

            tissueNeighbor = int(Kdt.query(G1.vs[vIndex]['r'])[1])            
            self.vertexG1ToVerticesG2[vIndex] = tissueNeighbor
            if self.vertexG2ToVerticesG1.has_key(tissueNeighbor):
                self.vertexG2ToVerticesG1[tissueNeighbor].append(vIndex)
            else:
                self.vertexG2ToVerticesG1[tissueNeighbor] = [vIndex]
            self.connections.append(self.Connection(vIndex, tissueNeighbor))
            self.connections[-1].exchangeFactor = exchangeFactor

                            
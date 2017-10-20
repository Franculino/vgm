from __future__ import division

import numpy as np
import pylab

__all__ = ['divide_into_slabs', 'unify_slabs']


def divide_into_slabs(G, slabThickness, overlap):
    """Divides a VascularGraph into subgraphs that are slabs along the z-axis.
    INPUT: G: VascularGraph to be cut into slabs.
           slabThickness: The thickness (in z-direction) of the resulting 
                          slabs.
           overlap: The overlap of the slabs.
    OUTPUT: A list of slab subgraphs.                      
    """
    minima, maxima, lengths = G.dimension_extrema()
    zLength = lengths[2]
    nSlabs = int(np.ceil(zLength / (slabThickness-overlap)))
    G.vs['z'] = [r[2] for r in G.vs['r']]
    zMin = minima[2]
    zMax = zMin + slabThickness
    slabSubgraphs = []
    for slab in xrange(nSlabs):
        slabVertices = G.vs(z_ge=zMin, z_lt=zMax)
        SG = G.subgraph(np.unique(pylab.flatten([[G.neighbors(v.index) 
                        for v in slabVertices], 
                        slabVertices.indices])).tolist())
        del SG.vs['z']
        slabSubgraphs.append(SG)
        zMin = zMax - overlap
        zMax = zMin + slabThickness
    del G.vs['z']
    return slabSubgraphs


def unify_slabs(slabSubgraphs):
    """Unite slabs to form a single VascularGraph. (This function is the 
    inverse of 'divide_into_slabs'.)
    INPUT: slabSubgraphs: List of slab subgraphs.
    OUTPUT: VascularGraph that is the union of the slab subgraphs.
    """
    G = slabSubgraphs[0]    
    for subgraph in slabSubgraphs[1:]:
        G.union_attribute_preserving(subgraph)
    return G

    
    


        
    
    
    

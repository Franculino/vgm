from __future__ import division

import pylab as pl
from pylab import flatten
import scipy as sp

import linearSystem
import vgm

__all__ = ['fuzzy_block_subgraph', 'fuzzy_block_conductance']
log = vgm.LogDispatcher.create_logger(__name__)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def fuzzy_block_subgraph(G, sideLengths, origin, **kwargs):
    """Extracts a cuboid subgraph from the input graph. This also includes the
    neighboring nodes that connect to nodes inside the cuboid, hence the name
    fuzzy-block.
    INPUT: G: Vascular graph in iGraph format.
           sideLengths: Dimensions of the block (cuboid) as a tuple/list.
           origin: The minimum x,y,z of the cuboid as tuple/list.
           **kwargs:
               dRange: Range of diameters to be included (edges with diameters
                       outside of this range are neglected). Provide as a 
                       tuple/list, i.e. [minDiameter, maxDiameter].
    OUTPUT: sg: Fuzzy-block subgraph. Additional vertex properties are added,
                namely offsets in x, y, and z direction compared to the cuboid.
                I.e. xoffset can take the values -1, 0, 1 corresponding to 
                positions in x-direction smaller than, equal to, or larger than
                the cuboid x-dimensions. Analogous for yoffset, and zoffset.
    """
        
    if len(sideLengths) == 1:
        sideLengths = [sideLengths[0] for i in xrange(3)]
    xMin = origin[0]; xMax = xMin + sideLengths[0]
    yMin = origin[1]; yMax = yMin + sideLengths[1]
    zMin = origin[2]; zMax = zMin + sideLengths[2]            
    block = G.vs(lambda x: xMin <= x['r'][0] <= xMax, 
                 lambda y: yMin <= y['r'][1] <= yMax, 
                 lambda z: zMin <= z['r'][2] <= zMax)             
    fbi = sp.unique(flatten([G.neighbors(x) for x in block.indices])).tolist()
    log.info('Vertex indices of the subgraph: ')
    log.info(fbi)
    sg = G.subgraph(fbi)

    if kwargs.has_key('dRange'):
        sg.delete_edges(sg.es(diameter_lt=kwargs['dRange'][0]).indices)
        sg.delete_edges(sg.es(diameter_gt=kwargs['dRange'][1]).indices)
    
    log.info("Subgraph has %i components" % len(sg.components()))

    sg.vs['xoffset'] = sp.zeros(sg.vcount())
    ltx = sg.vs(lambda x: x['r'][0] < xMin).indices 
    sg.vs(ltx)['xoffset'] = sp.ones(len(ltx)) * -1
    gtx = sg.vs(lambda x: x['r'][0] > xMax).indices
    sg.vs(gtx)['xoffset'] =  sp.ones(len(gtx))

    sg.vs['yoffset'] = sp.zeros(sg.vcount())
    lty = sg.vs(lambda y: y['r'][1] < yMin).indices 
    sg.vs(lty)['yoffset'] = sp.ones(len(lty)) * -1
    gty = sg.vs(lambda y: y['r'][1] > yMax).indices
    sg.vs(gty)['yoffset'] =  sp.ones(len(gty))
    
    sg.vs['zoffset'] = sp.zeros(sg.vcount())
    ltz = sg.vs(lambda z: z['r'][2] < zMin).indices 
    sg.vs(ltz)['zoffset'] = sp.ones(len(ltz)) * -1
    gtz = sg.vs(lambda z: z['r'][2] > zMax).indices
    sg.vs(gtz)['zoffset'] =  sp.ones(len(gtz))

    return sg


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


def fuzzy_block_conductance(sg,off1,off2):
    """Computes the effective conductance of the fuzzy-block in the direction
    off1-off2.
    INPUT: sg: Fuzzy-block subgraph in iGraph format (see fuzzy_block_subgraph).
           off1: First offset as a tuple / list , e.g. (-1,0,0) which would be 
                 the neighboring block in -x direction.
           off2: Second offset.       
    OUTPUT: Effective conductance in off1-off2 direction
    """
    
    if 'pBC' in sg.vs.attribute_names(): 
        del sg.vs['pBC']

    vIn = sg.vs(xoffset_eq=off1[0], 
                yoffset_eq=off1[1], 
                zoffset_eq=off1[2]).indices
    vOut = sg.vs(xoffset_eq=off2[0], 
                 yoffset_eq=off2[1], 
                 zoffset_eq=off2[2]).indices
    
    if min(len(vIn),len(vOut)) == 0:
        log.error("Cannot compute effective conductance")
        return
    if 'conductance' not in sg.es.attribute_names():
        log.warning("Adding uniform conductance to all edges of the subgraph.")
        sg.es['conductance'] = sp.ones(sg.ecount())

    sg.vs(vIn)['pBC']  = sp.ones(len(vIn))  * 2.0
    sg.vs(vOut)['pBC'] = sp.ones(len(vOut)) * 1.0

    LS = linearSystem.LinearSystem(sg)
    LS.solve('direct')
    flow = sum(pl.flatten([sg.es(sg.adjacent(x,'all'))['flow'] for x in vIn]))
    
    return flow # G = F / dp, here dp == 1.0



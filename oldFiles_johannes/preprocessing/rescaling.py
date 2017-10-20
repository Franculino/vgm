from __future__ import division, print_function
import numpy as np
import scipy as sp
import vgm
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

__all__ = ['rescale_cdd', 'correct_for_shrinkage', 'scale_to_volume_fraction',
           'scale_to_mean_capillary_diameter']
log = vgm.LogDispatcher.create_logger(__name__)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def rescale_cdd(G, copyGraph=True, zRange=None, 
                wallThickness=0.5, figname='fig.png', **kwargs):
    """Rescales a VascularGraph by matching its capillary diameter 
    distribution to that of the Cerebral Cortex paper of Weber et al. (2008).
    In a second step the dimensions of the entire sample are scaled by the
    mean scaling factor of the first step, or by a given factor if provided via
    the keyword arguments.
    INPUT: G: VascularGraph
           copyGraph: Modify original or make a copy? (Boolean.)
           zRange: The cortical depth for which the volume fraction is to be
                   computed. Default is [max(100, minZ), min(900, maxZ)].
           wallThickness: The wall thickness of the blood vessels. A uniform
                          thickness is assumed for all vessels.
           figname: The name of the figure in which the results are plotted.
                    If this is set to 'None', no figure will be created.
           **kwargs:
               dsf: The scaling factor by which the dimensions of the sample 
                    are to be multiplied with.
    OUTPUT: VascularGraph and figure (depending on input).
    """

    if zRange is None:
        # Compute min and max z of the sample (to be used when determining
        # the vascular volume fraction):
        z = [r[2] for r in G.vs['r']]
        minZ = min(z)
        maxZ = max(z)
        zRange = [max(100, minZ), min(900, maxZ)]

    # Vessel diameters (central bin values):
    diameter = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5,
                12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 
                22.5, 23.5, 24.5]
    # Frequencies taken from Cerebral Cortex paper (Macaque):
    frequency_original = [0.0, 0.00986079, 0.0516241, 0.0435035, 0.174594, 
                          0.180394, 0.12413, 0.143271, 0.134571, 0.0232019, 
                          0.050464, 0.0203016, 0.0168213, 0.00406032, 
                          0.0075406, 0.00348028, 0.00290023, 0.00232019,
                          0.00290023, 0.00116009, 0.00116009, 0.00116009, 
                          0.00116009, 0.000580046, 0.000580046]
    # Frequencies determined by fitting a Gaussian to the above frequencies
    # [:8], i.e. the capillaries. (The first frequency has been set to 0.0
    # manually, the original fit value is 9.21735097e-03.)
    frequency = [0.0,              2.38039294e-02,   5.10707093e-02,
                 9.10281750e-02,   1.34790874e-01,   1.65815778e-01,
                 1.69461843e-01,   1.43879430e-01,   1.01486014e-01,
                 5.94695149e-02,   2.89509823e-02,   1.17088144e-02,
                 3.93408053e-03,   1.09813132e-03,   2.54651373e-04,
                 4.90589754e-05,   7.85184348e-06,   1.04401203e-06,
                 1.15324102e-07,   1.05831601e-08,   8.06847325e-10,
                 5.11031967e-11,   2.68896705e-12,   1.17544849e-13,
                 4.26876684e-15]
    
    # Gaussian function and parameters resulting from above fit:
    gaussian0, p0_g = vgm.gaussian_normalized()
    mu = 6.1173144
    sigma = 2.3224276
    p_g = (mu, sigma)

    def gaussian(p, x):
        if x <= 2.0: return 0
        else: return gaussian0(p, x)

    maxD = 10.
    steps = 500
    dbins = np.linspace(0, maxD, steps)
    dlist = [(dbins[i+1] + dbins[i]) / 2. for i in xrange(len(dbins[:-1]))]
    frequencies = [gaussian(p_g, d) for d in dlist]
    sumf= sum(frequencies)
    normfreq = [f / sumf for f in frequencies]
    cumfreq = np.cumsum(frequencies) / sumf 
    N = len(G.es(diameter_le=8))
    # Diameter- and frequency-lists for plotting:
    dbinsp = np.linspace(0, 25, steps)
    dlistp = [(dbinsp[i+1] + dbinsp[i]) / 2. for i in xrange(len(dbinsp[:-1]))]
    frequenciesp = [gaussian(p_g, d) for d in dlistp]

    # Plot original diameter distribution:
    if figname is not None:    
        fig = plt.figure()
        plt.subplot(131)
        h = vgm.hist_pmf(G.es['diameter'],range(0,31,1),False)
        #plt.plot(diameter,frequency,color='red')
        plt.plot(dlistp,frequenciesp, color='red')
        plt.xlabel('Diameter [$\mu m$]')
        plt.ylabel('Frequency')    
        plt.text(0.63, 0.95,'MCD: %.1f $\mu m$' % \
                 (np.mean(G.es(diameter_le=8)['diameter'])),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)
        plt.text(0.63, 0.91,'VVF: %.1f %s' % \
                 (G.vascular_volume_fraction(zRange=zRange,
                                             wallThickness=wallThickness) * 100, '%'),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)

    # Create a copy of the VascularGraph, if desired:
    if copyGraph:
        G = deepcopy(G)
    
    # Make diameters pseudo-unique by (tiny) random dilation (maximum diameter
    # change possible is 1/1000 of the minimum diameter):
    N = G.ecount()
    eps = max(np.amin(G.es['diameter']) / 1000., 1e-20) # account for zero diameters
    randFactor = np.random.ranf(N) * eps + 1. 
    G.es['diameter'] = G.es['diameter'] * randFactor
    G.es['diameters'] = [dr[0] * dr[1] for dr in zip(G.es['diameters'], randFactor)]
    if 'crossSection' in G.es.attribute_names():
        G.es['crossSection'] = G.es['crossSection'] * randFactor**2.0
    if 'volume' in G.es.attribute_names():
        G.es['volume'] = G.es['volume'] * randFactor**2.0

    G.es['diameter_orig'] = G.es['diameter']

    for i in xrange(len(dlist)):
        d = dlist[i]
        dd = dlist[i+1]
        nf = normfreq[i]
        cf = cumfreq[i+1]
        
        Ndesired = nf * N
        eIndices = G.es(diameter_ge=d, diameter_lt=dd).indices
        Ncurrent = len(eIndices)

        if Ncurrent > Ndesired:
            Nmove = int(round(Ncurrent - Ndesired))
            diameters = np.array(G.es(eIndices)['diameter'])
            scalingFactor = np.sort(dd - diameters)[min(Nmove-1,len(diameters)-1)]        
            G.es(diameter_ge=d)['diameter'] = np.array(G.es(diameter_ge=d)['diameter']) + scalingFactor
        elif Ncurrent < Ndesired and d > 7.0:
            log.info('diameter break-off: %.5f\n' % d)
            break

    # Account for the fact that the Cerebral Cortex paper uses outer diameters,
    # while we require inner diameters:
    correctedDiameter = np.array(G.es['diameter']) - wallThickness * 2.0
    G.es['diameters'] = [x[0] * x[1] 
                         for x in zip(G.es['diameters'],
                                      correctedDiameter / np.array(G.es['diameter_orig']))]
    G.es['diameter'] = correctedDiameter
    G.es['diameter_change'] = np.array(G.es['diameter']) - np.array(G.es['diameter_orig'])

    if figname is not None:
        # plot absolute difference:
        plt.subplot(132)
        plt.scatter(G.es['diameter_orig'], G.es['diameter_change'])
        plt.xlabel('Diameter [$\mu m$]')
        plt.ylabel('Diameter addition [$\mu m$]')
        #plt.text(0.6, 0.06,'VVF: %.1f %s' % \
        #         (G.vascular_volume_fraction(zRange=zRange,
        #          wallThickness=wallThickness) * 100, '%'),
        #         horizontalalignment='center',
        #         verticalalignment='center',
        #         transform = plt.gca().transAxes)

    # length rescaling:
    if kwargs.has_key('dsf'): 
        sf = kwargs['dsf']
    else:    
        sf = np.mean(np.array(correctedDiameter)/np.array(G.es['diameter_orig']))
    G.es['length'] = np.array(G.es['length']) * sf
    G.vs['r'] = np.array(G.vs['r']) * sf
    try:
        G.es['volume'] = np.array(G.es['volume']) * sf
    except:
        pass
    try:
        for e in G.es:
            e['points'] = e['points'] * sf
    except:
        pass
    
    if figname is not None:
        # plot absolute difference:
        plt.text(0.6, 0.1,'MSF: %.1f' % sf,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)

        # plot relative change:
        #plt.scatter(G.es['diameter_orig'], np.array(G.es['diameter_change'])/np.array(G.es['diameter_orig']))
        plt.subplots_adjust(wspace=0.5)
        plt.subplot(133)
        h = vgm.hist_pmf(G.es['diameter'],range(0,31,1),False)
        #plt.plot(np.array(diameter)-wallThickness*2.0,frequency,color='red')
        plt.plot(np.array(dlistp)-wallThickness*2.0,frequenciesp,color='red')
        plt.xlabel('Diameter [$\mu m$]')
        plt.ylabel('Frequency')
        plt.text(0.63, 0.95,'MCD: %.1f $\mu m$' % \
                 (np.mean(G.es(diameter_le=8)['diameter'])),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)
        plt.text(0.63, 0.91,'VVF: %.1f %s' % \
                 (G.vascular_volume_fraction(zRange=zRange,
                                             wallThickness=wallThickness) * 100, '%'),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)
        plt.savefig(figname)
    
    for property in ['diameter_orig', 'diameter_change', 'cost']:
        try:
            del G.es[property]
        except:
            pass
    return G
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def correct_for_shrinkage(G,factor):
    """Corrects the geometrical properties of the vascular graph for sample
    shrinkage. Currently this modifies the following graph properties:
    vertices: r
    edges: diameters, diameter, depth, volume, length, points
    INPUT: G: Vascular graph in iGraph format.
           factor: Factor by which the geometrical properties are to be scaled
                   (i.e. enlarged).
    OUTPUT: None, G is modified in-place.
    """
    
    G.vs['r'] = map(lambda x: x * factor, G.vs['r']) 
    edgePropertyAndExponent = [('diameters', 1.), ('diameter', 1.),
                               ('depth', 1.), ('volume', 3.),
                               ('length', 1.), ('points', 1.)]
    for epe in edgePropertyAndExponent:
        if epe[0] in G.es.attribute_names(): 
            G.es[epe[0]] = map(lambda x: x * factor**epe[1], G.es[epe[0]])
                
                  
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def scale_to_volume_fraction(G,fraction=0.0225,**kwargs):
    """Corrects the geometrical properties of the vascular graph to achieve a 
    desired volume fraction by multiplication of vascular parameters with a
    given factor. Currently this modifies the following graph properties:
    vertices: r
    edges: diameters, diameter, depth, volume, length, points
    INPUT: G: Vascular graph in iGraph format.
           fraction: Volume-fraction which the vessels of the graph should
                     have after the scaling. The default is 0.0225.
           **kwargs:
             shape: The shape of the vascular graph. This may be either 
                    'cuboid' or 'cylinder'. If not provided, the shape is 
                    determined from the data.
             zRange: Range of zValues in which to consider the edges of the 
                     graph for volume fraction computation (as list, e.g.
                     [0,1000]). Without this keyword argument, all edges are 
                     considered.
                     Note that the scaling factor resulting from this 
                     computation is naturally not restricted to the vessels in 
                     zRange, it is applied to all vessels.
    OUTPUT: None, G is modified in-place.
    WARNING: This method yields mean capillary diameters that are 
             unphysiologically small. Use of 'rescale_cdd' is preferable. 
    """
    
    vv = vascular_volume(G,**kwargs)
    tv = total_volume(G,**kwargs)
    factor = (tv * fraction / vv)**(1/3.)
    correct_for_shrinkage(G,factor)
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def scale_to_mean_capillary_diameter(G,threshold=None,meanD=None,
                                     method='multiplication'):
    """Corrects the geometrical properties of the vascular graph to achieve a 
    desired mean capillary diameter. Diameters of the vessels are modified by 
    adding a common value, irrespective of the vessel type (capillary,
    arteriole, venole).

    INPUT: G: Vascular graph in iGraph format.
           threshold: Threshold below which vessels are considered as
                      capillaries (optional, if not provided threshold=7.0 
                      micron).
           meanD: Desired mean diameter of the capillaries (optional, if not
                  provided meanD=5.0 micron)
           method: This can be either 'multiplication' (the default), or
                   'addition'. Signifying whether a common factor is added or 
                   multiplied to all diameter values.
    OUTPUT: None, G is modified in-place.
    WARNING: This method yields incorrect diameter distributions. Use of 
             'rescale_cdd' is preferable. 
    """
    
    # Set default values, if not provided as function input:
    if threshold is None:
        threshold = 7.0 * units.scaling_factor_du('um', G['defaultUnits'])        
    if meanD is None:
        meanD = 5.0 * units.scaling_factor_du('um', G['defaultUnits'])        

    if method == 'multiplication':
        def compute_meanD_multiplication(factor):
            """Computes the mean diameter of the capillaries if the original
            diameter values are multiplied with 'factor'.
            INPUT: factor: The factor to be multiplied.
            OUTPUT: The new mean diameter of the capillaries.
            """
            d = sp.array(G.es['diameter']) * factor
            return sp.mean(d[d<=threshold])
            
        factorRange = sp.linspace(0,meanD/min(G.es['diameter']),1000)
        md = sp.array([compute_meanD_multiplication(f) for f in factorRange])
        factor = factorRange[sp.nonzero(abs(md-meanD) == 
                             min(abs(md-meanD)))[0][0]]
        
        G.es['diameter'] = sp.array(G.es['diameter']) * factor
        G.es['diameters'] = [dList * factor for dList in G.es['diameters']]
    elif method == 'addition':        
        def compute_meanD_addition(summand):
            """Computes the mean diameter of the capillaries if 'summand' is 
            added to the original diameter values
            INPUT: summand: The summand to be added.
            OUTPUT: The new mean diameter of the capillaries.
            """
            d = sp.array(G.es['diameter']) + summand
            return sp.mean(d[d<=threshold]) 
    
        summandRange = sp.linspace(0,threshold,1000)
        md = sp.array([compute_meanD_addition(s) for s in summandRange])
        summand = summandRange[sp.nonzero(abs(md-meanD) == 
                               min(abs(md-meanD)))[0][0]]
        
        G.es['diameter'] = sp.array(G.es['diameter']) + summand
        # consistent if 'diameter' is the mean of 'diameters':
        G.es['diameters'] = [dList * (sp.mean(dList)+summand) / 
                             sp.mean(dList) for dList in G.es['diameters']]
        # inconsistent:
        # G.es['diameters'] = [dList + summand for dList in G.es['diameters']]
    else:
        raise NotImplementedError
        
    # correct value only if it has been defined previously:
    if 'volume' in G.es.attribute_names():
        volume   = []
        for edge in G.es: 
            points = edge['points']
            diameters = edge['diameters']
            tmpVolume = 0
            for i in xrange(len(points)-1):
                l = norm(points[i+1] - points[i])
                A = sp.pi * sp.mean([diameters[i+1],diameters[i]])**2.0 / 4.0
                tmpVolume += l * A
            volume.append(tmpVolume)
        G.es['volume']   = volume    




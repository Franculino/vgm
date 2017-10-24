"""This module implements red blood cell transport in vascular networks 
discretely, i.e. resolving all RBCs. As would be expected, the computational 
expense increases with network size and timesteps can become very small. An
srXTM sample of ~20000 nodes will take about 1/2h to evolve 1ms at Ht0==0.5.
A performance analysis revealed the worst bottlenecks, which are:
_plot_rbc()
_update_blocked_edges_and_timestep()
Smaller CPU-time eaters are:
_update_flow_and_velocity()
_update_flow_sign()
_propagate_rbc()
"""
from __future__ import division

import numpy as np
from sys import stdout

from pyamg import smoothed_aggregation_solver, rootnode_solver, util
import pyamg
import scipy as sp
from scipy import finfo, ones, zeros
from scipy.sparse import lil_matrix, linalg
from scipy.integrate import quad
from scipy.optimize import root
from physiology import Physiology
from scipy.sparse.linalg import gmres
import units
import g_output
import vascularGraph
import pdb
import run_faster
import time as ttime
import vgm

__all__ = ['LinearSystemHtdTotFixedDT_NEW']
log = vgm.LogDispatcher.create_logger(__name__)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class LinearSystemHtdTotFixedDT_NEW(object):
    """The discrete model is extended such, that a fixed timestep is given. Hence,
    more than one RBC will move per time step.
    It is differentiated between capillaries and larger vessels. At larger Vessels 
    the RBCs are distributed based on the phase separation law. 
    """
    #@profile
    def __init__(self, G, invivo = True,dThreshold = 10.0,init = True,**kwargs):
        """Initializes a LinearSystemHtd instance.
        INPUT: G: Vascular graph in iGraph format.
               invivo: Boolean, whether the physiological blood characteristics 
                       are calculated using the invivo (=True) or invitro (=False)
                       equations
               dThreshold: Diameter threshold below which vessels are
                           considered capillaries, to which the reordering
                           algorithm can be applied.
               init: Assign initial conditions for RBCs (True) or keep old positions to
                        continue a simulation (False)
               **kwargs:
               ht0: The initial hematocrit in the capillary bed. ht0 needs to be provided if init=1 
               plasmaViscosity: The dynamic plasma viscosity. If not provided,
                                a literature value will be used.
               analyzeBifEvents: boolean if bifurcation events should be analyzed (Default = 0)
               innerDiam: boolean if inner or outer diamter of vessels is given in the graph 
                   (innerDiam = 1 --> inner diameter given) (Default = 0)
               species: 'rat', 'mouse' or 'human', default is 'rat'
        OUTPUT: None, however the following items are created:
                self.A: Matrix A of the linear system, holding the conductance
                        information.
                self.b: Vector b of the linear system, holding the boundary
                        conditions.
        """
        self._G = G
        self._P = Physiology(G['defaultUnits'])
        self._dThreshold = dThreshold
        self._invivo = invivo
        self._b = zeros(G.vcount())
        self._x = zeros(G.vcount())
        self._A = lil_matrix((G.vcount(),G.vcount()),dtype = float)
        self._eps = finfo(float).eps * 1e4
        #TODO those two are changed in evolve. depending if it is restarted or not. it would be more correct to do this here
        self._tPlot = 0.0
        self._tSample = 0.0
        self._filenamelist = []
        self._timelist = []
    	self._timelistAvg = []
        self._sampledict = {} 
        self._init = init
        self._scaleToDef = vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])
        self._vertexUpdate = None
        self._edgeUpdate = None
        G.es['source'] = [e.source for e in G.es]
        G.es['target'] = [e.target for e in G.es]
        G.es['countRBCs'] = [0]*G.ecount()
        G.es['crosssection'] = np.array([0.25*np.pi]*G.ecount())*np.array(G.es['diameter'])**2
        G.es['volume'] = [e['crosssection']*e['length'] for e in G.es]
        adjacent = []
        for i in xrange(G.vcount()):
            adjacent.append(G.adjacent(i))
        G.vs['adjacent'] = adjacent
        G['av'] = G.vs(av_eq=1).indices
        G['vv'] = G.vs(vv_eq=1).indices

        htd2htt = self._P.discharge_to_tube_hematocrit
        htt2htd = self._P.tube_to_discharge_hematocrit

        if kwargs.has_key('species'):
            self._species = kwargs['species']
        else:
            self._species = 'rat'

        print('Species')
        print(self._species)

        if kwargs.has_key('analyzeBifEvents'):
            self._analyzeBifEvents = kwargs['analyzeBifEvents']
        else:
            self._analyzeBifEvents = 0

        if kwargs.has_key('innerDiam'):
            self._innerDiam = kwargs['innerDiam']
        else:
            self._innerDiam = 0

        # Assure that both pBC and rBC edge properties are present:
        for key in ['pBC', 'rBC']:
            if not G.vs[0].attributes().has_key(key):
                G.vs[0][key] = None

        if self._analyzeBifEvents:
            self._rbcsMovedPerEdge = []
            self._edgesWithMovedRBCs = []
            self._rbcMoveAll = []
        else:
            if 'rbcMovedAll' in G.attributes():
                del(G['rbcMovedAll'])
            if 'rbcsMovedPerEdge' in G.attributes():
                del(G['rbcsMovedPerEdge'])
            if 'rbcMovedAll' in G.attributes():
                del(G['edgesMovedRBCs'])
        # Set initial pressure and flow to zero:
        if init:
            G.vs['pressure'] = zeros(G.vcount()) 
            G.es['flow'] = zeros(G.ecount())    

        G.vs['degree'] = G.degree()
        print('Initial flow, presure, ... assigned')

        if not init:
           if 'averagedCount' not in G.attributes():
               self._sampledict['averagedCount'] = 0
           else:
               self._sampledict['averagedCount'] = G['averagedCount']

        #Calculate total network Volume
        G['V'] = 0
        for e in G.es:
            G['V'] = G['V']+e['crosssection']*e['length']
        print('Total network volume calculated')

        # Compute the edge-specific minimal RBC distance:
        vrbc = self._P.rbc_volume(self._species)
        if self._innerDiam:
            G.es['minDist'] = [vrbc / (np.pi * e['diameter']**2 / 4) for e in G.es]
        else:
            eslThickness = self._P.esl_thickness
            G.es['minDist'] = [vrbc / (np.pi * (d-2*eslThickness(d))**2 / 4) for d in G.es['diameter']]
        G.es['nMax'] = [np.floor(e['length']/ e['minDist']) for e in G.es] 
        if len(G.es(nMax_eq=0)) > 0:
            sys.exit("BIGERROR nMax=0 exists --> check vessel lengths") 

        # Assign capillaries and non capillary vertices
        print('Start assign capillary and non capillary vertices')
        adjacent = [np.array(G.incident(i)) for i in G.vs]
        G.vs['isCap'] = [False]*G.vcount()
        self._interfaceVertices = []
        for i in xrange(G.vcount()):
            category = []
            for j in adjacent[i]:
                if G.es[int(j)]['diameter'] < dThreshold:
                    category.append(1)
                else:
                    category.append(0)
            if category.count(1) == len(category):
                G.vs[i]['isCap'] = True
            elif category.count(0) == len(category):
                G.vs[i]['isCap'] = False
            else:
                self._interfaceVertices.append(i)
        print('End assign capillary and non capillary vertices')

        # Arterial-side inflow:
        if init:
            if not 'httBC' in G.es.attribute_names():
                for vi in G['av']:
                    for ei in G.adjacent(vi):
                        G.es[ei]['httBC'] = self._P.tube_hematocrit(G.es[ei]['diameter'], 'a')

        print('Htt BC assigned')

        httBC_edges = G.es(httBC_ne=None).indices
        #Save initial value of httBC
        if init:
            G.es[httBC_edges]['httBC_init'] = G.es[httBC_edges]['httBC']
            httBCValue = np.mean(G.es[httBC_edges]['httBC'])
            for i in G.vs(vv_eq=1).indices:
                if G.es[G.adjacent(i)[0]]['httBC_init'] == None:
                    G.es[G.adjacent(i)[0]]['httBC_init'] = httBCValue

        # Assign initial RBC positions:
        if init:
            if 'ht0' not in kwargs.keys():
                print('ERROR no inital tube hematocrit given for distribution of RBCs')
            else:
                ht0 = kwargs['ht0']
            for e in G.es:
                lrbc = e['minDist']
                Nmax = max(int(np.floor(e['nMax'])), 1)
                if e['httBC'] is not None:
                    N = int(np.round(e['httBC'] * Nmax))
                else:
                    N = int(np.round(ht0 * Nmax))
                indices = sorted(np.random.permutation(Nmax)[:N])
                e['rRBC'] = np.array(indices) * lrbc + lrbc / 2.0
        print('Initial nRBC computed')    
        G.es['nRBC'] = [len(e['rRBC']) for e in G.es]

        if kwargs.has_key('plasmaViscosity'):
            self._muPlasma = kwargs['plasmaViscosity']
        else:
            self._muPlasma = self._P.dynamic_plasma_viscosity()

        # Compute nominal and specific resistance:
        self._update_nominal_and_specific_resistance()
        print('Resistance updated')

        # Compute the current tube hematocrit from the RBC positions:
        for e in G.es:
            e['htt'] = min(len(e['rRBC'])*vrbc/e['volume'],1)
            e['htd'] = min(htt2htd(e['htt'], e['diameter'], invivo), 1.0)
        print('Initial htt and htd computed')        

        # This initializes the full LS. Later, only relevant parts of
        # the LS need to be changed at any timestep. Also, RBCs are
        # removed from no-flow edges to avoid wasting computational
        # time on non-functional vascular branches / fragments:
        #Convert 'pBC' ['mmHG'] to default Units
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC'] = v['pBC']*self._scaleToDef
        self._update_eff_resistance_and_LS(None)
        print('Matrix created')
        self._solve('iterative2')
        print('Matrix solved')
        self._G.vs['pressure'] = self._x[:]
        #Convert deaultUnits to 'pBC' ['mmHG']
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC'] = v['pBC']/self._scaleToDef
        self._update_flow_and_velocity()
        print('Flow updated')
        self._verify_mass_balance()
        print('Mass balance verified updated')
        self._update_flow_sign()
        print('Flow sign updated')
        if 'posFirstLast' not in G.es.attribute_names():
            G.es['keep_rbcs'] = [[] for i in xrange(G.ecount())]
            G.es['posFirstLast'] = [None]*G.ecount()
            G.es['logNormal'] = [None]*G.ecount()
            httBCInit_edges = G.es(httBC_init_ne=None).indices
            print('Update logNormal')
            print(len(httBCInit_edges))
            print(G.ecount())
            for i in httBCInit_edges:
                if len(G.es[i]['rRBC']) > 0:
                    if G.es['sign'][i] == 1:
                        G.es[i]['posFirst_last'] = G.es['rRBC'][i][0]
                    else:
                        G.es[i]['posFirst_last'] = G.es['length'][i]-G.es['rRBC'][i][-1]
                else:
                    G.es[i]['posFirst_last'] = G.es['length'][i]
                G.es[i]['v_last'] = 0
                httBCValue = G.es[i]['httBC_init']
                if self._innerDiam:
                    LDValue = httBCValue
                else:
                    LDValue = httBCValue*(G.es[i]['diameter']/(G.es[i]['diameter']-2*eslThickness(G.es[i]['diameter'])))**2
                logNormalMu,logNormalSigma = self._compute_mu_sigma_inlet_RBC_distribution(LDValue)
                G.es[i]['logNormal'] = [logNormalMu,logNormalSigma]

        print('Initiallize posFirst_last')
        if 'signOld' in G.es.attribute_names():
            del(G.es['signOld'])
        self._update_out_and_inflows_for_vertices()
        print('updated out and inflows')

        #Calculate an estimated network turnover time (based on conditions at the beginning)
        flowsum = 0
        for vi in G['av']:
            for ei in G.adjacent(vi):
                flowsum = flowsum+G.es['flow'][ei]
        G['flowSumIn'] = flowsum
        G['Ttau'] = G['V']/flowsum
        print(flowsum)
        print(G['V'])
        print(self._eps)
        stdout.write("\rEstimated network turnover time Ttau=%f        \n" % G['Ttau'])

    #--------------------------------------------------------------------------
    def _compute_mu_sigma_inlet_RBC_distribution(self, httBC):
        """Updates the nominal and specific resistance of a given edge 
        sequence.
        INPUT: es: Sequence of edge indices as tuple. If not provided, all 
                   edges are updated.
        OUTPUT: None, the edge properties 'resistance' and 'specificResistance'
                are updated (or created).
        """
        mean_LD = httBC
        std_LD = 0.1*mean_LD
        
        #PDF log-normal
        f_x = lambda x,mu,sigma: 1./(x*np.sqrt(2*np.pi)*sigma)*np.exp(-1*(np.log(x)-mu)**2/(2*sigma**2))
        
        #PDF log-normal for line density
        f_LD = lambda z,mu,sigma: 1./((z-z**2)*np.sqrt(2*np.pi)*sigma)*np.exp(-1*(np.log(1./z-1)-mu)**2/(2*sigma**2))
        
        #f_mean integral dummy
        f_mean_LD_dummy = lambda z,mu,sigma: z*f_LD(z,mu,sigma)
        
        #calculate mean
        f_mean_LD = lambda mu,sigma: quad(f_mean_LD_dummy,0,1,args=(mu,sigma))[0]
        f_mean_LD_Calc = np.vectorize(f_mean_LD)
        
        #f_var integral dummy
        f_var_LD_dummy = lambda z,mu,sigma: (z-mean_LD)**2*f_LD(z,mu,sigma)
        
        #calculate mean
        f_var_LD = lambda mu,sigma: quad(f_var_LD_dummy,0,1,args=(mu,sigma))[0]
        f_var_LD_Calc = np.vectorize(f_var_LD)
        
        #Set up system of equations
        def f_moments_LD(m):
            x,y = m
            return (f_mean_LD_Calc(x,y)-mean_LD,f_var_LD_Calc(x,y)-std_LD**2)

        optionsSolve = {}
        optionsSolve['xtol'] = 1e-20
        if mean_LD < 0.35:
            sol = root(f_moments_LD,(0.89,0.5),method='lm',options=optionsSolve)
        elif mean_LD > 0.63:
            sol = root(f_moments_LD,(-0.6,0.45),method='lm',options=optionsSolve)
        else:
            sol = root(f_moments_LD,(mean_LD,std_LD),method='lm',options=optionsSolve)
        mu = sol['x'][0]
        sigma = sol['x'][1]

        return mu,sigma

    #--------------------------------------------------------------------------

    def _update_nominal_and_specific_resistance(self, esequence = None):
        """Updates the nominal and specific resistance of a given edge 
        sequence.
        INPUT: esequence: Sequence of edge indices as tuple. If not provided, all 
                   edges are updated. (WARNING: list should only contain int no np.int)
        OUTPUT: None, the edge properties 'resistance' and 'specificResistance'
                are updated (or created).
        """
        G = self._G
        muPlasma = self._muPlasma
        pi = np.pi  

        if esequence is None:
            es = G.es
        else:
            es = G.es(esequence)
        es['specificResistance'] = [128 * muPlasma / (pi * d**4)
                                        for d in es['diameter']]

        es['resistance'] = [l * sr for l, sr in zip(es['length'], 
                                                es['specificResistance'])]
        self._G = G

    #--------------------------------------------------------------------------

    def _update_hematocrit(self, esequence = None):
        """Updates the tube hematocrit of a given edge sequence.
        INPUT: es: Sequence of edge indices as tuple. If not provided, all 
                   edges are updated. (WARNING: list should only contain int no np.int)
        OUTPUT: None, the edge property 'htt' is updated (or created).
        """
        G = self._G
        htt2htd = self._P.tube_to_discharge_hematocrit
        invivo = self._invivo
        vrbc = self._P.rbc_volume(self._species)

        if esequence is None:
            es = G.es
        else:
            es = G.es(esequence)

        es['htt'] = [min(e['nRBC'] * vrbc / e['volume'],1) for e in es]
        es['htd'] = [min(htt2htd(e['htt'], e['diameter'], invivo), 1.0) for e in es]

    	self._G = G

    #--------------------------------------------------------------------------

    def _update_local_pressure_gradient(self):
        """Updates the local pressure gradient at all vertices.
        INPUT: None
        OUTPUT: None, the edge property 'lpg' is updated (or created, if it did
                not exist previously)
        """
        G = self._G
        G.es['lpg'] = np.array(G.es['specificResistance']) * \
                      np.array(G.es['flow'])

        self._G = G
    #--------------------------------------------------------------------------

    def _update_interface_vertices(self):
        """(Re-)assigns each interface vertex to either the capillary or non-
        capillary group, depending on whether ist inflow is exclusively from
        capillaries or not.
        """
        G = self._G
        dThreshold = self._dThreshold

        for v in self._interfaceVerticesI:
            p = G.vs[v]['pressure']
            G.vs[v]['isCap'] = True
            for n in self._interfaceNoncapNeighborsVI[v]:
                if G.vs[n]['pressure'] > p:
                    G.vs[v]['isCap'] = False
                    break

    #--------------------------------------------------------------------------

    def _update_flow_sign(self):
        """Updates the sign of the flow. The flow is defined as having a
        positive sign if its direction is from edge source to target, negative
        if vice versa and zero otherwise (in case of equal pressures).
        INPUT: None
        OUTPUT: None (the value of the edge property 'sign' will be updated to
                one of [-1, 0, 1])
        """
        G = self._G
        if 'sign' in G.es.attributes() and None not in G.es['sign']:
            G.es['signOld'] = G.es['sign']
        G.es['sign'] = [np.sign(G.vs[source]['pressure'] -
                                G.vs[target]['pressure']) for source,target in zip(G.es['source'],G.es['target'])]

    #-------------------------------------------------------------------------
    #@profile
    def _update_out_and_inflows_for_vertices(self):
        """Calculates the in- and outflow edges for vertices at the beginning.
        Afterwards in every single timestep it is check if something changed
        INPUT: None 
        OUTPUT: None, however the following parameters will be updated:
                G.vs['inflowE']: Time until next RBC reaches bifurcation.
                G.vs['outflowE']: Index of edge in which the RBC resides.
        """    
        G = self._G
        eslThickness = self._P.esl_thickness
        #Beginning    
        inEdges = []
        outEdges = []
        divergentV = []
        convergentV = []
        connectingV = []
        doubleConnectingV = []
        noFlowV = []
        noFlowE = []
        vertices = []
        dThreshold = self._dThreshold
        count = 0
        interfaceVertices = self._interfaceVertices
        print('In update out and inflows')
        if not 'sign' in G.es.attributes() or not 'signOld' in G.es.attributes():
            print('Initial vType Update')
            for v in G.vs:
                vI = v.index
                outE = []
                inE = []
                noFlowE = []
                pressure = G.vs[vI]['pressure']
                adjacents = G.adjacent(vI)
                for j,nI in enumerate(G.neighbors(vI)):
                    #outEdge
                    if pressure > G.vs[nI]['pressure']: 
                        outE.append(adjacents[j])
                    elif pressure == G.vs[nI]['pressure']: 
                        noFlowE.append(adjacents[j])
                    #inflowEdge
                    else: #G.vs[vI]['pressure'] < G.vs[nI]['pressure']
                        inE.append(adjacents[j])
                #Deal with vertices at the interface
                #isCap is defined based on the diameter of the InflowEdge
                if vI in interfaceVertices: 
                    capCountIn = 0
                    capCount = 0
                    for j in adjacents:
                        if G.es[j]['diameter'] <= dThreshold:
                            capCount += 1
                            if j in inE:
                                capCountIn += 1
                    if capCountIn == len(inE) and capCount > len(adjacents)/2.:
                        G.vs[vI]['isCap'] = True
                    else:
                        G.vs[vI]['isCap'] = False
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
                        edgeVI = G.adjacent(vI)[0]
                        G.es[edgeVI]['httBC'] = None
                        G.es[edgeVI]['posFirst_last'] = None
                        G.es[edgeVI]['v_last'] = None
                        print(G.es[edgeVI]['v_last'])
                    elif len(inE) == 0 and len(outE) == 0:
                        print('WARNING changed to noFlow edge')
                        edgeVI = G.adjacent(vI)[0]
                        noFlowV.append(vI)
                        noFlowE.append(edgeVI)
                    else:
                        print('ERROR in defining in and outlets')
                        print(vI)
                elif vI in G['vv']:
                    if len(inE) == 1 and len(outE) == 0:
                        pass
                    elif len(inE) == 0 and len(outE) == 1:
                        print('WARNING1 boundary condition changed: from vv --> av')
                        print(vI)
                        G.vs[vI]['av'] = 1
                        G.vs[vI]['vv'] = 0
			G.vs[vI]['vType'] = 1
                        edgeVI = G.adjacent(vI)[0]
                        G.es[edgeVI]['httBC'] = G.es[edgeVI]['httBC_init']
                        if len(G.es[edgeVI]['rRBC']) > 0:
                            if G.es['sign'][edgeVI] == 1:
                                G.es[edgeVI]['posFirst_last'] = G.es['rRBC'][edgeVI][0]
                            else:
                                G.es[edgeVI]['posFirst_last'] = G.es['length'][edgeVI]-G.es['rRBC'][edgeVI][-1]
                        else:
                            G.es[edgeVI]['posFirst_last'] = G.es['length'][edgeVI]
                        G.es[edgeVI]['v_last'] = G.es['v'][edgeVI]
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
                        if G.es['flow'][i] > 5.0e-08:
                            print('FLOWERROR')
                            print(vI)
                            print(inE)
                            print(outE)
                            print(noFlowE)
                            print(i)
                            print('Flow and diameter')
                            print(G.es['flow'][i])
                            print(G.es['diameter'][i])
                        noFlowE.append(i)
                    inE = []
                    outE = []
                    noFlowV.append(vI)
                    print('noFlow V')
                    print(vI)
                inEdges.append(inE)
                outEdges.append(outE)
            G.vs['inflowE'] = inEdges
            G.vs['outflowE'] = outEdges
            G.es['noFlow'] = [0]*G.ecount()
            if noFlowE != []:
                noFlowE = np.unique(noFlowE)
                G.es[noFlowE]['noFlow'] = [1]*len(noFlowE)
            G['divV'] = divergentV
            G['conV'] = convergentV
            G['connectV'] = connectingV
            G['dConnectV'] = doubleConnectingV
            G['noFlowV'] = noFlowV
            print('assign vertex types')
            #vertex type av = 1, vv = 2,divV = 3, conV = 4, connectV = 5, dConnectV = 6, noFlowV = 7
            G.vs['vType'] = [0]*G.vcount()
            G['av'] = G.vs(av_eq=1).indices
            G['vv'] = G.vs(vv_eq=1).indices
            G.vs[G['av']]['vType'] = [1]*len(G['av'])
            G.vs[G['vv']]['vType'] = [2]*len(G['vv'])
            G.vs[G['divV']]['vType'] = [3]*len(G['divV'])
            G.vs[G['conV']]['vType'] = [4]*len(G['conV'])
            G.vs[G['connectV']]['vType'] = [5]*len(G['connectV'])
            G.vs[G['dConnectV']]['vType'] = [6]*len(G['dConnectV'])
            G.vs[G['noFlowV']]['vType'] = [7]*len(G['noFlowV'])
            if len(G.vs(vType_eq=0).indices) > 0:
                print('BIGERROR vertex type not assigned')
                print(len(G.vs(vType_eq=0).indices))
            del(G['divV'])
            del(G['conV'])
            del(G['connectV'])
            del(G['dConnectV'])
        #Every Time Step
        else:
            if G.es['sign'] != G.es['signOld']:
                sign = np.array(G.es['sign'])
                signOld = np.array(G.es['signOld'])
                sumTes = abs(sign+signOld)
                #find edges where sign change took place
                edgeList = np.array(np.where(sumTes < abs(2))[0])
                edgeList = edgeList.tolist()
                sign0 = G.es(sign_eq=0,signOld_eq=0).indices
                for e in sign0:
                    edgeList.remove(e)
                stdout.flush()
                vertices = []
                for e in edgeList:
                    for vI in G.es[int(e)].tuple:
                        vertices.append(vI)
                vertices = np.unique(vertices).tolist()
                count = 0
                for vI in vertices:
                    #vI = v.index
                    count += 1
                    vI = int(vI)
                    outE = []
                    inE = []
                    noFlowE = []
                    neighbors = G.neighbors(vI)
                    pressure = G.vs[vI]['pressure']
                    adjacents = G.adjacent(vI)
                    for j in xrange(len(neighbors)):
                        nI = neighbors[j]
                        #outEdge
                        if pressure > G.vs[nI]['pressure']:
                            outE.append(adjacents[j])
                        elif pressure == G.vs[nI]['pressure']:
                            noFlowE.append(adjacents[j])
                        #inflowEdge
                        else: #G.vs[vI]['pressure'] < G.vs[nI]['pressure']
                            inE.append(adjacents[j])
                    #Deal with vertices at the interface
                    #isCap is defined based on the diameter of the InflowEdge
                    if vI in interfaceVertices: 
                        capCountIn = 0
                        capCount = 0
                        for j in adjacents:
                            if G.es[j]['diameter'] <= dThreshold:
                                capCount += 1
                                if j in inE:
                                    capCountIn += 1
                        if capCountIn == len(inE) and capCount > len(adjacents)/2.:
                            G.vs[vI]['isCap'] = True
                        else:
                            G.vs[vI]['isCap'] = False
                    #Group into divergent, convergent, connecting, doubleConnecting and noFlow Vertices
                    #it is now a divergent Vertex
                    if len(outE) > len(inE) and len(inE) >= 1:
                        #Find history of vertex
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow'] = [0]*len(inE)
                            G.es[outE]['noFlow'] = [0]*len(outE)
                        G.vs[vI]['vType'] = 3
                        G.vs[vI]['inflowE'] = inE
                        G.vs[vI]['outflowE'] = outE
                    #it is now a convergent Vertex
                    elif len(inE) > len(outE) and len(outE) >= 1:
                        if G.vs[vI]['vType']== 7:
                            G.es[inE]['noFlow'] = [0]*len(inE)
                            G.es[outE]['noFlow'] = [0]*len(outE)
                        G.vs[vI]['vType'] = 4
                        G.vs[vI]['inflowE'] = inE
                        G.vs[vI]['outflowE'] = outE
                    #it is now a connecting Vertex
                    elif len(outE) == len(inE) and len(outE) == 1:
                        if G.vs[vI]['vType'] == 7:
                            G.es[inE]['noFlow'] = [0]*len(inE)
                            G.es[outE]['noFlow'] = [0]*len(outE)
                        G.vs[vI]['vType'] = 5
                        G.vs[vI]['inflowE'] = inE
                        G.vs[vI]['outflowE'] = outE
                    #it is now a double connecting Vertex
                    elif len(outE) == len(inE) and len(outE) == 2:
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow'] = [0]*len(inE)
                            G.es[outE]['noFlow'] = [0]*len(outE)
                        G.vs[vI]['vType'] = 6
                        G.vs[vI]['inflowE'] = inE
                        G.vs[vI]['outflowE'] = outE
                    elif vI in G['av']:
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow'] = [0]*len(inE)
                            G.es[outE]['noFlow'] = [0]*len(outE)
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
                            print('WARNING direction out av changed to vv')
                            print(vI)
                            G.vs[vI]['av'] = 0
                            G.vs[vI]['vv'] = 1
                            G.vs[vI]['vType'] = 2
                            edgeVI = G.adjacent(vI)[0]
                            G.es[edgeVI]['httBC'] = None
                            G.es[edgeVI]['posFirst_last'] = None
                            G.es[edgeVI]['v_last'] = None
                            G.vs[vI]['inflowE'] = inE
                            G.vs[vI]['outflowE'] = outE
                    elif vI in G['vv']:
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow'] = [0]*len(inE)
                            G.es[outE]['noFlow'] = [0]*len(outE)
                        if G.vs[vI]['rBC'] != None:
                            for j in G.adjacent(vI):
                                if G.es[j]['flow'] > 1e-6:
                                    print(' ')
                                    print(vI)
                                    print(len(G.vs[vI]['inflowE']))
                                    print(len(G.vs[vI]['outflowE']))
                                    print('ERROR flow direction of out vertex changed')
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
                            edgeVI = G.adjacent(vI)[0]
                            G.es[edgeVI]['httBC'] = G.es[edgeVI]['httBC_init']
                            if len(G.es[edgeVI]['rRBC']) > 0:
                                if G.es['sign'][edgeVI] == 1:
                                    G.es[edgeVI]['posFirst_last'] = G.es['rRBC'][edgeVI][0]
                                else:
                                    G.es[edgeVI]['posFirst_last'] = G.es['length'][edgeVI]-G.es['rRBC'][edgeVI][-1]
                            else:
                                G.es[edgeVI]['posFirst_last'] = G.es['length'][edgeVI]
                            G.es[edgeVI]['v_last'] = G.es[edgeVI]['v']
                            G.vs[vI]['inflowE'] = inE
                            G.vs[vI]['outflowE'] = outE
                            if G.es[edgeVI]['logNormal'] == None:
                                httBCValue = G.es[edgeVI]['httBC_init']
                                if self._innerDiam:
                                    LDValue = httBCValue
                                else:
                                    print('httBCValue')
                                    print(httBCValue)
                                    print(edgeVI)
                                    LDValue = httBCValue*(G.es[edgeVI]['diameter']/(G.es[edgeVI]['diameter']-2*eslThickness(G.es[edgeVI]['diameter'])))**2
                                logNormalMu,logNormalSigma = self._compute_mu_sigma_inlet_RBC_distribution(LDValue)
                                G.es[edgeVI]['logNormal'] = [logNormalMu,logNormalSigma]
                    #it is now a noFlow Vertex
                    else:
                        if G.vs[vI]['degree']==1 and len(inE) == 1 and len(outE) == 0:
                            print('WARNING2 changed from noFlow to outflow')
                            print(vI)
                            G.vs[vI]['av'] = 0
                            G.vs[vI]['vv'] = 1
                            G.vs[vI]['vType'] = 2
                            edgeVI = G.adjacent(vI)[0]
                            G.es[edgeVI]['httBC'] = None
                            G.es[edgeVI]['posFirst_last'] = None
                            G.es[edgeVI]['v_last'] = None
                        elif G.vs[vI]['degree']==1 and len(inE) == 0 and len(outE) == 1:
                            print('WARNING2 changed from noFlow to inflow')
                            print(vI)
                            G.vs[vI]['av'] = 1
                            G.vs[vI]['vv'] = 0
                            G.vs[vI]['vType'] = 1
                            edgeVI = G.adjacent(vI)[0]
                            G.es[edgeVI]['httBC'] = G.es[edgeVI]['httBC_init']
                            if len(G.es[edgeVI]['rRBC']) > 0:
                                if G.es['sign'][edgeVI] == 1:
                                    G.es[edgeVI]['posFirst_last'] = G.es['rRBC'][edgeVI][0]
                                else:
                                    G.es[edgeVI]['posFirst_last'] = G.es['length'][edgeVI]-G.es['rRBC'][edgeVI][-1]
                            else:
                                G.es[edgeVI]['posFirst_last'] = G.es['length'][edgeVI]
                            G.es[edgeVI]['v_last'] = G.es[edgeVI]['v']
                        else:
                            noFlowEdges = []
                            for i in G.adjacent(vI):
                                if G.es['flow'][i] > 5.0e-08:
                                    print('FLOWERROR')
                                    print(vI)
                                    print(inE)
                                    print(outE)
                                    print(noFlowE)
                                    print(i)
                                    print('Flow and diameter')
                                    print(G.es['flow'][i])
                                    print(G.es['diameter'][i])
                                noFlowEdges.append(i)
                            G.vs[vI]['vType'] = 7
                            G.es[noFlowEdges]['noFlow'] = [1]*len(noFlowEdges)
                            G.vs[vI]['inflowE'] = []
                            G.vs[vI]['outflowE'] = []

            G['av'] = G.vs(av_eq=1).indices
            G['vv'] = G.vs(vv_eq=1).indices
        stdout.flush()
        if len(G.vs(av_eq=1,degree_gt=1))>0:
            print('BIGERROR av')
            G['avProb'] = G.vs(av_eq=1,degree_gt=1).indices
            vgm.write_pkl(G,'Gavprob.pkl') 
        if len(G.vs(vv_eq=1,degree_gt=1))>0:
            print('BIGERROR vv')
            G['vvProb'] = G.vs(vv_eq=1,degree_gt=1).indices
            vgm.write_pkl(G,'Gvvprob.pkl') 

    #--------------------------------------------------------------------------

    def _update_flow_and_velocity(self):
        """Updates the flow and red blood cell velocity in all vessels
        INPUT: None
        OUTPUT: None
        """

        G = self._G
        invivo = self._invivo
        vf = self._P.velocity_factor
        vrbc = self._P.rbc_volume(self._species)
        vfList = [1.0 if htt == 0.0 else max(1.0,vf(d, invivo, tube_ht=htt)) for d,htt in zip(G.es['diameter'],G.es['htt'])]

        self._G = run_faster.update_flow_and_v(self._G,self._invivo,vfList,vrbc)
        G =  self._G

    #--------------------------------------------------------------------------
    #@profile
    def _update_eff_resistance_and_LS(self, vertex = None):
        """Constructs the linear system A x = b where the matrix A contains the
        conductance information of the vascular graph, the vector b specifies
        the boundary conditions and the vector x holds the pressures at the
        vertices (for which the system needs to be solved). x will have the
        same units of [pressure] as the pBC vertices.

        Note that in this approach, A and b contain a mixture of dimensions,
        i.e. A and b have dimensions of [1.0] and [pressure] in the pBC case,
        [conductance] and [conductance*pressure] otherwise, the latter being
        rBCs. This has the advantage that no re-indexing is required as the
        matrices contain all vertices.
        INPUT: 
        OUTPUT: A: Matrix A of the linear system, holding the conductance
                   information.
                b: Vector b of the linear system, holding the boundary
                   conditions.
        """


        G = self._G
        P = self._P
        A = self._A
        b = self._b
        x = self._x
        invivo = self._invivo

        htt2htd = P.tube_to_discharge_hematocrit
        nurel = P.relative_apparent_blood_viscosity

        #TODO for the av network len(vertex) is always around 2000 --> does it really make sense to not always update the whole matrix?
        if vertex is None:
            vertexList = xrange(G.vcount())
            edgeList = xrange(G.ecount())
        else:
            neighborsList = []
            edgeList = []
            for i in vertex:
                neighborsList = neighborsList + G.neighbors(i)
                edgeList = edgeList + G.incident(i)
            vertexList=np.concatenate([vertex,np.array(neighborsList)])
            vertexList = np.unique(vertexList).tolist()
            edgeList = np.unique(edgeList).tolist()
        dischargeHt = [min(htt2htd(e, d, invivo), 1.0) for e,d in zip(G.es[edgeList]['htt'],G.es[edgeList]['diameter'])]
        G.es[edgeList]['effResistance'] = [ res * nurel(max(d,4.0), min(dHt,0.6),invivo) for res,dHt,d in zip(G.es[edgeList]['resistance'], \
            dischargeHt,G.es[edgeList]['diameter'])]

        edgeList = G.es(edgeList)
        vertexList = G.vs(vertexList)
        for vertex in vertexList:
            i = vertex.index
            A.data[i] = []
            A.rows[i] = []
            b[i] = 0.0
            if vertex['pBC'] is not None:
                A[i,i] = 1.0
                b[i] = vertex['pBC']
            else:
                aDummy = 0
                k = 0
                neighbors = []
                for edge in G.adjacent(i,'all'):
                    if G.is_loop(edge):
                        continue
                    j = G.neighbors(i)[k]
                    k += 1
                    conductance = 1 / G.es[edge]['effResistance']
                    neighbor = G.vs[j]
                    # +=, -= account for multiedges
                    aDummy += conductance
                    if neighbor['pBC'] is not None:
                        b[i] = b[i] + neighbor['pBC'] * conductance
                    #elif neighbor['rBC'] is not None:
                     #   b[i] = b[i] + neighbor['rBC']
                    else:
                        if j not in neighbors:
                            A[i,j] = - conductance
                        else:
                            A[i,j] = A[i,j] - conductance
                    neighbors.append(j)
                    if vertex['rBC'] is not None:
                        b[i] += vertex['rBC']
                A[i,i] = aDummy

        self._A = A
        self._b = b
        self._G = G

    #--------------------------------------------------------------------------
    #@profile
    def _propagate_rbc(self):
        """This assigns the current bifurcation-RBC to a new edge and
        propagates all RBCs until the next RBC reaches at a bifurcation.
        INPUT: None
        OUTPUT: None
        """
        G = self._G
        dt = self._dt
        eps = self._eps
        Physiol = self._P
        edgeList0 = G.es(noFlow_eq=0).indices #No flow Edges are not considered for the propagation of RBCs
        if self._analyzeBifEvents:
            rbcsMovedPerEdge = []
            edgesWithMovedRBCs = []
            rbcMoved = 0
        edgeList = G.es[edgeList0]
        #Edges are sorted based on the pressure at the outlet
        pOut = [G.vs[e['target']]['pressure'] if e['sign'] == 1.0 else G.vs[e['source']]['pressure']
            for e in edgeList]
        sortedE = zip(pOut,edgeList0)
        sortedE.sort()
        sortedE = [i[1] for i in sortedE]
        convEdges = [0]*G.ecount()
        httBCDoneEdges = [0]*G.ecount()
        nRBCList=np.array(G.es['nRBC'])
        edgeUpdate = []   #Edges where the number of RBCs changed --> need to be updated
        vertexUpdate = [] #Vertices where the number of RBCs changed in adjacent edges --> need to be updated
        #SECOND step go through all edges from smallest to highest pressure and move RBCs
        for ei in sortedE:
            noBifEvents = 0
            edgesInvolved = [] #all edges connected to the bifurcation vertex
            e = G.es[ei]
            sign = e['sign']
            vi = e['target'] if sign == 1 else e['source']
            vertex = G.vs[vi]
            edgesInvolved = G.incident(vi)
            nRBCSumBefore = np.sum(nRBCList[edgesInvolved])
            overshootsNo = 0 #Reset - Number of overshoots acutally taking place (considers possible number of bifurcation events)
            #TODO what is the difference between noFlow vertices and noFlow edges? are both checks necessary
            boolHttEdge, boolHttEdge2, boolHttEdge3 = 0,0,0
            if convEdges[ei] == 0 and vertex['vType'] != 7:
                bifRBCsIndex = self._initial_propagate_and_compute_bifRBCsIndex(e,sign)
                noBifEvents = len(bifRBCsIndex)
                if vertex['vType'] == 4 or vertex['vType'] == 6:
                    convEdges[ei] = 1
                #-------------------------------------------------------------------------------------------
                if noBifEvents > 0:
                    if vertex['vType'] == 2: #OUTFLOW Vertex
                        overshootsNo = noBifEvents
                        self._remove_RBCs(e,overshootsNo)
                        vertexUpdate.append(e['target'])
                        vertexUpdate.append(e['source'])
                        edgeUpdate.append(ei)
                    #-------------------------------------------------------------------------------------------
                    elif vertex['vType'] == 5: #CONNECTING Vertex
                        #print('CONNECTING')
                        oe = G.es[vertex['outflowE'][0]]
                        posNoBifEvents = self._calculate_possible_number_of_BifEvents(oe)
                        overshootsNo,posBifRBCsIndex = self._compare_noBifEvents_to_posNoBifEvents(\
                            posNoBifEvents,noBifEvents,bifRBCsIndex,sign)
                        if overshootsNo > 0:
                            overshootTime = self._compute_overshootTime(e,posBifRBCsIndex,sign)
                            position = self._compute_unconstrained_RBC_positions(oe,overshootTime,signConsidered=0)
                            position = self._check_overshootingNewVessel_and_overtakingRBCsInNewVessel(oe,position)
                            overshootsNoReduce = (position<0).tolist().count(True)
                            overshootsNo = overshootsNo-overshootsNoReduce
                            position = position[-1*overshootsNo::]
                            if overshootsNo > 0:
                                oe['countRBCs'] += len(position)
                                self._move_RBCs(oe,position,signConsidered=0)
                                self._remove_RBCs(e,overshootsNo)
                        noStuckRBCs = len(bifRBCsIndex)-overshootsNo
                        self._push_stuckRBCs_back(e,noStuckRBCs)
                    #-------------------------------------------------------------------------------------------
                    elif vertex['vType'] == 3: #DIVERGENT Vertex
                        #print('DIVERGENT')
                        outEdges = vertex['outflowE']
                        boolTrifurcation = 0
                        if len(outEdges) > 2:
                            boolTrifurcation = 1
                        if vertex['isCap']:
                            nonCap = 0
                            preferenceList = [x[1] for x in sorted(zip(np.array(G.es[outEdges]['flow'])/np.array(G.es[outEdges]['crosssection']), \
                                outEdges), reverse=True)]
                        else:
                            nonCap = 1
                            preferenceList = [x[1] for x in sorted(zip(G.es[outEdges]['flow'], outEdges), reverse=True)]
                            if boolTrifurcation:
                                ratio1,ratio2,ratio3 = np.array(G.es[preferenceList]['flow'])/e['flow']
                            else:
                                ratio1 = Physiol.phase_separation_effect(G.es[preferenceList[0]]['flow']/e['flow'], \
                                    G.es[preferenceList[0]]['diameter'],G.es[preferenceList[1]]['diameter'],e['diameter'],e['htd'])
                                #it can happen that ratio1 < 0.5 even if the flowRate is larger. in this case the preferenceList is changed
                                if ratio1 < 0.5:
                                    ratio1 = 1.0 - ratio1
                                    preferenceList = preferenceList[::-1]
                                ratio2 = 1.0 -  ratio1
                                ratio3 = 0.
                        oe = G.es[preferenceList[0]]
                        oe2 = G.es[preferenceList[1]]
                        posNoBifEventsPref = self._calculate_possible_number_of_BifEvents(oe)
                        posNoBifEventsPref2 = self._calculate_possible_number_of_BifEvents(oe2)
                        if boolTrifurcation:
                            oe3 = G.es[preferenceList[2]]
                            posNoBifEventsPref3 = self._calculate_possible_number_of_BifEvents(oe3)
                        else:
                            posNoBifEventsPref3 = 0
                        posNoBifEvents = int(posNoBifEventsPref+posNoBifEventsPref2+posNoBifEventsPref3)
                        overshootsNo,posBifRBCsIndex = self._compare_noBifEvents_to_posNoBifEvents(\
                            posNoBifEvents,noBifEvents,bifRBCsIndex,sign)
                        if nonCap:
                            overshootsNo1, overshootsNo2, overshootsNo3 = \
                                self._nonCapDiv_compute_overshootNos_from_ratios(boolTrifurcation,overshootsNo,ratio1,ratio2,ratio3)
                            overshootsNo1,overshootsNo2,overshootsNo3 = self._nonCapDiv_compare_overshootNos_to_posBifEvents(\
                                overshootsNo1,overshootsNo2,overshootsNo3,posNoBifEventsPref,posNoBifEventsPref2,posNoBifEventsPref3,overshootsNo)
                            overshootsNo = int(overshootsNo1 + overshootsNo2 + overshootsNo3)
                            posNoBifEvents = overshootsNo
                            posBifRBCsIndex = posBifRBCsIndex[-posNoBifEvents::] if sign == 1.0 \
                                else posBifRBCsIndex[:posNoBifEvents]
                        if overshootsNo > 0:
                            overshootTime = self._compute_overshootTime(e,posBifRBCsIndex,sign)
                            #Calculate position of overshootRBCs in every outEdge. flow direction of outEdge is considered
                            position1 = self._compute_unconstrained_RBC_positions(oe,overshootTime,signConsidered=1)
                            position2 = self._compute_unconstrained_RBC_positions(oe2,overshootTime,signConsidered=1)
                            if boolTrifurcation:
                                position3 = self._compute_unconstrained_RBC_positions(oe3,overshootTime,signConsidered=1)
                            if nonCap:
                                countNo1,countNo2,countNo3 = 0,0,0
                                positionPref1,positionPref2,positionPref3 = [],[],[]
                                distributeOrderNextIteration = [1,2,3]
                                for i in xrange(overshootsNo):
                                    distributeOrder = distributeOrderNextIteration[:]
                                    boolRBCassigned = 0
                                    for j in distributeOrder:
                                        if j == 1:
                                            if countNo1 < overshootsNo1:
                                                positionPref1 = self._divergent_add_RBCs_to_positionPref(oe,position1,positionPref1,i)
                                                countNo1 += 1
                                                distributeOrderNextIteration = [2,3,1]
                                                boolRBCassigned = 1
                                                break
                                        elif j == 2:
                                            if countNo2 < overshootsNo2:
                                                positionPref2 = self._divergent_add_RBCs_to_positionPref(oe2,position2,positionPref2,i)
                                                countNo2 += 1
                                                distributeOrderNextIteration = [3,1,2]
                                                boolRBCassigned = 1
                                                break
                                        elif j == 3:
                                            if countNo3 < overshootsNo3:
                                                positionPref3 = self._divergent_add_RBCs_to_positionPref(oe3,position3,positionPref3,i)
                                                countNo3 += 1
                                                distributeOrderNextIteration = [1,2,3]
                                                boolRBCassigned = 1
                                                break
                                    if not boolRBCassigned:
                                        print('BIGERROR all overshootRBCS should fit 1')
                                positionPref1 = self._nonCapDiv_push_RBCs_forward_to_fit(oe,positionPref1)
                                positionPref2 = self._nonCapDiv_push_RBCs_forward_to_fit(oe2,positionPref2)
                                if boolTrifurcation:
                                    positionPref3 = self._nonCapDiv_push_RBCs_forward_to_fit(oe3,positionPref3)
                            else:
                                #To begin with it is tried if all RBCs fit into the prefered outEdge. The time of arrival at the bifurcation is taken into account
                                #RBCs which would be too close together are put into the other edge
                                countPref1,countPref2,countPref3 = 0,0,0
                                positionPref1,positionPref2,positionPref3 = [],[],[]
                                pref1Full,pref2Full,pref3Full = 0,0,0
                                for i in xrange(overshootsNo): #Loop over all movable RBCs (begin with RBC which overshot the most)
                                    index = -1*(i+1) if sign == 1.0 else i
                                    index1 = -1*(i+1) if oe['sign'] == 1.0 else i
                                    index2 = -1*(i+1) if oe2['sign'] == 1.0 else i
                                    if boolTrifurcation:
                                        index3 = -1*(i+1) if oe3['sign'] == 1.0 else i
                                    if posNoBifEventsPref > countPref1 and pref1Full == 0:
                                        if positionPref1 != []: 
                                            dist1 = positionPref1[-1]-position1[index1] if oe['sign'] == 1.0 \
                                                else position1[index1]-positionPref1[-1]
                                            if dist1 < oe['minDist']:
                                                if posNoBifEventsPref2 > countPref2 and pref2Full == 0:
                                                    if positionPref2 != []:
                                                        dist2 = positionPref2[-1]-position2[index2] if oe2['sign'] == 1.0 \
                                                            else position2[index2]-positionPref2[-1]
                                                        if dist2 < oe2['minDist']:
                                                            if boolTrifurcation and posNoBifEventsPref3 > countPref3 and pref3Full == 0:
                                                                if positionPref3 != []: 
                                                                    dist3 = positionPref3[-1]-position3[index3] if oe3['sign'] == 1.0 \
                                                                        else position3[index3]-positionPref3[-1]
                                                                    if dist3 < oe3['minDist']:
                                                                        newOutEdge,[pref1Full,pref2Full,pref3Full] = self._CapDiv_compute_timeBlocked(\
                                                                                [dist1,dist2,dist3],[oe,oe2,oe3],[positionPref1,positionPref2,positionPref3],\
                                                                                [pref1Full,pref2Full,pref3Full])
                                                                        if newOutEdge == -1:
                                                                            break
                                                                        if newOutEdge == 1:
                                                                            positionPref1,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe,positionPref1,case=1)
                                                                            countPref1 += 1
                                                                        elif newOutEdge == 2:
                                                                            positionPref2,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe2,positionPref2,case=1)
                                                                            countPref2 += 1
                                                                        elif newOutEdge == 3:
                                                                            positionPref3,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe3,positionPref3,case=1)
                                                                            countPref3 += 1
                                                                    else:
                                                                        positionPref3.append(position3[index3])
                                                                        countPref3 += 1
                                                                else:
                                                                    positionPref3 = self._divergent_add_RBCs_to_positionPref(oe3,position3,positionPref3,i)
                                                                    countPref3 += 1
                                                            else:
                                                                newOutEdge,[pref1Full,pref2Full,pref3Full] = self._CapDiv_compute_timeBlocked(\
                                                                    [dist1,dist2,None],[oe,oe2,None],[positionPref1,positionPref2,None],[pref1Full,pref2Full,pref3Full])
                                                                if newOutEdge == -1:
                                                                    break
                                                                if newOutEdge == 1:
                                                                    positionPref1,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe,positionPref1,case=1)
                                                                    countPref1 += 1
                                                                elif newOutEdge == 2:
                                                                    positionPref2,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe2,positionPref2,case=1)
                                                                    countPref2 += 1
                                                        else:
                                                            positionPref2.append(position2[index2])
                                                            countPref2 += 1
                                                    else:
                                                        positionPref2 = self._divergent_add_RBCs_to_positionPref(oe2,position2,positionPref2,i)
                                                        countPref2 += 1
                                                else:
                                                    if boolTrifurcation and posNoBifEventsPref3 > countPref3 and pref3Full == 0:
                                                        if positionPref3 != []:
                                                            dist3 = positionPref3[-1]-position3[index3] if oe3['sign'] == 1.0 \
                                                                else position3[index3]-positionPref3[-1]
                                                            if dist3 < oe3['minDist']:
                                                                newOutEdge,[pref1Full,pref2Full,pref3Full] = self._CapDiv_compute_timeBlocked(\
                                                                    [dist1,None,dist3],[oe,None,oe3],[positionPref1,None,positionPref3],[pref1Full,pref2Full,pref3Full])
                                                                if newOutEdge == -1:
                                                                    break
                                                                if newOutEdge == 1:
                                                                    positionPref1,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe,positionPref1,case=1)
                                                                    countPref1 += 1
                                                                elif newOutEdge == 3:
                                                                    positionPref3,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe3,positionPref3,case=1)
                                                                    countPref3 += 1
                                                            else:
                                                                positionPref3.append(position3[index3])
                                                                countPref3 += 1
                                                        else:
                                                            positionPref3 = self._divergent_add_RBCs_to_positionPref(oe3,position3,positionPref3,i)
                                                            countPref3 += 1
                                                    else: #pref2 is completely full, pref3 as well or does not exist
                                                        positionPref1,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe,positionPref1,case=2)
                                                        if boolFull == 1:
                                                            pref1Full = 1
                                                        else:
                                                            countPref1 += 1 
                                            else:
                                                positionPref1.append(position1[index1])
                                                countPref1 += 1
                                        else:
                                            positionPref1 = self._divergent_add_RBCs_to_positionPref(oe,position1,positionPref1,i)
                                            countPref1 += 1
                                    elif posNoBifEventsPref2 > countPref2 and pref2Full == 0: #outEdge 1 full: 2 or 3 have to be used
                                        if positionPref2 != []:
                                            dist2 = positionPref2[-1]-position2[index2] if oe2['sign'] == 1.0 \
                                                else position2[index2]-positionPref2[-1]
                                            if dist2 < oe2['minDist']:
                                                if boolTrifurcation and posNoBifEventsPref3 > countPref3 and pref3Full == 0:
                                                    if positionPref3 != []:
                                                        dist3 = positionPref3[-1]-position3[index3] if oe3['sign'] == 1.0 \
                                                            else position3[index3]-positionPref3[-1]
                                                        if dist3 < oe3['minDist']:
                                                            newOutEdge,[pref1Full,pref2Full,pref3Full] = self._CapDiv_compute_timeBlocked(\
                                                                [None,dist2,dist3],[None,oe2,oe3],[None,positionPref2,positionPref3],\
                                                                [pref1Full,pref2Full,pref3Full])
                                                            if newOutEdge == -1:
                                                                break
                                                            if newOutEdge == 2:
                                                                positionPref2,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe2,positionPref2,case=1)
                                                                countPref2 += 1
                                                            elif newOutEdge == 3:
                                                                positionPref3,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe3,positionPref3,case=1)
                                                                countPref3 += 1
                                                        else:
                                                            positionPref3.append(position3[index3])
                                                            countPref3 += 1
                                                    else:
                                                        positionPref3 = self._divergent_add_RBCs_to_positionPref(oe3,position3,positionPref3,i)
                                                        countPref3 += 1
                                                else:
                                                    positionPref2,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe2,positionPref2,case=2)
                                                    if boolFull == 1:
                                                        pref2Full = 1
                                                    else:
                                                        countPref2 += 1
                                            else:
                                                positionPref2.append(position2[index2])
                                                countPref2 += 1
                                        else:
                                            positionPref2 = self._divergent_add_RBCs_to_positionPref(oe2,position2,positionPref2,i)
                                            countPref2 += 1
                                    else: #outEdge 1 and outEdge 2 are full: third outEdge?
                                        if boolTrifurcation and posNoBifEventsPref3 > countPref3 and pref3Full == 0:
                                            if positionPref3 != []:
                                                dist3 = positionPref3[-1]-position3[index3] if oe3['sign'] == 1.0 \
                                                    else position3[index3]-positionPref3[-1]
                                                if dist3 < oe3['minDist']:
                                                    positionPref3,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe3,positionPref3,case=2)
                                                    if boolFull == 1:
                                                        pref3Full = 1
                                                    else:
                                                        countPref3 += 1 
                                                else:
                                                    positionPref3.append(position3[index3])
                                                    countPref3 += 1
                                            else:
                                                positionPref3 = self._divergent_add_RBCs_to_positionPref(oe3,position3,positionPref3,i)
                                                countPref3 += 1
                                        else:
                                            break
                                overshootsNo = countPref2+countPref1+countPref3
                            oe['countRBCs'] += len(positionPref1)
                            oe2['countRBCs'] += len(positionPref2)
                            self._move_RBCs(oe,positionPref1[::-1],signConsidered=1)
                            self._move_RBCs(oe2,positionPref2[::-1],signConsidered=1)
                            if len(outEdges) >2:
                                oe3['countRBCs'] += len(positionPref3)
                                self._move_RBCs(oe3,positionPref3[::-1],signConsidered=1)
                            self._remove_RBCs(e,overshootsNo)
                        noStuckRBCs = len(bifRBCsIndex)-overshootsNo
                        self._push_stuckRBCs_back(e,noStuckRBCs)
                    #-------------------------------------------------------------------------------------------
                    elif vertex['vType'] == 4: #CONVERGENT vertex
                        #print('CONVERGENT')
                        boolTrifurcation = 0
                        bifRBCsIndex1 = bifRBCsIndex
                        noBifEvents1 = noBifEvents
                        outE = vertex['outflowE'][0]
                        oe = G.es[outE]
                        inflowEdges = vertex['inflowE'][:]
                        inflowEdges.remove(ei)
                        inE2 = inflowEdges[0]
                        e2 = G.es[inE2]
                        sign2 = e2['sign']
                        if convEdges[inE2] == 0:
                            convEdges[inE2] = 1
                            bifRBCsIndex2 = self._initial_propagate_and_compute_bifRBCsIndex(e2,sign2)
                            noBifEvents2 = len(bifRBCsIndex2)
                        else:
                            noBifEvents2 = 0
                            bifRBCsIndex2 = []
                        #Check if there is a third inEdge
                        if len(inflowEdges) > 1:
                            boolTrifurcation = 1
                            inE3 = inflowEdges[1]
                            e3 = G.es[inE3]
                            sign3 = e3['sign']
                            if convEdges[inE3] == 0:
                                convEdges[inE3] = 1
                                bifRBCsIndex3 = self._initial_propagate_and_compute_bifRBCsIndex(e3,sign3)
                                noBifEvents3 = len(bifRBCsIndex3)
                            else:
                                bifRBCsIndex3 = []
                                noBifEvents3 = 0
                        else:
                            bifRBCsIndex3 = []
                            noBifEvents3 = 0
                            boolTrifurcation = 0
                        posNoBifEvents = self._calculate_possible_number_of_BifEvents(oe)
                        if posNoBifEvents > 0:
                            overshootTime1 = self._compute_overshootTime(e,bifRBCsIndex1,sign)
                            dummy1 = np.ones(noBifEvents1,dtype=np.int)
                            if noBifEvents2 > 0:
                                overshootTime2 = self._compute_overshootTime(e2,bifRBCsIndex2,sign2)
                                dummy2 = 2*np.ones(noBifEvents2,dtype=np.int)
                            else:
                                overshootDist2 = []
                                overshootTime2 = []
                                dummy2 = np.array([],dtype=np.int)
                            if boolTrifurcation and noBifEvents3 > 0:
                                overshootTime3 = self._compute_overshootTime(e3,bifRBCsIndex3,sign3)
                                dummy3 = 3*np.ones(noBifEvents3,dtype=np.int)
                            else:
                                overshootDist3 = []
                                overshootTime3 = []
                                dummy3 = np.array([],dtype=np.int)
                            overshootTimes = zip(np.concatenate([overshootTime1,overshootTime2,overshootTime3]),np.concatenate([dummy1,dummy2,dummy3]))
                            overshootTimes.sort()
                            overshootsNo = np.min([len(overshootTimes),posNoBifEvents])
                            overshootTime,inEdge = zip(*overshootTimes)
                            overshootTime = np.array(overshootTime)
                            count1 = inEdge.count(1)
                            count2 = inEdge.count(2)
                            count3 = inEdge.count(3)
                            #position starts with least overshooting RBC and ends with highest overshooting RBC
                            position = self._compute_unconstrained_RBC_positions(oe,overshootTime,signConsidered=0)
                            position = self._check_overshootingNewVessel_and_overtakingRBCsInNewVessel(oe,position)
                            try:
                                indexOut = (position < 0).tolist().index(False)
                            except:
                                indexOut = len(position)
                            inEdgesOut = inEdge[0:indexOut]
                            count1 += -1*inEdgesOut.count(1)
                            count2 += -1*inEdgesOut.count(2)
                            count3 += -1*inEdgesOut.count(3)
                            position = position[indexOut::]
                            inEdge = inEdge[indexOut::]
                            oe['countRBCs']+=len(position)
                            self._move_RBCs(oe,position,signConsidered=0)
                            if count1 > 0:
                                self._remove_RBCs(e,count1)
                            if noBifEvents2 > 0 and count2 > 0:
                                self._remove_RBCs(e2,count2)
                            if boolTrifurcation:
                                if noBifEvents3 > 0 and count3 > 0:
                                    self._remove_RBCs(e3,count3)
                            overshootsNo = count1 + count2 + count3
                        else:
                            count1,count2,count3 = 0,0,0
                        noStuckRBCs1 = len(bifRBCsIndex1)-count1
                        self._push_stuckRBCs_back(e,noStuckRBCs1)
                        noStuckRBCs2 = len(bifRBCsIndex2)-count2
                        self._push_stuckRBCs_back(e2,noStuckRBCs2)
                        if boolTrifurcation:
                            noStuckRBCs3 = len(bifRBCsIndex3)-count3
                            self._push_stuckRBCs_back(e3,noStuckRBCs3)
                    #------------------------------------------------------------------------------------------
                    #if vertex is double connecting vertex
                    elif vertex['vType'] == 6:
                        #print('DOUBLE')
                        bifRBCsIndex1 = bifRBCsIndex
                        noBifEvents1 = noBifEvents
                        inflowEdges = vertex['inflowE'][:]
                        inflowEdges.remove(ei)
                        inE2 = inflowEdges[0]
                        e2 = G.es[inE2]
                        sign2 = e2['sign']
                        if convEdges[inE2] == 0:
                            convEdges[inE2] = 1
                            bifRBCsIndex2 = self._initial_propagate_and_compute_bifRBCsIndex(e2,sign2)
                            noBifEvents2 = len(bifRBCsIndex2)
                        else:
                            bifRBCsIndex2 = []
                            noBifEvents2 = 0
                        outEdges = vertex['outflowE']
                        if vertex['isCap']:
                            nonCap = 0
                            preferenceList = [x[1] for x in sorted(zip(np.array(G.es[outEdges]['flow'])/np.array(G.es[outEdges]['crosssection']), outEdges), reverse=True)]
                        else:
                            nonCap = 1
                            preferenceList = [x[1] for x in sorted(zip(G.es[outEdges]['flow'], outEdges), reverse=True)]
                            ratio1 = Physiol.phase_separation_effect(G.es[preferenceList[0]]['flow']/np.sum(G.es[outEdges]['flow']), \
                                G.es[preferenceList[0]]['diameter'],G.es[preferenceList[1]]['diameter'],e['diameter'],e['htd'])
                            #it can happen that ratio1 < 0.5 even if the flowRate is larger. in this case the preferenceList is changed
                            if ratio1 < 0.5:
                                ratio1 = 1.0 - ratio1
                                preferenceList = preferenceList[::-1]
                            ratio2 = 1.0 -  ratio1
                        oe = G.es[preferenceList[0]]
                        oe2 = G.es[preferenceList[1]]
                        posNoBifEventsPref = self._calculate_possible_number_of_BifEvents(oe)
                        posNoBifEventsPref2 = self._calculate_possible_number_of_BifEvents(oe2)
                        posNoBifEvents = int(posNoBifEventsPref+posNoBifEventsPref2)
                        overshootsNo = noBifEvents1 + noBifEvents2
                        if nonCap:
                            overshootsNo1,overshootsNo2,overshootsNo3 = \
                                self._nonCapDiv_compute_overshootNos_from_ratios(0,overshootsNo,ratio1,ratio2,0.)
                            overshootsNo1,overshootsNo2,overshootsNo3 = self._nonCapDiv_compare_overshootNos_to_posBifEvents(\
                                overshootsNo1,overshootsNo2,0,posNoBifEventsPref,posNoBifEventsPref2,0,overshootsNo)
                            overshootsNo = int(overshootsNo1 + overshootsNo2)
                            posNoBifEvents = overshootsNo
                        #If bifurcations are possible check how many overshoots there are at the inEdges
                        if posNoBifEvents > 0:
                            overshootTime1 = self._compute_overshootTime(e,bifRBCsIndex1,sign)
                            dummy1 = np.ones(noBifEvents1,dtype=np.int)
                            if noBifEvents2 > 0:
                                overshootTime2 = self._compute_overshootTime(e2,bifRBCsIndex2,sign2)
                                dummy2 = 2*np.ones(noBifEvents2,dtype=np.int)
                            else:
                                overshootDist2 = []
                                overshootTime2 = []
                                dummy2 = []
                            overshootTimes = zip(np.concatenate([overshootTime1,overshootTime2]),np.concatenate([dummy1,dummy2]))
                            overshootTimes.sort()
                            overshootsNo = np.min([len(overshootTimes),posNoBifEvents])
                            overshootTime,inEdge = zip(*overshootTimes)
                            overshootTime = np.array(overshootTime)
                            count1 = inEdge.count(1)
                            count2 = inEdge.count(2)
                            position1 = self._compute_unconstrained_RBC_positions(oe,overshootTime,signConsidered=1)
                            position2 = self._compute_unconstrained_RBC_positions(oe2,overshootTime,signConsidered=1)
                            if nonCap:
                                countNo1,countNo2 = 0,0
                                count1,count2 = 0,0
                                positionPref1,positionPref2 = [],[]
                                distributeOrderNextIteration = [1,2]
                                for i in xrange(overshootsNo):
                                    index = -1*(i+1)
                                    distributeOrder = distributeOrderNextIteration[:]
                                    boolRBCassigned = 0
                                    for j in distributeOrder:
                                        if j == 1:
                                            if countNo1 < overshootsNo1:
                                                positionPref1 = self._divergent_add_RBCs_to_positionPref(oe,position1,positionPref1,i)
                                                distributeOrderNextIteration = [2,1]
                                                boolRBCassigned = 1
                                                countNo1 += 1
                                                count1,count2 = self._doubleConnecting_update_inEdge_count(inEdge,index,count1,count2)
                                                break
                                        elif j == 2:
                                            if countNo2 < overshootsNo2:
                                                positionPref2 = self._divergent_add_RBCs_to_positionPref(oe2,position2,positionPref2,i)
                                                distributeOrderNextIteration = [1,2]
                                                boolRBCassigned = 1
                                                countNo2 += 1
                                                count1,count2 = self._doubleConnecting_update_inEdge_count(inEdge,index,count1,count2)
                                                break
                                    if not boolRBCassigned:
                                        print('BIGERROR all overshootRBCS should fit 2')
                                positionPref1 = self._nonCapDiv_push_RBCs_forward_to_fit(oe,positionPref1)
                                positionPref2 = self._nonCapDiv_push_RBCs_forward_to_fit(oe2,positionPref2)
                            else:
                                #It is tried if all RBCs fit into the prefered outEdge. The time of arrival at the RBCs is taken into account
                                #RBCs which would be too close together are put into the other edge
                                positionPref1,positionPref2 = [],[]
                                countPref1,countPref2 = 0,0 
                                pref1Full,pref2Full = 0,0
                                count1,count2 = 0,0
                                for i in xrange(overshootsNo):
                                    index = -1*(i+1)
                                    index1 = -1*(i+1) if oe['sign'] == 1.0 else i
                                    index2 = -1*(i+1) if oe2['sign'] == 1.0 else i
                                    if posNoBifEventsPref > countPref1 and pref1Full == 0:
                                        if positionPref1 != []:
                                            dist1 = positionPref1[-1]-position1[index1] if oe['sign'] == 1.0 \
                                                else position1[index1]-positionPref1[-1]
                                            if dist1 < oe['minDist']:
                                                if posNoBifEventsPref2 > countPref2 and pref2Full == 0:
                                                    if positionPref2 != []:
                                                        dist2 = positionPref2[-1]-position2[index2] if oe2['sign'] == 1.0 \
                                                            else position2[index2]-positionPref2[-1]
                                                        if dist2 < oe2['minDist']:
                                                            newOutEdge,[pref1Full,pref2Full,pref3FullDummy] = self._CapDiv_compute_timeBlocked(\
                                                                [dist1,dist2,None],[oe,oe2,None],[positionPref1,positionPref2,None],[pref1Full,pref2Full,None])
                                                            if newOutEdge == -1:
                                                                break
                                                            if newOutEdge == 1:
                                                                positionPref1,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe,positionPref1,case=1)
                                                                countPref1 += 1
                                                                count1,count2 = self._doubleConnecting_update_inEdge_count(inEdge,index,count1,count2)
                                                            elif newOutEdge == 2:
                                                                positionPref2,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe2,positionPref2,case=1)
                                                                countPref2 += 1
                                                                count1,count2 = self._doubleConnecting_update_inEdge_count(inEdge,index,count1,count2)
                                                        else:
                                                            positionPref2.append(position2[index2])
                                                            countPref2 += 1
                                                            count1,count2 = self._doubleConnecting_update_inEdge_count(inEdge,index,count1,count2)
                                                    else:
                                                        positionPref2 = self._divergent_add_RBCs_to_positionPref(oe2,position2,positionPref2,i)
                                                        countPref2 += 1
                                                        count1,count2 = self._doubleConnecting_update_inEdge_count(inEdge,index,count1,count2)
                                                else:
                                                    positionPref1,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe,positionPref1,case=2)
                                                    if boolFull == 1:
                                                        pref1Full = 1
                                                        break
                                                    else:
                                                        countPref1 += 1
                                                        count1,count2 = self._doubleConnecting_update_inEdge_count(inEdge,index,count1,count2)
                                            else:
                                                positionPref1.append(position1[index1])
                                                countPref1 += 1
                                                count1,count2 = self._doubleConnecting_update_inEdge_count(inEdge,index,count1,count2)
                                        else:
                                            positionPref1 = self._divergent_add_RBCs_to_positionPref(oe,position1,positionPref1,i)
                                            countPref1 += 1
                                            count1,count2 = self._doubleConnecting_update_inEdge_count(inEdge,index,count1,count2)
                                    elif posNoBifEventsPref2 > countPref2 and pref2Full == 0:
                                        if positionPref2 != []:
                                            dist2 = positionPref2[-1]-position2[index2] if oe2['sign'] == 1.0 \
                                                else position2[index2]-positionPref2[-1]
                                            if dist2 < oe2['minDist']:
                                                positionPref2,boolFull = self._CapDiv_add_blocked_RBC_to_positionPref(oe2,positionPref2,case=2)
                                                if boolFull == 1:
                                                    pref2Full = 1
                                                    break
                                                else:
                                                    countPref2 += 1
                                                    count1,count2 = self._doubleConnecting_update_inEdge_count(inEdge,index,count1,count2)
                                            else:
                                                positionPref2.append(position2[index2])
                                                countPref2 += 1
                                                count1,count2 = self._doubleConnecting_update_inEdge_count(inEdge,index,count1,count2)
                                        else:
                                            positionPref2 = self._divergent_add_RBCs_to_positionPref(oe2,position2,positionPref2,i)
                                            countPref2 += 1
                                            count1,count2 = self._doubleConnecting_update_inEdge_count(inEdge,index,count1,count2)
                                    else:
                                        break
                                overshootsNo = countPref2+countPref1
                            oe['countRBCs'] += len(positionPref1)
                            self._move_RBCs(oe,positionPref1[::-1],signConsidered=1)
                            oe2['countRBCs'] += len(positionPref2)
                            self._move_RBCs(oe2,positionPref2[::-1],signConsidered=1)
                            if count1 > 0:
                                self._remove_RBCs(e,count1)
                            if noBifEvents2 > 0 and count2 > 0:
                                self._remove_RBCs(e2,count2)
                        else:
                            countPref1,countPref2 = 0,0
                            count1,count2 = 0,0
                        noStuckRBCs1 = len(bifRBCsIndex1)-count1
                        self._push_stuckRBCs_back(e,noStuckRBCs1)
                        if noBifEvents2 > 0:
                            noStuckRBCs2 = len(bifRBCsIndex2)-count2
                            self._push_stuckRBCs_back(e2,noStuckRBCs2)
            #-------------------------------------------------------------------------------------------
            if e['httBC'] is not None:
                if httBCDoneEdges[ei] == 0:
                    httBCDoneEdges[ei] = 1
                    boolHttEdge = 1
                    rRBC = self._insert_RBCs_at_boundaryEdges(e)
                    if len(rRBC) >= 1.:
                        if e['sign'] == 1:
                            e['rRBC'] = np.concatenate([rRBC[::-1], e['rRBC']])
                        else:
                            e['rRBC'] = np.concatenate([e['rRBC'], e['length']-rRBC])
                        vertexUpdate.append(e['target'])
                        vertexUpdate.append(e['source'])
                        edgeUpdate.append(ei)
            if noBifEvents > 0:
                if vertex['vType'] == 6 or vertex['vType'] == 4:
                    if e2['httBC'] is not None:
                        if httBCDoneEdges[inE2] == 0:
                            boolHttEdge2 = 1
                            httBCDoneEdges[inE2] = 1
                            rRBC2 = self._insert_RBCs_at_boundaryEdges(e2)
                            if len(rRBC2) >= 1.:
                                if e2['sign'] == 1:
                                    e2['rRBC'] = np.concatenate([rRBC2[::-1], e2['rRBC']])
                                else:
                                    e2['rRBC'] = np.concatenate([e2['rRBC'], e2['length']-rRBC2])
                                vertexUpdate.append(e2['target'])
                                vertexUpdate.append(e2['source'])
                                edgeUpdate.append(e2.index)
                    if vertex['vType'] == 4 and boolTrifurcation:
                        if e3['httBC'] is not None:
                            if httBCDoneEdges[inE3] == 0:
                                boolHttEdge3 = 1
                                httBCDoneEdges[inE3] = 1
                                rRBC3 = self._insert_RBCs_at_boundaryEdges(e3)
                                if len(rRBC3) >= 1.:
                                    if e3['sign'] == 1:
                                        e3['rRBC'] = np.concatenate([rRBC3[::-1], e3['rRBC']])
                                    else:
                                        e3['rRBC'] = np.concatenate([e3['rRBC'], e3['length']-rRBC3])
                                    vertexUpdate.append(e3['target'])
                                    vertexUpdate.append(e3['source'])
                                    edgeUpdate.append(e3.index)
            #TODO those are checks which are not necessary
            if noBifEvents != 0 or boolHttEdge == 1 or boolHttEdge2 == 1 or boolHttEdge3 == 1:
                nRBCSumAfter = 0
                for i in edgesInvolved:
                    nRBCList[i] = len(G.es[i]['rRBC'])
                    nRBCSumAfter += nRBCList[i]
                if nRBCSumBefore != nRBCSumAfter:
                    if vertex['vType'] == 2:
                        if nRBCSumAfter + noBifEvents != nRBCSumBefore:
                            print('BIGERROR RBC CONSERVATION at outlet')
                    else:
                        if boolHttEdge == 1 or boolHttEdge2 == 1 or boolHttEdge3 == 1:
                            rbcsAdded = 0
                            if boolHttEdge == 1:
                                rbcsAdded += len(rRBC)
                            if boolHttEdge2 == 1:
                                rbcsAdded += len(rRBC2)
                            if boolHttEdge3 == 1:
                                rbcsAdded += len(rRBC3)
                            if nRBCSumAfter - rbcsAdded != nRBCSumBefore:
                                    print('BIGERROR RBC CONSERVATION at inlet')
                        else:
                            print('BIGERROR RBC CONSERVATION somewhere')
                for edge in G.es[edgesInvolved]:
                    if len(edge['rRBC']) > 0:
                        if edge['rRBC'][0] < 0 - eps or edge['rRBC'][-1] > edge['length'] + eps:
                            print('BIGERROR BEGINNING END 2')
                    if True in ((edge['rRBC'][1::]-edge['rRBC'][:-1])<edge['minDist']-eps).tolist():
                        print('BIGERROR BEGINNING END 3')
                        print(edge['rRBC'][1::]-edge['rRBC'][:-1])
                        print(edge['minDist'])
                        print(((edge['rRBC'][1::]-edge['rRBC'][:-1])<edge['minDist']))
            if overshootsNo != 0:
                vertexUpdate.append(e['target'])
                vertexUpdate.append(e['source'])
                edgeUpdate = edgeUpdate+edgesInvolved
                if self._analyzeBifEvents:
                    if vertex['vType'] == 3 or vertex['vType'] == 5:
                        rbcMoved += overshootsNo
                    elif vertex['vType'] == 6:
                        rbcMoved += count1 + count2
                    elif vertex['vType'] == 4:
                        rbcMoved += count1 + count2
                        if len(inflowEdges) > 2:
                            if count3 > 0:
                                rbcMoved += count3
                if self._analyzeBifEvents:
                    if vertex['vType'] == 3 or vertex['vType'] == 5:
                        rbcsMovedPerEdge.append(overshootsNo)
                        edgesWithMovedRBCs.append(e.index)
                    elif vertex['vType'] == 6:
                        if count1 > 0:
                            rbcsMovedPerEdge.append(count1)
                            edgesWithMovedRBCs.append(e.index)
                        if count2 > 0:
                            edgesWithMovedRBCs.append(e2.index)
                            rbcsMovedPerEdge.append(count2)
                    elif vertex['vType'] == 4:
                        if count1 > 0:
                            rbcsMovedPerEdge.append(count1)
                            edgesWithMovedRBCs.append(e.index)
                        if count2 > 0:
                            edgesWithMovedRBCs.append(e2.index)
                            rbcsMovedPerEdge.append(count2)
                        if len(inflowEdges) > 2:
                            if count3 > 0:
                                rbcsMovedPerEdge.append(count3)
                                edgesWithMovedRBCs.append(e.index)

        #-------------------------------------------------------------------------------------------
        self._vertexUpdate = np.unique(vertexUpdate)
        edgeUpdate = np.unique(edgeUpdate)
        self._edgeUpdate = edgeUpdate.tolist()
        G.es['nRBC'] = [len(e['rRBC']) for e in G.es]
        if self._analyzeBifEvents:
            self._rbcsMovedPerEdge.append(rbcsMovedPerEdge)
            self._edgesWithMovedRBCs.append(edgesWithMovedRBCs)
            self._rbcMoveAll.append(rbcMoved)
        self._G = G
    #--------------------------------------------------------------------------
    #@profile
    def _initial_propagate_and_compute_bifRBCsIndex(self,e,sign):
        """ Calculates bifRBCsIndex
        INPUT: e: igraph edge where the RBCs are propagated and for which bifRBCsIndex
                  is computed
               sign: sign of edge of interest
         OUTPUT: bifRBCsIndex: compute bifRBCsIndex: list with the 
                   indices of the RBCs which overshoot
        """
        if len(e['rRBC']) > 0:
            e['rRBC'] = e['rRBC'] + e['v'] * self._dt * sign
            if sign == 1.0:
                if e['rRBC'][-1] > e['length']:
                    bifRBCsIndex = range((np.array(e['rRBC'])>e['length']).tolist().index(True),len(e['rRBC']))
                else:
                    bifRBCsIndex = []
            else:
                if e['rRBC'][0] < 0:
                    try:
                        bifRBCsIndex = range(0,(e['rRBC']<0.).tolist().index(False))
                    except:
                        bifRBCsIndex = range(len(e['rRBC']))
                else:
                    bifRBCsIndex = []
        else:
            bifRBCsIndex = []

        return bifRBCsIndex

    #--------------------------------------------------------------------------
    def _move_RBCs(self,oe,position,signConsidered=0):
        """ Puts RBCs into new outEdge
        INPUT: oe: igraph edge to which the RBCs should be added
               position: position of the RBCs in the new edge (should start with
                   the value closest to 0 and increase) 
               signConsidered: (default = 0) 
                   if = 0: the operation length - position[::-1] is performed (for sign=-1)
                   if = 1: otherwise only the operation position[::-1] is performed (for sign=-1)
         OUTPUT: updated edge property rRBC
        """
        if oe['sign'] == 1.0:
            oe['rRBC'] = np.concatenate([position, oe['rRBC']])
        else:
            if not signConsidered:
                position = oe['length'] - position[::-1]
            else:
                position = position[::-1]
            oe['rRBC'] = np.concatenate([oe['rRBC'],position])

    #--------------------------------------------------------------------------
    def _compute_overshootTime(self,e,posBifRBCsIndex,sign):
        """ Removes RBCs from current edge
        INPUT: e: igraph edge to which RBCs should be propagated
               posBifRBCsIndex: RBC indices of the possible bifurcations events
               sign: sign of the edge
         OUTPUT: overshootDist: distance which RBCs overshooted (0 --> max(overshootDist))
                 overshootTime: time which the RBCs overshoot (0 --> max(overshootTime))
        """
        #overshootsDist --> array with the distances which the RBCs overshoot, 
        #starts wiht the RBC which overshoots the least 
        overshootDist = e['rRBC'][posBifRBCsIndex]-e['length'] if sign == 1.0 \
            else (0.-e['rRBC'][posBifRBCsIndex])[::-1]
        #overshootTime --> time which every RBCs overshoots
        overshootTime = overshootDist/e['v']                   

        return overshootTime

    #--------------------------------------------------------------------------
    def _remove_RBCs(self,e,overshootsNo):
        """ Removes RBCs from current edge
        INPUT: e: igraph edge from which the RBCs should be removed
               overshootsNo: number of RBCs which overshoot
         OUTPUT: updated edge property 'rRBC'
        """
        #Remove RBCs from old Edge
        if e['sign'] == 1.0:
            e['rRBC'] = e['rRBC'][:-overshootsNo]
        else:
            e['rRBC'] = e['rRBC'][overshootsNo::]

    #--------------------------------------------------------------------------
    def _calculate_possible_number_of_BifEvents(self,oe):
        """ Removes RBCs from current edge
        INPUT: oe: igraph edge to which RBCs should be propagated
         OUTPUT: posNoBifEvents: possible number of overshoots by the 
                   by the constraints in the outEdge (integer)
        """

        if len(oe['rRBC']) > 0:
            distToFirst = oe['rRBC'][0] if oe['sign'] == 1.0 else oe['length']-oe['rRBC'][-1]
        else:
            distToFirst = oe['length']
        posNoBifEvents = int(np.floor(distToFirst/oe['minDist']))
        #TODO the computation of the possible number of RBCs is acuatlly not necessary. because the ones which do
        #do not fit will be pushed outside anyways, while the position is assigned. However it might be useful for
        #the speed of the computations, because it reduces the number of RBCs over which is has to be looped
        #Check how many RBCs are allowed by nMax --> limitation results from np.floor(length/minDist) 
        #and that RBCs are only 'half' in the vessel #TODO the nMax part should not be necessary.
        #however the tube hematocrit formulation should be adjusted to account for the half RBCs at the
        #in- and outlet
        if posNoBifEvents + len(oe['rRBC']) > oe['nMax']:
            posNoBifEvents = int(oe['nMax'] - len(oe['rRBC']))

        return posNoBifEvents

    #--------------------------------------------------------------------------
    def _check_overshootingNewVessel_and_overtakingRBCsInNewVessel(self,oe,position):
        """ Adjust the np.array position. based on the following constraints. (1) RBC 
        should not overshoot the whole vessel. (2) RBCs should not overtake/overlap with
        RBCs which are already present. If (1) or (2) happens, RBCs are pushed backwards.
        The position of following RBCs has to be readjusted.
        INPUT: oe: edge to which the RBC shall be added
               position: np array with the position RBCs would have in the new edge without
               constraints. The values go from 0 --> x for sign=1 and sign = -1. The sign 
               of the edges is considered, for the constraints for pushing the RBCs backwards
         OUTPUT: position: updated array with RBC positions (values go from 0 --> x 
                 for sign=1 and sign = -1)
        """
        if len(oe['rRBC']) > 0:
            if oe['sign'] == 1.0:
                if position[-1] > oe['rRBC'][0]-oe['minDist']:
                    position[-1] = oe['rRBC'][0]-oe['minDist']
            else:
                if oe['length']-position[-1] < oe['rRBC'][-1]+oe['minDist']:
                    position[-1] = oe['length']-(oe['rRBC'][-1]+oe['minDist'])
        else:
            #Check if the RBCs overshooted the vessel
            if position[-1] > oe['length']:
                position[-1] = oe['length']

        #Position of the following RBCs is changed, such that they do not overlap
        for i in xrange(-1,-1*len(position),-1):
            if position[i]-position[i-1] < oe['minDist'] or position[i-1] > position[i]:
                position[i-1] = position[i]-oe['minDist']

        return position
    #--------------------------------------------------------------------------
    def _push_stuckRBCs_back(self,e,noStuckRBCs):
        """ Push RBCs which can not be propagated back into their edge of origin.
        All RBCs which have been propagated in that edge are pushed backwards such
        that overlapping is avoided.
        INPUT: e: igraph edge to which RBCs should be pushed backwards
               noStuckRBCs: number of RBCs which could not be propagated and are
                   hence pushed backwards
         OUTPUT: updated edge property 'rRBC'
        """
        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
        if noStuckRBCs > 0: 
            sign = e['sign']
            #move stuck RBCs back into vessel
            for i in xrange(noStuckRBCs):
                index = -1*(i+1) if sign == 1.0 else i
                e['rRBC'][index] = e['length']-i*e['minDist'] if sign == 1.0 else 0+i*e['minDist']
            noRBCs = len(e['rRBC']) 
            #Recheck if the distance between the newly introduces RBCs is still big enough 
            if sign == 1.0: 
                for i in xrange(-1*noStuckRBCs,-1*noRBCs,-1):
                    if e['rRBC'][i] < e['rRBC'][i-1] or abs(e['rRBC'][i]-e['rRBC'][i-1]) < e['minDist']:
                        e['rRBC'][i-1] = e['rRBC'][i]-e['minDist']
                    else:
                        break
            else:
                for i in xrange(noStuckRBCs-1,noRBCs-1):
                    if e['rRBC'][i] > e['rRBC'][i+1] or abs(e['rRBC'][i]-e['rRBC'][i+1]) < e['minDist']:
                        e['rRBC'][i+1] = e['rRBC'][i]+e['minDist']
                    else:
                        break

    #--------------------------------------------------------------------------
    def _compare_noBifEvents_to_posNoBifEvents(self,posNoBifEvents,noBifEvents,bifRBCsIndex,sign):
        """ Compare possible number of bifurcation events with the number of bifurcation events 
        taking place.
        INPUT: posNoBifEvents: possible number of bifurcation events, based on constraints in outEdges
               noBifEvents: number of bifurcation events taking place
               bifRBCsIndex: list of RBC indices which are overshooting
               sign: sign of the edge in which the RBC are currently in
         OUTPUT: overshootsNo: number of bifurcation which are possible and take place
                 posBifRBCsIndex: adjusted list of RBC indices which are overshooting
        """
        if posNoBifEvents > noBifEvents:
            posBifRBCsIndex = bifRBCsIndex
            overshootsNo = noBifEvents
        elif posNoBifEvents == 0:
            posBifRBCsIndex = []
            overshootsNo = 0
        else:
            posBifRBCsIndex = bifRBCsIndex[-posNoBifEvents::] if sign == 1.0 \
              else bifRBCsIndex[:posNoBifEvents]
            overshootsNo = posNoBifEvents

        return overshootsNo,posBifRBCsIndex

    #----------------------------------------------------------------------------------------------------
    def _nonCapDiv_compute_overshootNos_from_ratios(self,boolTrifurcation,overshootsNo,ratio1,ratio2,ratio3):
        """ Compute overshootsNo for the outlfow edge of non capillary divergent bifurcations 
        INPUT: boolTrifuration: bool if we are looking at a bifurcation with 3 outflows
               overshootsNo: number of RBCs overshooting in total
               ratio1: desired RBC ratio for the preferred outEdge
               ratio2: desired RBC ratio for the second preferred outEdge
               ratio3: desired RBC ratio for the third preferred outEdge
        OUTPUT: overshootsNo1, overshootsNo2, overshootsNo3:
                 overshootNos for the outflow edges (preferred outEdge is No1)
        """

        #Optimization Functions
        if overshootsNo > 0:
            def errorDistributeRBCs_ratio1(n1):
                return n1/float(overshootsNo)-ratio1
            def errorDistributeRBCs_ratio2(n2):
                return n2/float(overshootsNo)-ratio2
            def errorDistributeRBCs_ratio1_and_ratio2(n12):
                return [n12[0]/float(overshootsNo)-ratio1,n12[1]/float(overshootsNo)-ratio2]

            if not boolTrifurcation:
                if ratio1 != 0 and overshootsNo != 0:
                    resultMinimizeError = root(errorDistributeRBCs_ratio1,np.ceil(ratio1 * overshootsNo))
                    overshootsNo1 = int(np.round(resultMinimizeError['x']))
                else:
                    #TODO this line might not be necessary because ratio1 should never be equal to 0 and 
                    #overshootsNo != 0 is checked before --> rethink and doubleCheck
                    overshootsNo1 = 0
                overshootsNo2 = overshootsNo - overshootsNo1
                overshootsNo3 = 0
            else:
                if ratio1 != 0 and ratio2 != 0 and overshootsNo != 0:
                    resultMinimizeError = root(errorDistributeRBCs_ratio1_and_ratio2,\
                        [np.ceil(ratio1 * overshootsNo),np.ceil(ratio2 * overshootsNo)])
                    overshootsNo1 = int(np.round(resultMinimizeError['x'][0]))
                    overshootsNo2 = int(np.round(resultMinimizeError['x'][1]))
                elif ratio1 != 0 and overshootsNo != 0:
                    #TODO is ratio2 ever = 0?
                    resultMinimizeError = root(errorDistributeRBCs_ratio1,np.ceil(ratio1 * overshootsNo))
                    overshootsNo1 = int(np.round(resultMinimizeError['x']))
                    overshootsNo2 = 0
                elif ratio2 != 0 and overshootsNo != 0:
                    #TODO is ratio1 ever = 0?
                    resultMinimizeError = root(errorDistributeRBCs_ratio2,np.ceil(ratio2 * overshootsNo))
                    overshootsNo2 = int(np.round(resultMinimizeError['x']))
                    overshootsNo1 = 0
                overshootsNo3 = overshootsNo - overshootsNo1 - overshootsNo2
        else:
            overshootsNo1 = 0
            overshootsNo2 = 0
            overshootsNo3 = 0

        return overshootsNo1, overshootsNo2, overshootsNo3

    #----------------------------------------------------------------------------------------------------
    def _nonCapDiv_compare_overshootNos_to_posBifEvents(self,overshootsNo1,overshootsNo2,overshootsNo3, \
        posNoBifEventsPref,posNoBifEventsPref2,posNoBifEventsPref3,overshootsNo):
        """ Compare computed overshootNos to possible number of bifurcation events
        INPUT: overshootsNo1, overshootsNo2, overshootsNo3:
                 overshootNos for the outflow edges which are desired (preferred outEdge is No1)
               posNoBifEventsPref, posNoBifEventsPref2,posNoBifEventsPref3:
                 possible number of bifurcation Events for the available out edges
        OUTPUT: overshootsNo1, overshootsNo2, overshootsNo3:
                 overshootNos for the outflow edges which are infact possible (preferred outEdge is No1)
        """
        if overshootsNo1 > posNoBifEventsPref:
            overshootsNo2 += overshootsNo1 - posNoBifEventsPref
            overshootsNo1 = posNoBifEventsPref
        if overshootsNo2 > posNoBifEventsPref2:
            #possible bifurcation event > currentNewRBCs + additional RBCs from edge 2
            if posNoBifEventsPref > overshootsNo1 +  (overshootsNo2 - posNoBifEventsPref2):
                overshootsNo1 += overshootsNo2 - posNoBifEventsPref2
            else:
                overshootsNo1 = posNoBifEventsPref
                if posNoBifEventsPref3 > overshootsNo - (posNoBifEventsPref + posNoBifEventsPref2):
                    overshootsNo3 = overshootsNo - (posNoBifEventsPref + posNoBifEventsPref2)
                else:
                    overshootsNo3 = posNoBifEventsPref3
            overshootsNo2 = posNoBifEventsPref2
        if overshootsNo3 > posNoBifEventsPref3:
            #possible bifurcation event > currentNewRBCs + additional RBCs from edge 2
            if posNoBifEventsPref > overshootsNo1 +  (overshootsNo3 - posNoBifEventsPref3):
                overshootsNo1 += overshootsNo3 - posNoBifEventsPref3
            else:
                overshootsNo1 = posNoBifEventsPref
                if posNoBifEventsPref2 > overshootsNo - (posNoBifEventsPref3 + posNoBifEventsPref):
                    overshootsNo2 = overshootsNo - (posNoBifEventsPref3 + posNoBifEventsPref)
                else:
                    overshootsNo2 = posNoBifEventsPref2
            overshootsNo3 = posNoBifEventsPref3

        return overshootsNo1, overshootsNo2, overshootsNo3

    #----------------------------------------------------------------------------------------------------
    def _compute_unconstrained_RBC_positions(self,oe,overshootTime,signConsidered=0):
        """ Compute the unconstrained position of RBCs in possible outEdge. 
        INPUT: oe: possible outEdge
               overshootTime: array of time values by how many ms the different RBCs overshooted
               signConsidered: bool, 0: same position for 1 and -1 edges; 1: different computation for sign=-1
        OUTPUT: position: unconstrained position of RBCs in possible outEdge
        """
        if not signConsidered:
            position = overshootTime*oe['v']
        else:
            if oe['sign'] == 1.0:
                position = overshootTime*oe['v']
            else:
                position = oe['length']-overshootTime[::-1]*oe['v']

        return position
    #----------------------------------------------------------------------------------------------------
    def _divergent_add_RBCs_to_positionPref(self,oe,position,positionPref,i):
        """ Add one RBC to the positionPref for non capillary divergent bifurcations (vessel overshooting and 
        overlapping old RBCs and overlapping of propagated RBCs is considered)
        INPUT: oe: outEdge in which the RBCs should be placed
               position: unconstrained position of RBCs in the new outEdge
               positionPref: position of RBCs in outEdge
               i: index of the RBC which is currently under investigation
        OUTPUT: positionPref: position of RBCs in outEdge + 1 RBC
        """
        eps = self._eps
        index = -1*(i+1) if oe['sign'] == 1.0 else i

        if positionPref == []:
            if len(oe['rRBC']) > 0:
                if oe['sign'] == 1:
                    if position[index] > oe['rRBC'][0]-oe['minDist']:
                        positionPref.append(oe['rRBC'][0]-oe['minDist'])
                    else:
                        positionPref.append(position[index])
                else:
                    if position[index] < oe['rRBC'][-1]+oe['minDist']:
                        positionPref.append(oe['rRBC'][-1]+oe['minDist'])
                    else:
                        positionPref.append(position[index])
            else:
                if oe['sign'] == 1:
                    if position[index] > oe['length']:
                        positionPref.append(oe['length'])
                    else:
                        positionPref.append(position[index])
                else:
                    if position[index] < 0:
                        positionPref.append(0)
                    else:
                        positionPref.append(position[index])
        else: #This case only occurs for nonCaps. In capillaries other outflows are tried first
            positionPref.append(position[index])
            if oe['sign'] == 1:
                if positionPref[-1] > positionPref[-2] or positionPref[-2]-positionPref[-1] < oe['minDist']-eps:
                    positionPref[-1] = positionPref[-2] - oe['minDist']
            else:
                if positionPref[-1] < positionPref[-2] or positionPref[-1]-positionPref[-2] < oe['minDist']-eps:
                    positionPref[-1] = positionPref[-2] + oe['minDist']

        return positionPref

    #--------------------------------------------------------------------------
    def _nonCapDiv_push_RBCs_forward_to_fit(self,oe,positionPref):
        """ Adjust positionPref for nonCapillary divergent bifurcations. Overshooting and
        overlapping has already been considered. However, RBCs might have been pushed outside of vessel.
        Those RBCs are now pushed forward, such that the possible number of bifurcation events for
        that edge is achieved.
        INPUT: oe: outEdge in which the RBCs should be placed
               positionPref: position of RBCs in new outEdge. Constrained by overshooting and
                overlapping 
        OUTPUT: positionPref: updated list of positionPref
        """
        eps = self._eps
        if positionPref != []:
            if oe['sign'] == 1:
                if positionPref[-1] < 0:
                    positionPref[-1] = 0.0
                    for i in xrange(-1,-1*(len(positionPref)),-1):
                        if positionPref[i-1]-positionPref[i] < oe['minDist'] - eps:
                            positionPref[i-1] = positionPref[i] + oe['minDist']
                        else:
                            break
            else:
                if positionPref[-1] > oe['length']:
                    positionPref[-1] = oe['length']
                    for i in xrange(-1,-1*(len(positionPref)),-1):
                        if positionPref[i]-positionPref[i-1] < oe['minDist'] - eps:
                            positionPref[i-1] = positionPref[i] - oe['minDist']
                        else:
                            break

        return positionPref

    #----------------------------------------------------------------------------------------------------
    def _CapDiv_compute_timeBlocked(self,dists,oes,positionPrefs,prefsFull):
        """  
        If the RBC has to wait at all possible outlfow Edges, the time the RBC
        is blocked iscomputed. The RBC will be added to the outEdge where
        it is blocked the least amount of time. 
        If less than 3 outedges are available. The list values for the outflow 
        edge which is is not available have to be set to None, 
        e.g only edge 1 & 3 --> dists = [dist1,None,dist3]
        INPUT: dists: [dist1,dist2,dist3] distance between the last RBC assigned 
                    to the outEdge and the one under investigation
               oes: [oe,oe2,oe3] list of outEdges in which the RBC can proceed
               positionPrefs: [positionPref1,positionPref2,positionPref3] list
                    of RBC positions, which have already been propagated at the
                    current bifurcation.
        OUTPUT: newOutEdge: index of the new outEdge (1,2 or 3), -1 if all full
        """
        assert len(dists) == len(oes) == len(positionPrefs) == len(prefsFull)

        eps = self._eps
        num = len(dists)
        timesBlocked = np.array([np.NaN]*num)

        # update the timesBlocked
        for j in range(num):
            if positionPrefs[j] == None:
                continue

            space = positionPrefs[j][-1]
            if oes[j]['sign'] != 1.0:
                space = oes[j]['length']-space
            if space - oes[j]['minDist'] >= eps:
                timesBlocked[j] = (oes[j]['minDist']-dists[j])/oes[j]['v']
            else:
                prefsFull[j] = 1

        try:
            return np.nanargmin(timesBlocked) + 1,prefsFull
        except ValueError:
            return -1,prefsFull

    #----------------------------------------------------------------------------------------------------
    def _CapDiv_add_blocked_RBC_to_positionPref(self,oe,positionPref,case=1):
        """  
        INPUT: oe: outEdge to which the RBC is added
               positionPref: position of RBCs in new outEdge. Constrained by overshooting and
                overlapping 
               case: case=1 (default): it has already been checked that the RBC still fits into the vessel 
                    (this is the case after the timeBlocked formulation). case=2 it still needs to be checked
                    if the RBC has been pushed outside
        OUTPUT: positionPref: updated list of positionPref (+1 RBC if it fits) 
                boolFull: always 0 if case = 1 (because not needed), 1 if edge is full (only for case=2)
        """
        if oe['sign'] == 1.0:
            positionPref.append(positionPref[-1]-oe['minDist'])
        else:
            positionPref.append(positionPref[-1]+oe['minDist'])

        boolFull = 0
        if case == 2:
            if oe['sign'] == 1.0:
                if positionPref[-1] < 0:
                    boolFull = 1
                    positionPref = positionPref[:-1]
            else:
                if positionPref[-1] > oe['length']:
                    boolFull = 1
                    positionPref = positionPref[:-1]

        return positionPref, boolFull
    #----------------------------------------------------------------------------------------------------
    def _doubleConnecting_update_inEdge_count(self,inEdge,index,count1,count2):
        """  
        INPUT: inEdge: list of the inEdge where the RBCs orginates froms
               index: index of the current RBC under investigation
               count1: counter for the number of RBCs orginating from inEdge1
               count2: counter for the number of RBCs orginating from inEdge2
        OUTPUT: count1: counter for the number of RBCs orginating from inEdge1 (updated)
                count2: counter for the number of RBCs orginating from inEdge2 (updated)

        """
        if inEdge[index] == 1:
            count1 += 1
        else:
            count2 += 1

        return count1,count2
    #----------------------------------------------------------------------------------------------------

    def _insert_RBCs_at_boundaryEdges(self,e):
        """
        Insert new RBCs for inlet edges.
        Uses logNormal distribution that satisfies in the mean
        the tube hematocrit boundary conditions.
        INPUT: e: inlet edge
        OUTPUT: rRBC: numpy array with the positions of the newly introduced RBCs
        """
        dt = self._dt
        rRBC = []
        lrbc = e['minDist']
        htt = e['httBC']
        length = e['length']
        nMaxNew = e['nMax']-len(e['rRBC'])

        if len(e['rRBC']) > 0:
            posFirst = e['rRBC'][0] if e['sign'] == 1.0 else e['length']-e['rRBC'][-1]
            e['posFirst_last'] = posFirst
            e['v_last'] = e['v']
            cum_length = posFirst
        else:
            cum_length = e['posFirst_last'] + e['v_last'] * dt
            posFirst = cum_length
            e['posFirst_last'] = posFirst
            if e['v'] > e['v_last']:
                e['v_last'] = e['v']

        while cum_length >= lrbc and nMaxNew > 0:
            if len(e['keep_rbcs']) != 0:
                if posFirst - e['keep_rbcs'][0] < 0:
                    break
                if posFirst - e['keep_rbcs'][0] > e['length']:
                    rRBC.append(e['length'])
                    posFirst = e['length']
                else:
                    rRBC.append(posFirst - e['keep_rbcs'][0])
                    posFirst = posFirst - e['keep_rbcs'][0]
                nMaxNew -= 1
                cum_length = posFirst
                e['keep_rbcs'] = []
                e['posFirst_last'] = posFirst
            else:
                number = np.exp(e['logNormal'][0]+e['logNormal'][1]*np.random.randn(1)[0])
                spacing = lrbc+lrbc*number
                if posFirst - spacing >= 0:
                    if posFirst - spacing > e['length']:
                        rRBC.append(e['length'])
                        posFirst = e['length']
                    else:
                        rRBC.append(posFirst - spacing)
                        posFirst = posFirst - spacing
                    nMaxNew += -1
                    cum_length = posFirst
                    e['posFirst_last'] = posFirst
                else:
                    e['keep_rbcs'] = [spacing]
                    if len(rRBC) == 0:
                        e['posFirst_last'] = posFirst
                    else:
                        e['posFirst_last'] = rRBC[-1]
                    break

        if len(e['keep_rbcs']) == 0:
            number = np.exp(e['logNormal'][0]+e['logNormal'][1]*np.random.randn(1)[0])
            spacing = lrbc+lrbc*number
            e['keep_rbcs'] = [spacing]

        return np.array(rRBC)

    #--------------------------------------------------------------------------
    #@profile
    def evolve(self, time, method, dtfix,**kwargs):
        """Solves the linear system A x = b using a direct or AMG solver.
        INPUT: time: The duration for which the flow should be evolved. In case of
	 	     Reset in plotPrms or samplePrms = False, time is the duration 
	 	     which is added
               method: Solution-method for solving the linear system. This can
                       be either 'direct' or 'iterative'
               dtfix: given timestep
               **kwargs
               precision: The accuracy to which the ls is to be solved. If not
                          supplied, machine accuracy will be used. (This only
                          applies to the iterative solver)
               plotPrms: Provides the parameters for plotting the RBC 
                         positions over time. List format with the following
                         content is expected: [start, stop, step, reset].
                         'reset' is a boolean which determines if the current 
                         RBC evolution should be added to the existing history
                         or started anew. In case of Reset=False, start and stop
			 are added to the already elapsed time.
               samplePrms: Provides the parameters for sampling, i.e. writing 
                           a series of data-snapshots to disk for later 
                           analysis. List format with the following content is
                           expected: [start, stop, step, reset]. 'reset' is a
                           boolean which determines if the data samples should
                           be added to the existing database or a new database
                           should be set up. In case of Reset=False, start and stop
                          are added to the already elapsed time.
               SampleDetailed:Boolean whether every step should be samplede(True) or
			      if the sampling is done by the given samplePrms(False)
         OUTPUT: None (files are written to disk)
        """
        G = self._G
        if len(G.es(nMax_eq=0)) > 0:
            sys.exit("BIGERROR nMax=0 exists --> check vessel lengths") 
        tPlot = self._tPlot 
        tSample = self._tSample 
        filenamelist = self._filenamelist
        self._dt = dtfix
        timelist = self._timelist
        timelistAvg = self._timelistAvg
        init = self._init

        SampleDetailed = False
        if 'SampleDetailed' in kwargs.keys():
            SampleDetailed = kwargs['SampleDetailed']

        doSampling, doPlotting = [False, False]

        if 'plotPrms' in kwargs.keys():
            pStart, pStop, pStep = kwargs['plotPrms']
            doPlotting = True
            if init == True:
                tPlot = 0.0
                filenamelist = []
                timelist = []
            else:
                tPlot = G['iterFinalPlot']
                pStart = G['iterFinalPlot']+pStart+pStep
                pStop = G['iterFinalPlot']+pStop

        if 'samplePrms' in kwargs.keys():
            sStart, sStop, sStep = kwargs['samplePrms']
            doSampling = True
            if init == True:
                self._tSample = 0.0
                self._sampledict = {}
                timelistAvg = []
            else:
                if 'iterFinalSample' not in G.attributes():
                    G['iterFinalSample'] = 0
                self._tSample = G['iterFinalSample']
                sStart = G['iterFinalSample']+sStart+sStep
                sStop = G['iterFinalSample']+sStop

        t1 = ttime.time()
        if init:
            self._t = 0.0
            BackUpTStart = 0.025*time
            BackUpT = 0.025*time
            BackUpCounter = 0
        else:
            if 'dtFinal' not in G.attributes():
                G['dtFinal'] = 0
            if 'BackUpCounter' not in G.attributes():
                G['BackUpCounter'] = 0
            if 'iterFinalSample' not in G.attributes():
                G['iterFinalSample'] = 0
            self._t = G['dtFinal']
            self._tSample = G['iterFinalSample']
            BackUpT = 0.025*time
            print('Simulation starts at')
            print(self._t)
            print('First BackUp should be done at')
            time = G['dtFinal']+time
            BackUpCounter = G['BackUpCounter']+1
            BackUpTStart = G['dtFinal']+BackUpT
            print(BackUpTStart)
            print('BackUp should be done every')
            print(BackUpT)

        #Convert 'pBC' ['mmHG'] to default Units
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC'] = v['pBC']*self._scaleToDef

        tSample = self._tSample
        start_timeTot = ttime.time()
        t = self._t
        iteration = 0
        while True:
            if t >= time:
                break
            iteration += 1
            start_time = ttime.time()
            self._update_eff_resistance_and_LS(self._vertexUpdate)
            print('Matrix updated')
            self._solve(method, **kwargs)
            print('Matrix solved')
            self._G.vs['pressure'] = self._x[:]
            print('Pressure copied')
            self._update_flow_and_velocity()
            print('Flow updated')
            self._update_flow_sign()
            print('Flow sign updated')
            self._verify_mass_balance()
            print('Mass balance verified updated')
            self._update_out_and_inflows_for_vertices()
            print('In and outflows updated')
            stdout.flush()
            #TODO plotting
            #if doPlotting and tPlot >= pStart and tPlot <= pStop:
            #    filename = 'iter_'+str(int(round(tPlot)))+'.vtp'
                #filename = 'iter_'+('%.3f' % t)+'.vtp'
            #    filenamelist.append(filename)
            #    timelist.append(tPlot)
            #    self._plot_rbc(filename)
            #    pStart = tPlot + pStep
                #self._sample()
                #filename = 'sample_'+str(int(round(tPlot)))+'.vtp'
                #self._sample_average()
            if SampleDetailed:
                print('sample detailed')
                stdout.flush()
                self._t = t
                self._tSample = tSample
                self._sample()
                filenameDetailed ='G_iteration_'+str(iteration)+'.pkl'
                #Convert deaultUnits to ['mmHG']
                #for 'pBC' and 'pressure'
                for v in G.vs:
                    if v['pBC'] != None:
                        v['pBC'] = v['pBC']/self._scaleToDef
                    v['pressure'] = v['pressure']/self._scaleToDef
                vgm.write_pkl(G,filenameDetailed)
                #Convert 'pBC' ['mmHG'] to default Units
                for v in G.vs:
                    if v['pBC'] != None:
                        v['pBC'] = v['pBC']*self._scaleToDef
                    v['pressure'] = v['pressure']*self._scaleToDef
            else:
                if doSampling and tSample >= sStart and tSample <= sStop:
                    print('DO sampling')
                    stdout.flush()
                    self._t = t
                    self._tSample = tSample
                    print('start sampling')
                    stdout.flush()
                    self._sample()
                    sStart = tSample + sStep
                    print('sampling DONE')
                    if t > BackUpTStart:
                        print('BackUp should be done')
                        print(BackUpCounter)
                        stdout.flush()
                        G['dtFinal'] = t
                        G['iterFinalSample'] = tSample
                        G['BackUpCounter'] = BackUpCounter
                        if self._analyzeBifEvents:
                            G['rbcsMovedPerEdge'] = self._rbcsMovedPerEdge
                            G['edgesMovedRBCs'] = self._edgesWithMovedRBCs
                            G['rbcMovedAll'] = self._rbcMoveAll
                        filename1 = 'sampledict_BackUp_'+str(BackUpCounter)+'.pkl'
                        filename2 = 'G_BackUp'+str(BackUpCounter)+'.pkl'
                        self._sample_average()
                        print(filename1)
                        print(filename2)
                        #Convert deaultUnits to 'pBC' ['mmHG']
                        for v in G.vs:
                            if v['pBC'] != None:
                                v['pBC'] = v['pBC']/self._scaleToDef
                            v['pressure'] = v['pressure']/self._scaleToDef
                        g_output.write_pkl(self._sampledict,filename1)
                        vgm.write_pkl(G,filename2)
                        self._sampledict = {}
                        self._sampledict['averagedCount'] = G['averagedCount']
                        #Convert 'pBC' ['mmHG'] to default Units
                        for v in G.vs:
                            if v['pBC'] != None:
                                v['pBC'] = v['pBC']*self._scaleToDef
                            v['pressure'] = v['pressure']*self._scaleToDef
                        BackUpCounter += 1
                        BackUpTStart += BackUpT
                        print('BackUp Done')
            print('START RBC propagate')
            stdout.flush()
            self._propagate_rbc()
            print('RBCs propagated')
            self._update_hematocrit(self._edgeUpdate)
            print('Hematocrit updated')
            tPlot = tPlot + self._dt
            self._tPlot = tPlot
            tSample = tSample + self._dt
            self._tSample = tSample
            t = t + self._dt
            log.info(t)
            print('TIME')
            print(t)
            print("Execution Time Loop:")
            print(ttime.time()-start_time, "seconds")
            print(' ')
            print(' ')
            stdout.write("\r%f" % tPlot)
            stdout.flush()
        stdout.write("\rDone. t=%f        \n" % tPlot)
        log.info("Time taken: %.2f" % (ttime.time()-t1))
        print("Execution Time:")
        print(ttime.time()-start_timeTot, "seconds")

        self._update_eff_resistance_and_LS(None)
        self._solve(method, **kwargs)
        self._G.vs['pressure'] = self._x[:]
        print('Pressure copied')
        self._update_flow_and_velocity()
        self._update_flow_sign()
        self._update_out_and_inflows_for_vertices()
        self._verify_mass_balance()
        print('Mass balance verified updated')
        self._t = t
        self._tSample = tSample
        stdout.flush()

        G['dtFinal'] = t
        if self._analyzeBifEvents:
            G['rbcsMovedPerEdge'] = self._rbcsMovedPerEdge
            G['edgesMovedRBCs'] = self._edgesWithMovedRBCs
            G['rbcMovedAll'] = self._rbcMoveAll
        #G['iterFinalPlot'] = tPlot
        G['iterFinalSample'] = tSample
        G['BackUpCounter'] = BackUpCounter
        filename1 = 'sampledict_BackUp_'+str(BackUpCounter)+'.pkl'
        filename2 = 'G_BackUp'+str(BackUpCounter)+'.pkl'
        #if doPlotting:
        #    filename =  'iter_'+str(int(round(tPlot+1)))+'.vtp'
        #    filenamelist.append(filename)
        #    timelist.append(tPlot)
        #    self._plot_rbc(filename)
        #    g_output.write_pvd_time_series('sequence.pvd', 
        #                                   filenamelist, timelist)
        if doSampling:
            self._sample()
            #Convert deaultUnits to 'pBC' ['mmHG']
            for v in G.vs:
                if v['pBC'] != None:
                    v['pBC'] = v['pBC']/self._scaleToDef
                v['pressure'] = v['pressure']/self._scaleToDef
            self._sample_average()
            g_output.write_pkl(self._sampledict, 'sampledict.pkl')
            g_output.write_pkl(self._sampledict,filename1)
        vgm.write_pkl(G, 'G_final.pkl')
        vgm.write_pkl(G,filename2)

    #--------------------------------------------------------------------------

    def _plot_rbc(self, filename, tortuous=False):
        """Plots the current RBC distribution to vtp format.
        INPUT: filename: The name of the output file. This should have a .vtp
                         extension in order to be recognized by Paraview.
               tortuous: Whether or not to trace the tortuous path of the 
                         vessels. If false, linear tubes are assumed.
        OUTPUT: None, file written to disk.
        """
        G = self._G
        pgraph = vascularGraph.VascularGraph(0)
        r = []
        if tortuous:
            for e in G.es:
                if len(e['rRBC']) == 0:
                    continue
                p = e['points']
                cumlength = np.cumsum([np.linalg.norm(p[i] - p[i+1]) 
                                       for i in xrange(len(p[:-1]))])
                for rRBC in e['rRBC']:
                    i = np.nonzero(cumlength > rRBC)[0][0]
                    r.append(p[i-1] + (p[i] - p[i-1]) * 
                             (rRBC - cumlength[i-1]) / 
                             (cumlength[i] - cumlength[i-1]))
        else:
            for e in G.es:
                #points = e['points']
                #nPoints = len(points)
                rsource = G.vs[e['source']]['r']
                dvec = G.vs[e['target']]['r'] - G.vs[e['source']]['r']
                length = e['length']
                for rRBC in e['rRBC']:
                    #index = int(round(npoints * rRBC / length))
                    r.append(rsource + dvec * rRBC/length)
                    
        if len(r) > 0:
            pgraph.add_vertices(len(r))
            pgraph.vs['r'] = r
            g_output.write_vtp(pgraph, filename, False)
        else:
	    print('Network is empty - no plotting')

    #--------------------------------------------------------------------------
    
    def _sample(self):
        """Takes a snapshot of relevant current data and adds it to the sample
        database.
        INPUT: None
        OUTPUT: None, data added to self._sampledict
        """
        sampledict = self._sampledict
        G = self._G
        invivo = self._invivo
        
        htt2htd = self._P.tube_to_discharge_hematocrit
        du = self._G['defaultUnits']
        scaleToDef = self._scaleToDef
        #Convert default units to ['mmHG']
        pressure = np.array([1/scaleToDef]*G.vcount())*G.vs['pressure']
        
        for eprop in ['flow', 'v', 'htt', 'htd','nRBC','effResistance']:
            if not eprop in sampledict.keys():
                sampledict[eprop] = []
            sampledict[eprop].append(G.es[eprop])
        for vprop in ['pressure']:
            if not vprop in sampledict.keys():
                sampledict[vprop] = []
            sampledict[vprop].append(pressure)
        if not 'time' in sampledict.keys():
            sampledict['time'] = []
        sampledict['time'].append(self._tSample)

    #--------------------------------------------------------------------------

    def _sample_average(self):
        """Averages the self._sampleDict data and writes it to disc.
        INPUT: sampleAvgFilename: Name of the sample average out-file.
        OUTPUT: None
        """
        sampledict = self._sampledict
        G = self._G
        if 'averagedCount' in sampledict.keys():
            avCount = sampledict['averagedCount']
        else:
            avCount = 0
        avCountNew = len(sampledict['time'])
        avCountE = np.array([avCount]*G.ecount())
        avCountNewE = np.array([avCountNew]*G.ecount())
        for eprop in ['flow', 'v', 'htt', 'htd','nRBC','effResistance']:
            if eprop+'_avg' in G.es.attribute_names():
                G.es[eprop + '_avg'] = (avCountE*G.es[eprop+'_avg']+ \
                    avCountNewE*np.average(sampledict[eprop], axis=0))/(avCountE+avCountNewE)
            else:
                G.es[eprop + '_avg'] = np.average(sampledict[eprop], axis=0)
            #if not [eprop + '_avg'] in sampledict.keys():
            #    sampledict[eprop + '_avg'] = []
            sampledict[eprop + '_avg'] = G.es[eprop + '_avg']
        avCountV = np.array([avCount]*G.vcount())
        avCountNewV = np.array([avCountNew]*G.vcount())
        for vprop in ['pressure']:
            if vprop+'_avg' in G.vs.attribute_names():
                G.vs[vprop + '_avg'] = (avCountV*G.vs[vprop+'_avg']+ \
                    avCountNewV*np.average(sampledict[vprop], axis=0))/(avCountV+avCountNewV)
            else:
                G.vs[vprop + '_avg'] = np.average(sampledict[vprop], axis=0)
            #if not [vprop + '_avg'] in sampledict.keys():
            #    sampledict[vprop + '_avg'] = []
            sampledict[vprop + '_avg'] = G.vs[vprop + '_avg']
        sampledict['averagedCount'] = avCount + avCountNew
        G['averagedCount'] = avCount + avCountNew


    #--------------------------------------------------------------------------
    #@profile
    def _solve(self, method, **kwargs):
        """Solves the linear system A x = b using a direct or AMG solver.
        INPUT: method: This can be either 'direct' or 'iterative'
               **kwargs
               precision: The accuracy to which the ls is to be solved. If not
                          supplied, machine accuracy will be used. (This only
                          applies to the iterative solver)
        OUTPUT: None, self._x is updated.
        """
        A = self._A.tocsr()
        if method == 'direct':
            linalg.use_solver(useUmfpack=True)
            x = linalg.spsolve(A, self._b)
        elif method == 'iterative':
            if kwargs.has_key('precision'):
                eps = kwargs['precision']
            else:
                eps = finfo(float).eps
            #AA = ruge_stuben_solver(A)
            AA = smoothed_aggregation_solver(A, max_levels=10, max_coarse=500)
            #PC = AA.aspreconditioner(cycle='V')
            #x,info = linalg.cg(A, self._b, tol=eps, maxiter=30, M=PC)
            #(x,flag) = pyamg.krylov.fgmres(A,self._b, maxiter=30, tol=eps)
            #x = abs(AA.solve(self._b, tol=self._eps/10000000000000000000, accel='cg')) # abs required, as (small) negative pressures may arise
            x = abs(AA.solve(self._b, tol=self._eps/10000000, accel='cg')) # abs required, as (small) negative pressures may arise
        elif method == 'iterative2':
         # Set linear solver
             ml = rootnode_solver(A, smooth=('energy', {'degree':2}), strength='evolution' )
             M = ml.aspreconditioner(cycle='V')
             # Solve pressure system
             #counter = gmres_counter()
             #x,info = gmres(A, self._b, tol=self._eps/10000, maxiter=200, M=M,callback=counter)
             x,info = gmres(A, self._b, tol=self._eps/10000, maxiter=200, M=M)
             if info != 0:
                 print('SOLVEERROR in Solving the Matrix')
                 print(info)
             test = A * x
             res = np.array(test)-np.array(self._b)
        self._x = x
        ##self._res=res

    #--------------------------------------------------------------------------

    def _verify_mass_balance(self):
        """Computes the mass balance, i.e. sum of flows at each node and adds
        the result as a vertex property 'flowSum'.
        INPUT: None
        OUTPUT: None (result added as vertex property)
        """
        G = self._G
        G.vs['flowSum'] = [sum([G.es[e]['flow'] * np.sign(G.vs[v]['pressure'] -
                                                    G.vs[n]['pressure'])
                               for e, n in zip(G.adjacent(v), G.neighbors(v))])
                           for v in xrange(G.vcount())]
        for i in xrange(G.vcount()):
            if G.vs[i]['flowSum'] > 5e-4 and i not in G['av'] and i not in G['vv']:
                print('')
                print(i)
                print(G.vs['flowSum'][i])
                print('FLOWERROR')
                for j in G.adjacent(i):
                    print(G.es['flow'][j])

    #--------------------------------------------------------------------------

    def _verify_rbc_balance(self):
        """Computes the rbc balance, i.e. sum of rbc flows at each node and
        adds the result as a vertex property 'rbcFlowSum'.
        INPUT: None
        OUTPUT: None (result added as vertex property)
        """
        G = self._G
        vf = self._P.velocity_factor
        invivo = self._invivo
        lrbc = self._P.effective_rbc_length
        tubeHt = [0.0 if e['tubeHt'] is None else e['tubeHt'] for e in G.es]
        G.vs['rbcFlowSum'] = [sum([4.0 * G.es[e]['flow'] * vf(G.es[e]['diameter'],invivo) * tubeHt[e] /
                                   np.pi / G.es[e]['diameter']**2 / lrbc(G.es[e]['diameter']) *
                                   np.sign(G.vs[v]['pressure'] - G.vs[n]['pressure'])
                                   for e, n in zip(G.adjacent(v), G.neighbors(v))])
                              for v in xrange(G.vcount())]

    #--------------------------------------------------------------------------

    def _verify_p_consistency(self):
        """Checks for local pressure maxima at non-pBC vertices.
        INPUT: None.
        OUTPUT: A list of local pressure maxima vertices and the maximum 
                pressure difference to their respective neighbors."""
        G = self._G
        localMaxima = []
        #Convert 'pBC' ['mmHG'] to default Units
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC'] = v['pBC']*self._scaleToDef

        for i, v in enumerate(G.vs):
            if v['pBC'] is None:
                pdiff = [v['pressure'] - n['pressure']
                         for n in G.vs[G.neighbors(i)]]
                if min(pdiff) > 0:
                    localMaxima.append((i, max(pdiff)))         
        #Convert defaultUnits to 'pBC' ['mmHG']
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC'] = v['pBC']/self._scaleToDef

        return localMaxima

    #--------------------------------------------------------------------------
    
    def _residual_norm(self):
        """Computes the norm of the current residual.
        """
        return np.linalg.norm(self._A * self._x - self._b)
                

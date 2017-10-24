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

from copy import deepcopy
from pyamg import smoothed_aggregation_solver, rootnode_solver, util
import pyamg
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

__all__ = ['LinearSystemHtdTotFixedDTTrackBox']
log = vgm.LogDispatcher.create_logger(__name__)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class LinearSystemHtdTotFixedDTTrackBox(object):
    """Implements and extends the discrete red blood cell transport as proposed
    by Obrist and coworkers (2010).
    """
    #@profile
    def __init__(self, G, invivo=True,dThreshold=10.0, init=True,**kwargs):
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
               ht0: The initial hematocrit in the capillary bed. If G already 
                    contains the relevant properties from an earlier simulation
                    (i.e. rRBC edge property), ht0 can be set to 'current'.
                    This will make use of the current RBC distribution as the
                    initial hematocrit
               hd0: The initial hematocrit in the capillary bed is calculated by the given
                    initial discharge hematocrit. If G already contains the relevant 
                    properties from an earlier simulation (i.e. rRBC edge property), hd0 
                    can be set to 'current'. This will make use of the current RBC 
                    distribution as the initial hematocrit
               plasmaViscosity: The dynamic plasma viscosity. If not provided,
                                a literature value will be used.
               analyzeBifEvents: boolean if bifurcation events should be analyzed (Default = 0)
               innerDiam: boolean if inner or outer diamter of vessels is given in the graph 
                   (innerDiam = 1 --> inner diameter given) (Default = 0)
        OUTPUT: None, however the following items are created:
                self.A: Matrix A of the linear system, holding the conductance
                        information.
                self.b: Vector b of the linear system, holding the boundary
                        conditions.
        """
        self._G = G
        self._P = Physiology(G['defaultUnits'])
        self._dThreshold = dThreshold
        self._invivo=invivo
        nVertices = G.vcount()
        self._b = zeros(nVertices)
        self._x = zeros(nVertices)
        self._A = lil_matrix((nVertices,nVertices),dtype=float)
        self._eps = finfo(float).eps * 1e4
        self._tPlot = 0.0
        self._tSample = 0.0
        self._filenamelist = []
        self._timelist = []
        #self._filenamelistAvg = []
	self._timelistAvg = []
        self._sampledict = {} 
	#self._transitTimeDict = {}
	self._init=init
        self._scaleToDef=vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])
        self._dtFix=0.0
        self._vertexUpdate=None
        self._edgeUpdate=None
        self._RBCdict={}
        self._RBCindexCurrent=0
        G.es['source']=[e.source for e in G.es]
        G.es['target']=[e.target for e in G.es]
        G.es['crosssection']=np.array([0.25*np.pi]*G.ecount())*np.array(G.es['diameter'])**2
        #Used because the Pries functions are onlt defined for vessels till 3micron
        G.es['diamCalcEff']=[i if i >= 3. else 3.0 for i in G.es['diameter'] ]
        G.es['keep_rbcs']=[[] for i in range(G.ecount())]
        G.es['RBCindex']=[None]*G.ecount()
        self._allEdgesBox, self._borderEdges, self._internalEdges = G.get_edges_in_boundingBox(xCoords=(7475,7600),yCoords=(7300,7500),zCoords=(1050,1225))
        #RBC velocity to small
        #self._allEdgesBox, self._borderEdges, self._internalEdges = G.get_edges_in_boundingBox(xCoords=(6910,7050),yCoords=(7135,7320),zCoords=(1225,1375))
        #diam < 7micron
        #self._allEdgesBox, self._borderEdges, self._internalEdges = G.get_edges_in_boundingBox(xCoords=(7055,7205),yCoords=(7675,7800),zCoords=(950,1150))
        G['allEdgesBox']=self._allEdgesBox
        G['internalEdges']=self._internalEdges
        G['borderEdges']=self._borderEdges
        adjacent=[]
        #self._spacing=[]
        for i in range(G.vcount()):
            adjacent.append(G.adjacent(i))
        G.vs['adjacent']=adjacent

        htd2htt=self._P.discharge_to_tube_hematocrit
        htt2htd = self._P.tube_to_discharge_hematocrit

        if kwargs.has_key('analyzeBifEvents'):
            self._analyzeBifEvents=kwargs['analyzeBifEvents']
        else:
            self._analyzeBifEvents = 0

        if kwargs.has_key('innerDiam'):
            self._innerDiam = kwargs['innerDiam']
        else:
            self._innerDiam = 0
        #TODO should be changed
        self._innerDiam=1

        # Assure that both pBC and rBC edge properties are present:
        for key in ['pBC', 'rBC']:
            if not G.vs[0].attributes().has_key(key):
                G.vs[0][key] = None

        if self._analyzeBifEvents:
            self._rbcsMovedPerEdge=[]
            self._edgesWithMovedRBCs=[]
            self._rbcMoveAll=[]
        else:
            if 'rbcMovedAll' in G.attributes():
                del(G['rbcMovedAll'])
            if 'rbcsMovedPerEdge' in G.attributes():
                del(G['rbcsMovedPerEdge'])
            if 'rbcMovedAll' in G.attributes():
                del(G['edgesMovedRBCs'])

        # Set initial pressure and flow to zero:
	if init:
            G.vs['pressure']=zeros(nVertices) 
            G.es['flow']=zeros(G.ecount())    
            G.vs['degree']=G.degree()
        print('Initial flow, presure, ... assigned')

        #Read sampledict (must be in folder, from where simulation is started)
        if not init:
            if 'averagedCount' not in G.attributes():
                self._sampledict['averagedCount']=0
            else:
                self._sampledict['averagedCount']=G['averagedCount']

        #Calculate total network Volume
        G['V']=0
        for e in G.es:
            G['V']=G['V']+e['crosssection']*e['length']
        print('Total network volume calculated')

        # Compute the edge-specific minimal RBC distance:
        vrbc = self._P.rbc_volume()
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
        adjacent=[np.array(G.incident(i)) for i in G.vs]
        G.vs['isCap']=[None]*G.vcount()
        self._interfaceVertices=[]
        for i in xrange(G.vcount()):
            #print(i)
            category=[]
            for j in adjacent[i]:
                #print(j)
                #print(G.es.attribute_names())
                if G.es[int(j)]['diameter'] < dThreshold:
                    category.append(1)
                else:
                    category.append(0)
            if category.count(1) == len(category):
                G.vs[i]['isCap']=True
            elif category.count(0) == len(category):
                G.vs[i]['isCap']=False
            else:
                self._interfaceVertices.append(i)
        print('End assign capillary and non capillary vertices')

        # Arterial-side inflow:
        if 'htdBC' in G.es.attribute_names():
           G.es['httBC']=[e['htdBC'] if e['htdBC'] == None else \
                self._P.discharge_to_tube_hematocrit(e['htdBC'],e['diameter'],invivo) for e in G.es()]
        if not 'httBC' in G.es.attribute_names():
            for vi in G['av']:
                for ei in G.adjacent(vi):
                    G.es[ei]['httBC'] = self._P.tube_hematocrit(
                                            G.es[ei]['diameter'], 'a')

        #Convert tube hematocrit boundary condition to htdBC (in case it does not already exist)
        if not 'htdBC' in G.es.attribute_names():
           G.es['htdBC']=[e['httBC'] if e['httBC'] == None else \
                self._P.tube_to_discharge_hematocrit(e['httBC'],e['diameter'],invivo) for e in G.es()]
        print('Htt BC assigned')

        httBC_edges = G.es(httBC_ne=None).indices
        #Save initial value of httBC
        G.es[httBC_edges]['httBC_init']=G.es[httBC_edges]['httBC']

        # Assign initial RBC positions:
	if init:
            if kwargs.has_key('hd0'):
                hd0=kwargs['hd0']
                if hd0 == 'current':
                    ht0=hd0
                else:
                    ht0='dummy'
            if kwargs.has_key('ht0'):
                ht0=kwargs['ht0']
            if ht0 != 'current':
                for e in G.es:
                    lrbc = e['minDist']
                    Nmax = max(int(np.floor(e['nMax'])), 1)
                    if e['httBC'] is not None:
                        N = int(np.round(e['httBC'] * Nmax))
                    else:
                        if kwargs.has_key('hd0'):
                            ht0=self._P.discharge_to_tube_hematocrit(hd0,e['diameter'],invivo)
                        N = int(np.round(ht0 * Nmax))
                    indices = sorted(np.random.permutation(Nmax)[:N])
                    e['rRBC'] = np.array(indices) * lrbc + lrbc / 2.0
                    #e['tRBC'] = np.array([])
        	    #e['path'] = np.array([])
        print('Initial nRBC computed')    
        G.es['nRBC']=[len(e['rRBC']) for e in G.es]
        for i in self._allEdgesBox:
            G.es[i]['RBCindex']=range(self._RBCindexCurrent+1,self._RBCindexCurrent+1+len(G.es[i]['rRBC']))
            self._RBCindexCurrent += len(G.es[i]['rRBC'])

        if kwargs.has_key('plasmaViscosity'):
            self._muPlasma = kwargs['plasmaViscosity']
        else:
            self._muPlasma = self._P.dynamic_plasma_viscosity()

        # Compute nominal and specific resistance:
        self._update_nominal_and_specific_resistance()
        print('Resistance updated')

        # Compute the current tube hematocrit from the RBC positions:
        for e in G.es:
            e['htt']=e['nRBC']*vrbc/(e['crosssection']*e['length'])
            e['htd']=min(htt2htd(e['htt'], e['diameter'], invivo), 0.95)
        print('Initial htt and htd computed')        

        # This initializes the full LS. Later, only relevant parts of
        # the LS need to be changed at any timestep. Also, RBCs are
        # removed from no-flow edges to avoid wasting computational
        # time on non-functional vascular branches / fragments:
        #Convert 'pBC' ['mmHG'] to default Units
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']*self._scaleToDef
        self._update_eff_resistance_and_LS(None, None)
        print('Matrix created')
        self._solve('iterative2')
        print('Matrix solved')
        self._G.vs['pressure'] = deepcopy(self._x)
        #Convert deaultUnits to 'pBC' ['mmHG']
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']/self._scaleToDef
        self._update_flow_and_velocity()
        print('Flow updated')
        self._verify_mass_balance()
        print('Mass balance verified updated')
        self._update_flow_sign()
        print('Flow sign updated')
        G.es['posFirstLast']=[None]*G.ecount()
        G.es['logNormal']=[None]*G.ecount()
        for i in httBC_edges:
            if len(G.es[i]['rRBC']) > 0:
                if G.es['sign'][i] == 1:
                    G.es[i]['posFirst_last']=G.es['rRBC'][i][0]
                else:
                    G.es[i]['posFirst_last']=G.es['length'][i]-G.es['rRBC'][i][-1]
            else:
                G.es[i]['posFirst_last']=G.es['length'][i]
            G.es[i]['v_last']=G.es[i]['v']
            httBCValue=G.es[i]['httBC']
            if self._innerDiam:
                LDValue = httBCValue
            else:
                LDValue=httBCValue*(G.es[i]['diameter']/(G.es[i]['diameter']-2*eslThickness(G.es[i]['diameter'])))**2
            logNormalMu,logNormalSigma=self._compute_mu_sigma_inlet_RBC_distribution(LDValue)
            G.es[i]['logNormal']=[logNormalMu,logNormalSigma]
        print('Initiallize posFirst_last')
        self._update_out_and_inflows_for_vertices()
        print('updated out and inflows')
	
        #Calculate an estimated network turnover time (based on conditions at the beginning)
        flowsum=0
	for vi in G['av']:
            for ei in G.adjacent(vi):
                flowsum=flowsum+G.es['flow'][ei]
        G['flowSumIn']=flowsum
        G['Ttau']=G['V']/flowsum
        print(flowsum)
        print(G['V'])
        print(self._eps)
        stdout.write("\rEstimated network turnover time Ttau=%f        \n" % G['Ttau'])

        #for e in self._G.es(flow_le=self._eps*1e6):
        #    e['rRBC'] = []
            
    #--------------------------------------------------------------------------

    def _compute_mu_sigma_inlet_RBC_distribution(self, httBC):
        """Updates the nominal and specific resistance of a given edge 
        sequence.
        INPUT: es: Sequence of edge indices as tuple. If not provided, all 
                   edges are updated.
        OUTPUT: None, the edge properties 'resistance' and 'specificResistance'
                are updated (or created).
        """
        #mean_LD=0.28
        mean_LD=httBC
        std_LD=0.1

        #PDF log-normal
        f_x = lambda x,mu,sigma: 1./(x*np.sqrt(2*np.pi)*sigma)*np.exp(-1*(np.log(x)-mu)**2/(2*np.sigma**2))

        #PDF log-normal for line density
        f_LD = lambda z,mu,sigma: 1./((z-z**2)*np.sqrt(2*np.pi)*sigma)*np.exp(-1*(np.log(1./z-1)-mu)**2/(2*sigma**2))

        #f_mean integral dummy
        f_mean_LD_dummy = lambda z,mu,sigma: z*f_LD(z,mu,sigma)

        ##calculate mean
        f_mean_LD = lambda mu,sigma: quad(f_mean_LD_dummy,0,1,args=(mu,sigma))[0]
        f_mean_LD_Calc=np.vectorize(f_mean_LD)

        #f_var integral dummy
        f_var_LD_dummy = lambda z,mu,sigma: (z-mean_LD)**2*f_LD(z,mu,sigma)

        #calculate mean
        f_var_LD = lambda mu,sigma: quad(f_var_LD_dummy,0,1,args=(mu,sigma))[0]
        f_var_LD_Calc=np.vectorize(f_var_LD)

        #Set up system of equations
        def f_moments_LD(m):
            x,y=m
            return (f_mean_LD_Calc(x,y)-mean_LD,f_var_LD_Calc(x,y)-std_LD**2)

        optionsSolve={}
        optionsSolve['xtol']=1e-20
        if mean_LD < 0.35:
            sol=root(f_moments_LD,(0.89,0.5),method='lm',options=optionsSolve)
        else:
            sol=root(f_moments_LD,(mean_LD,std_LD),method='lm',options=optionsSolve)
        mu=sol['x'][0]
        sigma=sol['x'][1]

        return mu,sigma

    #--------------------------------------------------------------------------

    def _update_nominal_and_specific_resistance(self, esequence=None):
        """Updates the nominal and specific resistance of a given edge 
        sequence.
        INPUT: es: Sequence of edge indices as tuple. If not provided, all 
                   edges are updated.
        OUTPUT: None, the edge properties 'resistance' and 'specificResistance'
                are updated (or created).
        """
        G = self._G
        muPlasma=self._muPlasma
        pi=np.pi  

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

    def _update_minDist_and_nMax(self, esequence=None):
        """Updates the length of the RBCs for each edge and the maximal Number
		of RBCs for each edge
        INPUT: es: Sequence of edge indices as tuple. If not provided, all 
                   edges are updated.
        OUTPUT: None, the edge properties 'nMax' and 'minDist'
                are updated (or created).
        """
        G = self._G

        if esequence is None:
            es = G.es
        else:
            es = G.es(esequence)
        # Compute the edge-specific minimal RBC distance:
        vrbc = self._P.rbc_volume()
        G.es['nMax'] = [np.pi * e['diameter']**2 / 4 * e['length'] / vrbc
                        for e in G.es]
        G.es['minDist'] = [e['length'] / e['nMax'] for e in G.es]

	self._G=G

    #--------------------------------------------------------------------------

    def _update_hematocrit(self, esequence=None):
        """Updates the tube hematocrit of a given edge sequence.
        INPUT: es: Sequence of edge indices as tuple. If not provided, all 
                   edges are updated.
        OUTPUT: None, the edge property 'htt' is updated (or created).
        """
        G = self._G
        htt2htd = self._P.tube_to_discharge_hematocrit
        invivo=self._invivo

        if esequence is None:
            es = range(G.ecount())
        else:
            es = esequence
        for e in es:
            G.es[int(e)]['htt'] = G.es[int(e)]['nRBC'] * G.es[int(e)]['minDist'] / G.es[int(e)]['length']
            G.es[int(e)]['htd']=min(htt2htd(G.es[int(e)]['htt'], G.es[int(e)]['diameter'], invivo), 0.95)
	self._G=G

    #--------------------------------------------------------------------------

    def _update_local_pressure_gradient(self):
        """Updates the local pressure gradient at all vertices.
        INPUT: None
        OUTPUT: None, the edge property 'lpg' is updated (or created, if it did
                not exist previously)
        """
        G = self._G
#        G.es['lpg'] = np.array(G.es['specificResistance']) * \
#                      np.array(G.es['flow']) * np.array(G.es['resistance']) / \
#                      np.array(G.es['effResistance'])
        G.es['lpg'] = np.array(G.es['specificResistance']) * \
                      np.array(G.es['flow'])

        self._G=G
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
        if 'sign' in G.es.attributes():
            G.es['signOld']=G.es['sign']
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
        G=self._G
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
        dThreshold = self._dThreshold
        count=0
        interfaceVertices=self._interfaceVertices
        if not 'inflowE' in G.vs.attributes():
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
                        #Deal with vertices at the interface
                        #isCap is defined based on the diameter of the InflowEdge
                        if vI in interfaceVertices and G.es[adjacents[j]]['diameter'] > dThreshold:
                            G.vs[vI]['isCap']=False
                        else:
                            G.vs[vI]['isCap']=True
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
                        G.es[edgeVI]['httBC']=None
                        G.es[edgeVI]['posFirst_last']=None
                        G.es[edgeVI]['v_last']=None
                    elif len(inE) == 0 and len(outE) == 0:
                        print('WARNING changed to noFlow edge')
                        noFlowV.append(vI)
                        noFlowE.append(edgeVI)
                    else:
                        print('ERROR in defining in and outlets')
                elif vI in G['vv']:
                    if len(inE) == 1 and len(outE) == 0:
                        pass
                    elif len(inE) == 0 and len(outE) == 1:
                        print('WARNING1 boundary condition changed: from vv --> av')
                        print(vI)
                        G.vs[vI]['av'] = 1
                        G.vs[vI]['vv'] = 0
                        G.vs[vI]['vType'] = 1
                        edgeVI=G.adjacent(vI)[0]
                        G.es[edgeVI]['httBC']=G.es[edgeVI]['httBC_init']
                        if len(G.es[edgeVI]['rRBC']) > 0:
                            if G.es['sign'][edgeVI] == 1:
                                G.es[edgeVI]['posFirst_last']=G.es['rRBC'][edgeVI][0]
                            else:
                                G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]-G.es['rRBC'][edgeVI][-1]
                        else:
                            G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]
                        G.es[edgeVI]['v_last']=G.es[edgeVI]['v']
                    elif len(inE) == 0 and len(outE) == 0:
                        print('WARNING changed to noFlow edge')
                        noFlowV.append(vI)
                        noFlowE.append(edgeVI)
                    else:
                        print('ERROR in defining in and outlets')
                #print problem cases
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
                    inE=[]
                    outE=[]
                    noFlowV.append(vI)
                inEdges.append(inE)
                outEdges.append(outE)
            G.vs['inflowE']=inEdges
            G.vs['outflowE']=outEdges
            G.es['noFlow']=[0]*G.ecount()
            noFlowE=np.unique(noFlowE)
            G.es[noFlowE]['noFlow']=[1]*len(noFlowE)
            G['divV']=divergentV
            G['conV']=convergentV
            G['connectV']=connectingV
            G['dConnectV']=doubleConnectingV
            G['noFlowV']=noFlowV
            print('assign vertex types')
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
            del(G['divV'])
            del(G['conV'])
            del(G['connectV'])
            del(G['dConnectV'])
        #Every Time Step
        else:
            print('Update_Out_and_inflows')
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
                            #Deal with vertices at the interface
                            #isCap is defined based on the diameter of the InflowEdge
                            if vI in interfaceVertices and G.es[adjacents[j]]['diameter'] > dThreshold:
                                G.vs[vI]['isCap']=False
                            else:
                                G.vs[vI]['isCap']=True
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
                        #print(' ')
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
                            print('WARNING direction out av changed to vv')
                            print(vI)
                            G.vs[vI]['av'] = 0
                            G.vs[vI]['vv'] = 1
                            G.vs[vI]['vType'] = 2
                            edgeVI=G.adjacent(vI)[0]
                            G.es[edgeVI]['httBC']=None
                            G.es[edgeVI]['posFirst_last']=None
                            G.es[edgeVI]['v_last']=None
                            G.vs[vI]['inflowE']=inE
                            G.vs[vI]['outflowE']=outE 
                    elif vI in G['vv']:
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
                            edgeVI=G.adjacent(vI)[0]
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
                        noFlowEdges=[]
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
                        G.vs[vI]['vType']=7
                        G.es[noFlowEdges]['noFlow']=[1]*len(noFlowEdges)
                        G.vs[vI]['inflowE']=[]
                        G.vs[vI]['outflowE']=[]
        G['av']=G.vs(av_eq=1).indices
        G['vv']=G.vs(vv_eq=1).indices
        stdout.flush()

    #--------------------------------------------------------------------------

    def _update_flow_and_velocity(self):
        """Updates the flow and red blood cell velocity in all vessels
        INPUT: None
        OUTPUT: None
        """

        G = self._G
        invivo=self._invivo
        vf = self._P.velocity_factor
        vrbc = self._P.rbc_volume()
        vfList=[1.0 if htt == 0.0 else max(1.,vf(d, invivo, tube_ht=htt)) for d,htt in zip(G.es['diameter'],G.es['htt'])]

        self._G=run_faster.update_flow_and_v(self._G,self._invivo,vfList,vrbc)
        G= self._G

        #G = self._G
        #invivo=self._invivo
        #vf = self._P.velocity_factor
        #pi=np.pi
        #G.es['flow'] = np.array([abs(G.vs[e.source]['pressure'] -                                           
        #                    G.vs[e.target]['pressure']) /res                        
        #                    for e,res in zip(G.es,G.es['effResistance'])])
        # RBC velocity is not defined if tube_ht==0, using plasma velocity
        # instead:
        #G.es['v'] = [4 * flow * vf(d, invivo, tube_ht=htt) /                  
        #            (pi * d**2) if htt > 0 else                                
        #            4 * flow / (pi * d**2)                                     
         #           for flow,d,htt in zip(G.es['flow'],G.es['diameter'],G.es['htt'])]

    #--------------------------------------------------------------------------

    def _update_eff_resistance_and_LS(self, newGraph=None, vertex=None):
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
        INPUT: newGraph: Vascular graph in iGraph format to replace the
                         previous self.G. (Optional, default=None.)
        OUTPUT: A: Matrix A of the linear system, holding the conductance
                   information.
                b: Vector b of the linear system, holding the boundary
                   conditions.
        """

        #if newGraph is not None:
        #    self._G = newGraph

        G = self._G
        P = self._P
        A = self._A
        b = self._b
        x = self._x
        invivo = self._invivo

        htt2htd = P.tube_to_discharge_hematocrit
        nurel = P.relative_apparent_blood_viscosity

        if vertex is None:
            vertexList = range(G.vcount())
            edgeList = range(G.ecount())
        else:
            vertexList=[]
            edgeList=[]
            for i in vertex:
                vList = np.concatenate([[i],
                     G.neighbors(i)]).tolist()
                eList = G.adjacent(i)
                vertexList=np.concatenate([vertexList,vList]).tolist()
                edgeList=np.concatenate([edgeList,eList]).tolist()
            vertexList=np.unique(vertexList).tolist()
            edgeList=np.unique(edgeList).tolist()
            edgeList=[int(i) for i in edgeList]
            vertexList=[int(i) for i in vertexList]
        dischargeHt = [min(htt2htd(e, d, invivo), 0.95) for e,d in zip(G.es[edgeList]['htt'],G.es[edgeList]['diameter'])]
        G.es[edgeList]['effResistance'] =[ res * nurel(d, dHt,invivo) for res,dHt,d in zip(G.es[edgeList]['resistance'], \
            dischargeHt,G.es[edgeList]['diamCalcEff'])]

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
                aDummy=0
                k=0
                neighbors=[]
                for edge in G.adjacent(i,'all'):
                    if G.is_loop(edge):
                        continue
                    j=G.neighbors(i)[k]
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
                A[i,i]=aDummy

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
        dt = self._dt # Time to propagate RBCs with current velocity.
        eps=self._eps
	#No flow Edges are not considered for the propagation of RBCs
        edgeList0=G.es(noFlow_eq=0).indices
        if self._analyzeBifEvents:
            rbcsMovedPerEdge=[]
            edgesWithMovedRBCs=[]
            rbcMoved = 0
        edgeList=G.es[edgeList0]
        RBCindexCurrent = self._RBCindexCurrent
        edgesBox=self._allEdgesBox
        borderEdges=self._borderEdges
        internalEdges=self._internalEdges
        #Edges are sorted based on the pressure at the outlet
        pOut=[G.vs[e['target']]['pressure'] if e['sign'] == 1.0 else G.vs[e['source']]['pressure']
            for e in edgeList]
        sortedE=zip(pOut,edgeList0)
        sortedE.sort()
        sortedE=[i[1] for i in sortedE]

        convEdges2=[]
        edgeUpdate=[]   #Edges where the number of RBCs changed --> need to be updated
        vertexUpdate=[] #Vertices where the number of RBCs changed in adjacent edges --> need to be updated
        #SECOND step go through all edges from smallest to highest pressure and move RBCs
        for ei in sortedE:
            edgesInvolved=[] #all edges connected to the bifurcation vertex
            e = G.es[ei]
            sign=e['sign']
            #Get bifurcation vertex
            if sign == 1:
                vi=e.target
            else:
                vi=e.source
            for i in G.vs[vi]['inflowE']:
                 edgesInvolved.append(i)
            for i in G.vs[vi]['outflowE']:
                 edgesInvolved.append(i)
            overshootsNo=0 #Reset - Number of overshoots acutally taking place (considers possible number of bifurcation events)
            #If there is a BC for the edge new RBCs have to be introduced
            boolHttEdge = 0
            boolHttEdge2 = 0
            boolHttEdge3 = 0
            if ei not in convEdges2 and G.vs[vi]['vType'] != 7:
                if e['httBC'] is not None:
                    boolHttEdge = 1
                    rRBC = []
                    RBCindex = []
                    lrbc = e['minDist']
                    htt = e['httBC']
                    length = e['length']
                    nMaxNew=e['nMax']-len(e['rRBC'])
                    if len(e['rRBC']) > 0:
                        #if cum_length > distToFirst:
                        posFirst=e['rRBC'][0] if e['sign']==1.0 else e['length']-e['rRBC'][-1]
                        e['posFirst_last']=posFirst
                        e['v_last'] = e['v']
                        cum_length = posFirst
                    else:
                        cum_length = e['posFirst_last'] + e['v_last'] * dt
                        posFirst = cum_length
                        e['posFirst_last']=posFirst
                        e['v_last']=e['v']
                    while cum_length >= lrbc and nMaxNew > 0:
                        if len(e['keep_rbcs']) != 0:
                            if posFirst - e['keep_rbcs'][0] >= 0:
                                rRBC.append(posFirst - e['keep_rbcs'][0])
                                nMaxNew += -1
                                posFirst=posFirst - e['keep_rbcs'][0]
                                cum_length = posFirst
                                e['keep_rbcs']=[]
                                e['posFirst_last']=posFirst
                                e['v_last']=e['v']
                                if e.index in edgesBox:
                                    RBCindexCurrent += 1
                                    RBCindex.append(RBCindexCurrent)
                            else:
                                if len(e['rRBC']) > 0:
                                    e['posFirst_last'] = posFirst
                                    e['v_last']=e['v']
                                else:
                                    e['posFirst_last'] += e['v_last'] * dt
                                break
                        else:
                            #number of RBCs randomly chosen to average htt
                            number=np.exp(e['logNormal'][0]+e['logNormal'][1]*np.random.randn(1)[0])
                            #self._spacing.append(number)
                            spacing = lrbc+lrbc*number
                            if posFirst - spacing >= 0:
                                rRBC.append(posFirst - spacing)
                                nMaxNew += -1
                                posFirst=posFirst - spacing
                                cum_length = posFirst
                                e['posFirst_last']=posFirst
                                e['v_last']=e['v']
                                if e.index in edgesBox:
                                    RBCindexCurrent += 1
                                    RBCindex.append(RBCindexCurrent)
                            else:
                                e['keep_rbcs']=[spacing]
                                e['v_last']=e['v']
                                if len(rRBC) == 0:
                                    e['posFirst_last']=posFirst
                                else:
                                    e['posFirst_last']=rRBC[-1]
                                break
                    rRBC = np.array(rRBC)
                    RBCindex=np.array(RBCindex)
                    if len(rRBC) >= 1.:
                        if e['sign'] == 1:
                            e['rRBC'] = np.concatenate([rRBC[::-1], e['rRBC']])
                            if e['RBCindex'] != None:
                                e['RBCindex']=np.concatenate([RBCindex[::-1], e['RBCindex']])
                        else:
                            e['rRBC'] = np.concatenate([e['rRBC'], length-rRBC])
                            if e['RBCindex'] != None:
                                e['RBCindex']=np.concatenate([e['RBCindex'],RBCindex])
                        vertexUpdate.append(e.target)
                        vertexUpdate.append(e.source)
                        edgeUpdate.append(ei)
            #Check if the RBCs in the edge have been moved already (--> convergent bifurcation)
            #Recheck if bifurcation vertex is a noFlow Vertex (vType=7)
            #if ei not in convEdges2 and G.vs[vi]['vType'] != 7:
                #If RBCs are present move all RBCs
                if len(e['rRBC']) > 0:
                    e['rRBC'] = e['rRBC'] + e['v'] * dt * e['sign']
                    #Deal with bifurcation events and overshoots in every edge
                    #bifRBCsIndes - array with overshooting RBCs from smallest to largest index
                    bifRBCsIndex=[]
                    nRBC=len(e['rRBC'])
                    if sign == 1.0:
                        if e['rRBC'][-1] > e['length']:
                            for i,j in enumerate(e['rRBC'][::-1]):
                                if j > e['length']:
                                    bifRBCsIndex.append(nRBC-1-i)
                                else:
                                    break
                        bifRBCsIndex=bifRBCsIndex[::-1]
                    else:
                        if e['rRBC'][0] < 0:
                            for i,j in enumerate(e['rRBC']):
                                if j < 0:
                                    bifRBCsIndex.append(i)
                                else:
                                    break
                    noBifEvents=len(bifRBCsIndex)
                else:
                    noBifEvents = 0
                #Convergent Edge without a bifurcation event
                if noBifEvents == 0 and G.vs[vi]['vType']==4:
                    convEdges2.append(ei)
        #-------------------------------------------------------------------------------------------
                #Check if a bifurcation event is taking place
                if noBifEvents > 0:
                    #If Edge is outlflow Edge --> remove RBCs
                    if G.vs[vi]['vType'] == 2:
                        overshootsNo=noBifEvents
                        e['rRBC']=[e['rRBC'][:-noBifEvents] if sign == 1.0 else e['rRBC'][noBifEvents::]][0]
                        if e['RBCindex'] != None:
                            e['RBCindex']=[e['RBCindex'][:-noBifEvents] if sign == 1.0 else e['RBCindex'][noBifEvents::]][0]
                        vertexUpdate.append(e.target)
                        vertexUpdate.append(e.source)
                        edgeUpdate.append(ei)
            #-------------------------------------------------------------------------------------------
                    #if vertex is connecting vertex
                    elif G.vs[vi]['vType'] == 5:
                        #print('at connecting vertex')
                        outE=G.vs[vi]['outflowE'][0]
                        oe=G.es[outE]
                        #Calculate possible number of bifurcation Events
			#distToFirst = distance to first vertex in vessel
                        if len(oe['rRBC']) > 0:
                            distToFirst=oe['rRBC'][0] if oe['sign'] == 1.0 else oe['length']-oe['rRBC'][-1]
                        else:
                            distToFirst=oe['length']
                        #Check how many RBCs fit into the new Vessel
                        posNoBifEvents=int(np.floor(distToFirst/oe['minDist']))
                        #Check how many RBCs are allowed by nMax --> limitation results from np.floor(length/minDist) 
			#and that RBCs are only 'half' in the vessel 
                        if posNoBifEvents + len(oe['rRBC']) > oe['nMax']:
                            posNoBifEvents = int(oe['nMax'] - len(oe['rRBC']))
                        #OvershootsNo: compare posNoBifEvents with noBifEvents
                        #posBifRBCsIndex --> array with possible number of bifurcations taking place
                        if posNoBifEvents > noBifEvents:
                            posBifRBCsIndex = bifRBCsIndex
                            overshootsNo=noBifEvents
                        elif posNoBifEvents == 0:
                            posBifRBCsIndex=[]
                            overshootsNo=0
                        else:
                            posBifRBCsIndex=[bifRBCsIndex[-posNoBifEvents::] if sign == 1.0 \
                                else bifRBCsIndex[:posNoBifEvents]]
                            overshootsNo=posNoBifEvents
                        if overshootsNo > 0:
                            #overshootsDist --> array with the distances which the RBCs overshoot, 
			    #starts wiht the RBC which overshoots the least 
                            overshootDist=[e['rRBC'][posBifRBCsIndex]-[e['length']]*overshootsNo if sign == 1.0
                                else [0]*overshootsNo-e['rRBC'][posBifRBCsIndex]][0]
                            if sign != 1.0:
                                overshootDist = overshootDist[::-1]
                            #overshootTime --> time which every RBCs overshoots
                            overshootTime=overshootDist / ([e['v']]*overshootsNo)
                            #position --> where the overshooting RBCs would be located in the outEdge
                            position=np.array(overshootTime)*np.array([oe['v']]*overshootsNo)
                            #Check if RBCs overshoot the whole downstream vessel
                            if len(oe['rRBC']) == 0:
                                if position[-1] > oe['length']:
                                    position = np.array(position)-np.array([position[-1]-oe['length']]*len(position))
                            #Maxmimum number of overshoots possible is infact limited by the overshootDistance of the first RBC
                            #If the RBCs travel with the same speed than the bulk flow this check is not necessary
			    #BUT due to different velocity factors RBCs cann "ran into each other" at connecting bifurcations
                            overshootsNoReduce=0
                            #Check if RBCs ran into another
                            for i in range(overshootsNo-1):
                                index=-1*(i+1)
                                if position[index]-position[index-1] < oe['minDist']:
                                    position[index-1]=position[index]-oe['minDist']
                                if position[index-1] < 0:
                                    overshootsNoReduce += 1
                            overshootsNo = overshootsNo-overshootsNoReduce
                            position=position[-1*overshootsNo::]
                            #Check if the RBCs overshoots RBCs present in the outflow vessel
                            overshootsNoReduce2=0
                            if len(oe['rRBC']) > 0:
                                if oe['sign'] == 1 and position[-1] > oe['rRBC'][0]-oe['minDist']:
                                    posLead=position[-1]
                                    position = np.array(position)-np.array([posLead-(oe['rRBC'][0]-oe['minDist'])]*len(position))
                                    for i in range(overshootsNo):
                                        if position[i] < 0:
                                            overshootsNoReduce2 += 1
                                        else:
                                            break
                                elif oe['sign'] == -1 and position[-1] > oe['length']-oe['rRBC'][-1]-oe['minDist']:
                                    posLead=position[-1]
                                    position = np.array(position)-np.array([posLead-(oe['length']-oe['rRBC'][-1]-oe['minDist'])]*len(position))
                                    for i in range(overshootsNo):
                                        if position[i] < 0:
                                            overshootsNoReduce2 += 1
                                        else:
                                            break
                            overshootsNo = overshootsNo-overshootsNoReduce2
                            if overshootsNo == 0:
                                position = []
                            else:
                                position=position[-1*overshootsNo::]
                            #Define RBCindex
                            if oe.index in borderEdges and e.index not in edgesBox:
                                print('All RBC indices created')
                                print('its a connecting bifurcation')
                                RBCindex=range(RBCindexCurrent+len(position),RBCindexCurrent,-1)
                                print('RBCindexCurrent')
                                print(RBCindexCurrent)
                                RBCindexCurrent += len(position)
                                print('RBCindex')
                                print(RBCindex)
                                print('RBCindexCurrent')
                                print(RBCindexCurrent)
                                edgesBoxBool=1
                            elif e.index in edgesBox and oe.index in edgesBox:
                                print('its a connecting bifurcation')
                                if e['sign'] == 1.0:
                                    RBCindex=e['RBCindex'][-overshootsNo::]
                                else:
                                    RBCindex=e['RBCindex'][:overshootsNo]
                                    RBCindex=RBCindex[::-1]
                                edgesBoxBool=1
                                print('RBCindex')
                                print(RBCindex)
                            else:
                                edgesBoxBool=0
                            #Add rbcs to new Edge       
                            if overshootsNo > 0:
                                if oe['sign'] == 1.0:
                                    oe['rRBC']=np.concatenate([position, oe['rRBC']])
                                    if edgesBoxBool:
                                        oe['RBCindex']=np.concatenate([RBCindex,oe['RBCindex']])
                                else:
                                    position = [oe['length']]*overshootsNo - position[::-1]
                                    oe['rRBC']=np.concatenate([oe['rRBC'],position])
                                    if edgesBoxBool:
                                        oe['RBCindex']=np.concatenate([oe['RBCindex'],RBCindex[::-1]])
                            #Remove RBCs from old Edge
                                if sign == 1.0:
                                    e['rRBC']=e['rRBC'][:-overshootsNo]
                                    if e['RBCindex'] != None:
                                        e['RBCindex']=e['RBCindex'][:-overshootsNo]
                                else:
                                    e['rRBC']=e['rRBC'][overshootsNo::]
                                    if e['RBCindex'] != None:
                                        e['RBCindex']=e['RBCindex'][overshootsNo::]
                            #Recheck if the distance between the newly introduces RBCs is still big enough
                                    #TODO: tnhis should not be necessary, 
                                    #TODO it is now implemented, that it is check if a RBC overshoots a whole vessel in 
                            #if len(oe['rRBC']) > 0:
                            #    #Check if RBCs overshot the whole vessel
                            #    if overshootsNo == len(oe['rRBC']):
                            #        if oe['sign'] == 1.0:
                            #            iter=overshootsNo-2
                            #            if oe['rRBC'][-1] > oe['length']:
                            #                oe['rRBC'][-1]=oe['length']
                            #        else:
                            #            iter=overshootsNo-1
                            #            if oe['rRBC'][0] < 0:
                            #                oe['rRBC'][0]=0
                            #    else:
                            #        if oe['sign']==1.0:
                            #            iter=overshootsNo-1
                            #        else:
                            #            iter=overshootsNo
                            #    if oe['sign']==1.0:
                            #        for i in range(iter,-1,-1):
                            #            if oe['rRBC'][i+1]-oe['rRBC'][i] + eps < oe['minDist'] \
                            #                or oe['rRBC'][i+1]-oe['rRBC'][i] + eps < 0:
                            #                print('ERROR CHECK should not be necessary')
                            #                print(oe['rRBC'][i+1]-oe['rRBC'][i])
                            #                print(oe['minDist'])
                            #                print(oe['rRBC'][i+1]-oe['rRBC'][i]-oe['minDist'])
                            #                print(eps*10000)
                            #                if oe['rRBC'][i]-oe['rRBC'][i-1] == oe['minDist']:
                            #                    print('YES')
                            #                oe['rRBC'][i]=oe['rRBC'][i+1]-oe['minDist']
                            #    else:
                            #        for i in range(-1*iter,0,1):
                            #            if oe['rRBC'][i]-oe['rRBC'][i-1] + eps < oe['minDist'] \
                            #                or oe['rRBC'][i]-oe['rRBC'][i-1] + eps < 0:
                            #                print('ERROR 2 CHECK should not be necessary')
                            #                print(oe['rRBC'][i]-oe['rRBC'][i-1])
                            #                print(oe['minDist'])
                            #                print(oe['rRBC'][i]-oe['rRBC'][i-1]-oe['minDist'])
                            #                print(eps*10000)
                            #                if oe['rRBC'][i]-oe['rRBC'][i-1] == oe['minDist']:
                            #                    print('YES')
                            #                oe['rRBC'][i]=oe['rRBC'][i-1]+oe['minDist']
                            #                oe['rRBC'][i]=oe['rRBC'][i-1]+oe['minDist']
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        noStuckRBCs=len(bifRBCsIndex)-overshootsNo
                        #move stuck RBCs back into vessel
                        for i in range(noStuckRBCs):
                            index=-1*(i+1) if sign == 1.0 else i
                            e['rRBC'][index]=e['length']-i*e['minDist'] if sign == 1.0 else 0+i*e['minDist']
                        #Recheck if the distance between the newly introduces RBCs is still big enough 
                        if len(e['rRBC']) > 1:
                            moved = 0
                            count = 0
                            if sign == 1.0:
                                for i in range(-1,-1*(len(e['rRBC'])),-1):
                                    index=i-1
                                    if e['rRBC'][i] < e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]-e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs and moved == 0:
                                        break
                            else:
                                for i in range(len(e['rRBC'])-1):
                                    index=i+1
                                    if e['rRBC'][i] > e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]+e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs+1 and moved == 0:
                                        break
          #-------------------------------------------------------------------------------------------
                    #if vertex is divergent vertex
                    elif G.vs[vi]['vType'] == 3:
                        #print('at divergent vertex')
                        outEdges=G.vs[vi]['outflowE']
                        outE=outEdges[0]
                        outE2=outEdges[1]
                        #Check if there are two or three outEdgs
                        if len(outEdges) > 2:
                            outE3=outEdges[2]
                        #Differ between capillaries and non-capillaries
                        if G.vs[vi]['isCap']:
                            preferenceList = [x[1] for x in sorted(zip(np.array(G.es[outEdges]['flow'])/np.array(G.es[outEdges]['crosssection']), \
                                outEdges), reverse=True)]
                        else:
                            preferenceList = [x[1] for x in sorted(zip(G.es[outEdges]['flow'], outEdges), reverse=True)]
                        #Define prefered OutEdges based on bifurcation rule
                        outEPref=preferenceList[0]
                        outEPref2=preferenceList[1]
                        oe=G.es[outEPref]
                        oe2=G.es[outEPref2]
                        if len(outEdges) > 2:
                            outEPref3 = preferenceList[2]
                            oe3=G.es[outEPref3]
                        #Calculate distance to first RBC for outEPref
                        if len(oe['rRBC']) > 0:
                            distToFirst=oe['rRBC'][0] if oe['sign'] == 1.0 \
                                else oe['length']-oe['rRBC'][-1]
                        else:
                            distToFirst=oe['length']
                        #Calculate distance to first RBC for outEPref2
                        if len(oe2['rRBC']) > 0:
                            distToFirst2=oe2['rRBC'][0] if oe2['sign'] == 1.0 \
                                else oe2['length']-oe2['rRBC'][-1]
                        else:
                            distToFirst2=oe2['length']
                        #Calculate distance to first RBC for outEPref3 (if it exists)
                        if len(outEdges) > 2:
                            if len(oe3['rRBC']) > 0:
                                distToFirst3=oe3['rRBC'][0] if oe3['sign'] == 1.0 \
                                    else oe3['length']-oe3['rRBC'][-1]
                            else:
                                distToFirst3=oe3['length']
                        #Check how many RBCs are allowed by nMax for outEPref
                        posNoBifEventsPref=np.floor(distToFirst/oe['minDist'])
                        if posNoBifEventsPref + len(oe['rRBC']) > oe['nMax']:
                            posNoBifEventsPref = oe['nMax'] - len(oe['rRBC'])
                        #Check how many RBCs are allowed by nMax for outEPref2
                        posNoBifEventsPref2=np.floor(distToFirst2/oe2['minDist'])
                        if posNoBifEventsPref2 + len(oe2['rRBC']) > oe2['nMax']:
                            posNoBifEventsPref2 = oe2['nMax'] - len(oe2['rRBC'])
                        #Check how many RBCs are allowed by nMax for outEPref3
                        if len(outEdges) > 2:
                            posNoBifEventsPref3=np.floor(distToFirst3/oe3['minDist'])
                            if posNoBifEventsPref3 + len(oe3['rRBC']) > oe3['nMax']:
                                posNoBifEventsPref3 = oe3['nMax'] - len(oe3['rRBC'])
                        #Calculate total number of bifurcation events possible
                        if len(outEdges) > 2:
                            posNoBifEvents=int(posNoBifEventsPref+posNoBifEventsPref2+posNoBifEventsPref3)
                        else:
                            posNoBifEvents=int(posNoBifEventsPref+posNoBifEventsPref2)
                        #Compare possible number of bifurcation events with number of bifurcations taking place
                        if posNoBifEvents > noBifEvents:
                            posBifRBCsIndex=bifRBCsIndex
                            overshootsNo=noBifEvents
                        elif posNoBifEvents == 0:
                            posBifRBCsIndex=[]
                            overshootsNo=0
                        else:
                            posBifRBCsIndex=[bifRBCsIndex[-posNoBifEvents::] if sign == 1.0 \
                                else bifRBCsIndex[:posNoBifEvents]]
                            overshootsNo=posNoBifEvents
                        if overshootsNo > 0:
                            #overshootDist starts with the RBC which overshoots the least
                            overshootDist=[e['rRBC'][posBifRBCsIndex]-[e['length']]*overshootsNo if sign == 1.0
                                else [0]*overshootsNo-e['rRBC'][posBifRBCsIndex]][0]
                            if sign != 1.0:
                                overshootDist = overshootDist[::-1]
                            #overshootTime starts with the RBC which overshoots the least
                            overshootTime=overshootDist / ([e['v']]*overshootsNo)
                            #Calculate position of overshootRBCs in every outEdge
                            #the values in position are stored such that they can directly concatenated with outE['rRBC']
			    #flow direction of outEdge is considered
                            #position = [pos_min ... pos_max]
                            if oe['sign'] == 1.0:
                                position1=np.array(overshootTime)*np.array([oe['v']]*overshootsNo)
                            else:
                                position1=np.array([oe['length']]*overshootsNo)-np.array(overshootTime[::-1])* \
                                    np.array([oe['v']]*overshootsNo)
                            if oe2['sign'] == 1.0:
                                position2=np.array(overshootTime)*np.array([oe2['v']]*overshootsNo)
                            else:
                                position2=np.array([oe2['length']]*overshootsNo)-np.array(overshootTime[::-1])* \
                                    np.array([oe2['v']]*overshootsNo)
                            if len(outEdges) > 2:
                                if oe3['sign'] == 1.0:
                                    position3=np.array(overshootTime)*np.array([oe3['v']]*overshootsNo)
                                else:
                                    position3=np.array([oe3['length']]*overshootsNo)-np.array(overshootTime[::-1])* \
                                        np.array([oe3['v']]*overshootsNo)
                            #To begin with it is tried if all RBCs fit into the prefered outEdge. The time of arrival at the RBCs is take into account
                            #RBCs which would be too close together are put into the other edge
                            #postion2/position3 is used if there is not enough space in the prefered outEdge and hence the RBC is moved to the other outEdge
                            positionPref3=[]
                            positionPref2=[]
                            positionPref1=[]
                            counterPref3=[]
                            counterPref2=[]
                            counterPref1=[]
                            #number of RBCs in the Edges
                            countPref1=0
                            countPref2=0
                            countPref3=0
                            pref1Full=0
                            pref2Full=0
                            pref3Full=0
                            #Loop over all movable RBCs (begin with RBC which overshot the most)
                            for i in range(overshootsNo):
                                index=-1*(i+1) if sign == 1.0 else i
                                index1=-1*(i+1) if oe['sign'] == 1.0 else i
                                index2=-1*(i+1) if oe2['sign'] == 1.0 else i
                                #The possible number of RBCs results from the distance the first RBC overshoots
                                #it can happen that due to that more RBCs are blocked than expected, that is checked with the following values
                                if len(outEdges) > 2:
                                    index3=-1*(i+1) if oe3['sign'] == 1.0 else i
                                #check if RBC still fits into Prefered OutE
                                if posNoBifEventsPref > countPref1 and pref1Full == 0:
                                    #Check if there are RBCs present in outEPref
                                    #RBCs have already been put into outEPref1
                                    if positionPref1 != []: 
                                        #Check if distance to preceding RBC is big enough
                                        dist1=positionPref1[-1]-position1[index1] if oe['sign'] == 1.0 \
                                            else position1[index1]-positionPref1[-1]
                                        #If the distance is not big enough check if RBC fits into outEPref2
                                        if dist1 < oe['minDist']:
                                            #if RBCs are present in the outEdgePref2
                                            #check if RBC still fits into outEPref2
                                            if posNoBifEventsPref2 > countPref2 and pref2Full == 0:
                                                if positionPref2 != []:
                                                    #Check if distance to preceding RBC is big enough
                                                    dist2=positionPref2[-1]-position2[index2] if oe2['sign'] == 1.0 \
                                                        else position2[index2]-positionPref2[-1]
                                                    #If the distance is not big enough check if RBC fits into outEPref3
                                                    if dist2 < oe2['minDist']:
                                                        #Check if there is a third outEdge
                                                        if len(outEdges)>2:
                                                            #check if RBC still fits into outEPref3
                                                            if posNoBifEventsPref3 > countPref3 and pref3Full == 0:
                                                                #Check if there are RBCs in the third outEdge
                                                                if positionPref3 != []: 
                                                                    #Check if distance to preceding RBC is big enough
                                                                    dist3=positionPref3[-1]-position3[index3] if oe3['sign'] == 1.0 \
                                                                        else position3[index3]-positionPref3[-1]
                                                                    #if there is not enough space in the third outEdge
                                                                    #Check in which edge the RBC is blocked the shortest time
                                                                    if dist3 < oe3['minDist']:
                                                                        space1 =  positionPref1[-1] if oe['sign'] == 1.0 \
                                                                            else oe['length']-positionPref1[-1]
                                                                        if np.floor(space1/oe['minDist']) >= 1:
                                                                            timeBlocked1=(oe['minDist']-dist1)/oe['v']
                                                                        else:
                                                                            timeBlocked1=None
                                                                            pref1Full=1
                                                                        space2 =  positionPref2[-1] if oe2['sign'] == 1.0 \
                                                                            else oe2['length']-positionPref2[-1]
                                                                        if np.floor(space2/oe2['minDist']) >= 1:
                                                                            timeBlocked2=(oe2['minDist']-dist2)/oe2['v']
                                                                        else:
                                                                            timeBlocked2=None
                                                                            pref2Full=1
                                                                        space3 =  positionPref3[-1] if oe3['sign'] == 1.0 \
                                                                            else oe3['length']-positionPref3[-1]
                                                                        if np.floor(space3/oe3['minDist']) >= 1:
                                                                            timeBlocked3=(oe3['minDist']-dist3)/oe3['v']
                                                                        else:
                                                                            timeBlocked3=None
                                                                            pref3Full=1
                                                                        if pref1Full == 1 and pref2Full == 1 and pref3Full == 1:
                                                                            break
                                                                        #Define newOutEdge
                                                                        newOutEdge=0
                                                                        if timeBlocked1 == None: #2 or 3
                                                                            if timeBlocked2 == None:
                                                                                newOutEdge=3
                                                                            elif timeBlocked3 == None:
                                                                                newOutEdge=2
                                                                            elif timeBlocked2 <= timeBlocked3:
                                                                                newOutEdge=2
                                                                            elif timeBlocked3 <= timeBlocked2:
                                                                                newOutEdge=3
                                                                        elif timeBlocked2 == None: #1 or 3
                                                                            if timeBlocked3 == None:
                                                                                newOutEdge=1
                                                                            elif timeBlocked1 <= timeBlocked3:
                                                                                newOutEdge=1
                                                                            elif timeBlocked3 <= timeBlocked1:
                                                                                newOutEdge=3
                                                                        elif timeBlocked3 == None: #1 or 2
                                                                            if timeBlocked2 <= timeBlocked1:
                                                                                newOutEdge=2
                                                                            elif timeBlocked1 <= timeBlocked2:
                                                                                newOutEdge=1
                                                                        else:
                                                                            if np.min([timeBlocked1,timeBlocked2,timeBlocked3]) == timeBlocked1:
                                                                                newOutEdge=1
                                                                            elif np.min([timeBlocked1,timeBlocked2,timeBlocked3]) == timeBlocked2:
                                                                                newOutEdge=2
                                                                            elif np.min([timeBlocked1,timeBlocked2,timeBlocked3]) == timeBlocked3:
                                                                                newOutEdge=3
                                                                        if newOutEdge == 1:
                                                                            if oe['sign'] == 1.0:
                                                                                position1[index1]=positionPref1[-1]-oe['minDist']
                                                                                if position1[index1] > 0:
                                                                                    positionPref1.append(position1[index1])
                                                                                    counterPref1.append(index)
                                                                                    countPref1 += 1
                                                                                else:
                                                                                    print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 1')
                                                                                    print(position1[index1])
                                                                                    print(np.floor(space1/oe['minDist']))
                                                                                    print(timeBlocked1)
                                                                                    print(pref1Full)
                                                                            else:
                                                                                position1[index1]=positionPref1[-1]+oe['minDist']
                                                                                if position1[index1] < oe['length']:
                                                                                    positionPref1.append(position1[index1])
                                                                                    counterPref1.append(index)
                                                                                    countPref1 += 1
                                                                                else:
                                                                                    print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 2')
                                                                                    print(position1[index1])
                                                                                    print(oe['length'])
                                                                                    print(np.floor(space1/oe['minDist']))
                                                                        elif newOutEdge == 2:
                                                                            if oe2['sign'] == 1.0:
                                                                                position2[index2]=positionPref2[-1]-oe2['minDist']
                                                                                if position2[index2] > 0:
                                                                                    positionPref2.append(position2[index2])
                                                                                    counterPref2.append(index)
                                                                                    countPref2 += 1
                                                                                else:
                                                                                    print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 3')
                                                                                    print(position2[index2])
                                                                                    print(np.floor(space2/oe2['minDist']))
                                                                                    print(timeBlocked2)
                                                                                    print(pref2Full)
                                                                            else:
                                                                                position2[index2]=positionPref2[-1]+oe2['minDist']
                                                                                if position2[index2] < oe2['length']:
                                                                                    positionPref2.append(position2[index2])
                                                                                    counterPref2.append(index)
                                                                                    countPref2 += 1
                                                                                else:
                                                                                    print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 4')
                                                                                    print(position2[index2])
                                                                                    print(oe2['length'])
                                                                                    print(np.floor(space2/oe2['minDist']))
                                                                        elif newOutEdge == 3:
                                                                            if oe3['sign'] == 1.0:
                                                                                position3[index3]=positionPref3[-1]-oe3['minDist']
                                                                                if position3[index3] > 0:
                                                                                    positionPref3.append(position3[index3])
                                                                                    counterPref3.append(index)
                                                                                    countPref3 += 1
                                                                                else:
                                                                                    print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 5')
                                                                                    print(position3[index3])
                                                                                    print(np.floor(space3/oe3['minDist']))
                                                                                    print(timeBlocked3)
                                                                                    print(pref3Full)
                                                                            else:
                                                                                position3[index3]=positionPref3[-1]+oe3['minDist']
                                                                                if position3[index3] < oe3['length']:
                                                                                    positionPref3.append(position3[index3])
                                                                                    counterPref3.append(index)
                                                                                    countPref3 += 1
                                                                                else:
                                                                                    print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 6')
                                                                                    print(position3[index3])
                                                                                    print(oe3['length'])
                                                                                    print(np.floor(space3/oe3['minDist']))
                                                                    #There is enough space in outEdge 3
                                                                    else:
                                                                        positionPref3.append(position3[index3])
                                                                        counterPref3.append(index)
                                                                        countPref3 += 1
                                                                #No RBCs have been put in outEPref3 so far
                                                                else:
                                                                    if oe3['sign'] == 1.0:
                                                                        #If there are are already RBCs present in outE3
                                                                        #Check that there is no overtaking of RBCs
                                                                        if len(oe3['rRBC']) > 0:
                                                                            if oe3['rRBC'][0]-position3[index3] < oe3['minDist']:
                                                                                position3[index3]=oe3['rRBC'][0]-oe3['minDist']
                                                                        #There are no RBCs present in outE3
                                                                        else:
                                                                            #Avoid overshooting whole vessels
                                                                            if position3[index3] > oe3['length']:
                                                                                position3[index3]=oe3['length']
                                                                    else:
                                                                        #If there are are already RBCs present in outE3
                                                                        #Check that there is no overtaking of RBCs
                                                                        if len(oe3['rRBC']) > 0:
                                                                            if position3[index3]-oe3['rRBC'][-1] < oe3['minDist']:
                                                                              position3[index3]=oe3['rRBC'][-1]+oe3['minDist']
                                                                        else:
                                                                            #Avoid overshooting whole vessels
                                                                            if position3[index3] < 0:
                                                                                position3[index3]=0
                                                                    positionPref3.append(position3[index3])
                                                                    counterPref3.append(index)
                                                                    countPref3 += 1
                                                            #There is no spcae in the third outEdge anymore
                                                            else:
                                                                #Check if another RBCs still fits into the vessel
                                                                space1 =  positionPref1[-1] if oe['sign'] == 1.0 \
                                                                    else oe['length']-positionPref1[-1]
                                                                if np.floor(space1/oe['minDist']) > 1:
                                                                    timeBlocked1=(oe['minDist']-dist1)/oe['v']
                                                                else:
                                                                    timeBlocked1=None
                                                                    pref1Full=1
                                                                space2 =  positionPref2[-1] if oe2['sign'] == 1.0 \
                                                                    else oe2['length']-positionPref2[-1]
                                                                if np.floor(space2/oe2['minDist']) > 1:
                                                                    timeBlocked2=(oe2['minDist']-dist2)/oe2['v']
                                                                else:
                                                                    timeBlocked2=None
                                                                    pref2Full=1
                                                                if pref1Full == 1 and pref2Full == 1:
                                                                    break
                                                                #Define newOutEdge
                                                                newOutEdge=0
                                                                if timeBlocked1 == None:
                                                                    newOutEdge=2
                                                                elif timeBlocked2 == None:
                                                                    newOutEdge=1
                                                                else:
                                                                    if timeBlocked1 <= timeBlocked2:
                                                                        newOutEdge=1
                                                                    else:
                                                                        newOutEdge=2
                                                                if newOutEdge == 1:
                                                                    if oe['sign'] == 1.0:
                                                                        position1[index1]=positionPref1[-1]-oe['minDist']
                                                                        if position1[index1] > 0:
                                                                            positionPref1.append(position1[index1])
                                                                            counterPref1.append(index)
                                                                            countPref1 += 1
                                                                        else:
                                                                            print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 7')
                                                                            print(position1[index1])
                                                                            print(np.floor(space1/oe['minDist']))
                                                                            print(timeBlocked1)
                                                                            print(pref1Full)
                                                                    else:
                                                                        position1[index1]=positionPref1[-1]+oe['minDist']
                                                                        if position1[index1] < oe['length']:
                                                                            positionPref1.append(position1[index1])
                                                                            counterPref1.append(index)
                                                                            countPref1 += 1
                                                                        else:
                                                                            print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 8')
                                                                            print(position1[index1])
                                                                            print(oe['length'])
                                                                            print(np.floor(space1/oe['minDist']))
                                                                elif newOutEdge == 2:
                                                                    if oe2['sign'] == 1.0:
                                                                        position2[index2]=positionPref2[-1]-oe2['minDist']
                                                                        if position2[index2] > 0:
                                                                            positionPref2.append(position2[index2])
                                                                            counterPref2.append(index)
                                                                            countPref2 += 1
                                                                        else:
                                                                            print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 9')
                                                                            print(position2[index2])
                                                                            print(np.floor(space2/oe2['minDist']))
                                                                            print(timeBlocked2)
                                                                            print(pref2Full)
                                                                    else:
                                                                        position2[index2]=positionPref2[-1]+oe2['minDist']
                                                                        if position2[index2] < oe2['length']:
                                                                            positionPref2.append(position2[index2])
                                                                            counterPref2.append(index)
                                                                            countPref2 += 1
                                                                        else:
                                                                            print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 10')
                                                                            print(position2[index2])
                                                                            print(oe2['length'])
                                                                            print(np.floor(space2/oe2['minDist']))
                                                        #There is no third outEdge, therefore it is checked in which edge the RBC is blocked
                                                        #the shortest time
                                                        else:
                                                            #Check if another RBCs still fits into the vessel
                                                            space1 =  positionPref1[-1] if oe['sign'] == 1.0 \
                                                                else oe['length']-positionPref1[-1]
                                                            if np.floor(space1/oe['minDist']) > 1:
                                                                timeBlocked1=(oe['minDist']-dist1)/oe['v']
                                                            else:
                                                                timeBlocked1=None
                                                                pref1Full=1
                                                            space2 =  positionPref2[-1] if oe2['sign'] == 1.0 \
                                                                else oe2['length']-positionPref2[-1]
                                                            if np.floor(space2/oe2['minDist']) > 1:
                                                                timeBlocked2=(oe2['minDist']-dist2)/oe2['v']
                                                            else:
                                                                timeBlocked2=None
                                                                pref2Full=1
                                                            if pref1Full == 1 and pref2Full == 1:
                                                                break
                                                            #Define newOutEdge
                                                            newOutEdge=0
                                                            if timeBlocked1 == None:
                                                                newOutEdge=2
                                                            elif timeBlocked2 == None:
                                                                newOutEdge=1
                                                            else:
                                                                if timeBlocked1 <= timeBlocked2:
                                                                    newOutEdge=1
                                                                else:
                                                                    newOutEdge=2
                                                            if newOutEdge == 1:
                                                                if oe['sign'] == 1.0:
                                                                    position1[index1]=positionPref1[-1]-oe['minDist']
                                                                    if position1[index1] > 0:
                                                                        positionPref1.append(position1[index1])
                                                                        counterPref1.append(index)
                                                                        countPref1 += 1
                                                                    else:
                                                                        print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 11')
                                                                        print(position1[index1])
                                                                        print(np.floor(space1/oe['minDist']))
                                                                        print(timeBlocked1)
                                                                        print(pref1Full)
                                                                else:
                                                                    position1[index1]=positionPref1[-1]+oe['minDist']
                                                                    if position1[index1] < oe['length']:
                                                                        positionPref1.append(position1[index1])
                                                                        counterPref1.append(index)
                                                                        countPref1 += 1
                                                                    else:
                                                                        print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 12')
                                                                        print(position1[index1])
                                                                        print(oe['length'])
                                                                        print(np.floor(space1/oe['minDist']))
                                                            elif newOutEdge == 2:
                                                                if oe2['sign'] == 1.0:
                                                                    position2[index2]=positionPref2[-1]-oe2['minDist']
                                                                    if position2[index2] > 0:
                                                                        positionPref2.append(position2[index2])
                                                                        counterPref2.append(index)
                                                                        countPref2 += 1
                                                                    else:
                                                                        print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 13')
                                                                        print(position2[index2])
                                                                        print(np.floor(space2/oe2['minDist']))
                                                                        print(timeBlocked2)
                                                                        print(pref2Full)
                                                                else:
                                                                    position2[index2]=positionPref2[-1]+oe2['minDist']
                                                                    if position2[index2] < oe2['length']:
                                                                        positionPref2.append(position2[index2])
                                                                        counterPref2.append(index)
                                                                        countPref2 += 1
                                                                    else:
                                                                        print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 14')
                                                                        print(position2[index2])
                                                                        print(oe2['length'])
                                                                        print(np.floor(space2/oe2['minDist']))
                                                    #there is enough space for the RBC in outEPref2
                                                    else:
                                                        positionPref2.append(position2[index2])
                                                        counterPref2.append(index)
                                                        countPref2 += 1
                                                #no RBCs have been put in outEPref2 so far
                                                else:
                                                    if oe2['sign'] == 1.0:
                                                        #If there are are already RBCs present in outE2
                                                        #Check that there is no overtaking of RBCs
                                                        if len(oe2['rRBC']) > 0:
                                                            if oe2['rRBC'][0]-position2[index2] < oe2['minDist']:
                                                                position2[index2]=oe2['rRBC'][0]-oe2['minDist']
                                                        #There are no RBCs present in outE2
                                                        else:
                                                            #Avoid overshooting whole vessels
                                                            if position2[index2] > oe2['length']:
                                                                position2[index2]=oe2['length']
                                                    else:
                                                        if len(oe2['rRBC']) > 0:
                                                            if position2[index2]-oe2['rRBC'][-1] < oe2['minDist']:
                                                                position2[index2]=oe2['rRBC'][-1]+oe2['minDist']
                                                        else:
                                                            if position2[index2] < 0:
                                                                position2[index2]=0
                                                    positionPref2.append(position2[index2])
                                                    counterPref2.append(index)
                                                    countPref2 += 1
                                            #There is no space in the second outEdge
					    #Check if there is a third outEdge
                                            else:
                                                if len(outEdges)>2:
                                                    #check if RBC still fits into outEPref3
                                                    if posNoBifEventsPref3 > countPref3 and pref3Full == 0:
                                                        #Check if RBCs have already been put into 
                                                        if positionPref3 != []:
                                                        #Check if distance to preceding RBC is big enough
                                                            dist3=positionPref3[-1]-position3[index3] if oe3['sign'] == 1.0 \
                                                                else position3[index3]-positionPref3[-1]
                                                            if dist3 < oe3['minDist']:
                                                                #Check if another RBCs still fits into the vessel
                                                                space1 =  positionPref1[-1] if oe['sign'] == 1.0 \
                                                                    else oe['length']-positionPref1[-1]
                                                                if np.floor(space1/oe['minDist']) > 1:
                                                                    timeBlocked1=(oe['minDist']-dist1)/oe['v']
                                                                else:
                                                                    timeBlocked1=None
                                                                    pref1Full=1
                                                                space3 =  positionPref3[-1] if oe3['sign'] == 1.0 \
                                                                    else oe3['length']-positionPref3[-1]
                                                                if np.floor(space3/oe3['minDist']) > 1:
                                                                    timeBlocked3=(oe3['minDist']-dist3)/oe3['v']
                                                                else:
                                                                    timeBlocked3=None
                                                                    pref3Full=1
                                                                if pref1Full == 1 and pref3Full == 1:
                                                                    break
                                                                #Define newOutEdge
                                                                newOutEdge=0
                                                                if timeBlocked1 == None:
                                                                    newOutEdge=3
                                                                elif timeBlocked3 == None:
                                                                    newOutEdge=1
                                                                else:
                                                                    if timeBlocked1 <= timeBlocked3:
                                                                        newOutEdge=1
                                                                    else:
                                                                        newOutEdge=3
                                                                if newOutEdge == 1:
                                                                    if oe['sign'] == 1.0:
                                                                        position1[index1]=positionPref1[-1]-oe['minDist']
                                                                        if position1[index1] > 0:
                                                                            positionPref1.append(position1[index1])
                                                                            counterPref1.append(index)
                                                                            countPref1 += 1
                                                                        else:
                                                                            print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN! 15')
                                                                            print(position1[index1])
                                                                            print(np.floor(space1/oe['minDist']))
                                                                            print(timeBlocked1)
                                                                            print(pref1Full)
                                                                    else:
                                                                        position1[index1]=positionPref1[-1]+oe['minDist']
                                                                        if position1[index1] < oe['length']:
                                                                            positionPref1.append(position1[index1])
                                                                            counterPref1.append(index)
                                                                            countPref1 += 1
                                                                        else:
                                                                            print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN! 16')
                                                                            print(position1[index1])
                                                                            print(oe['length'])
                                                                            print(np.floor(space1/oe['minDist']))
                                                                elif newOutEdge == 3:
                                                                    if oe3['sign'] == 1.0:
                                                                        position3[index3]=positionPref3[-1]-oe3['minDist']
                                                                        if position3[index3] > 0:
                                                                            positionPref3.append(position3[index3])
                                                                            counterPref3.append(index)
                                                                            countPref3 += 1
                                                                        else:
                                                                            print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN! 17')
                                                                            print(position3[index3])
                                                                            print(np.floor(space3/oe3['minDist']))
                                                                            print(timeBlocked3)
                                                                            print(pref3Full)
                                                                    else:
                                                                        position3[index3]=positionPref3[-1]+oe3['minDist']
                                                                        if position3[index3] < oe3['length']:
                                                                            positionPref3.append(position3[index3])
                                                                            counterPref3.append(index)
                                                                            countPref3 += 1
                                                                        else:
                                                                            print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 18')
                                                                            print(position3[index3])
                                                                            print(oe3['length'])
                                                                            print(np.floor(space3/oe3['minDist']))
                                                            #There is enough space in outEdge 3
                                                            else:
                                                                positionPref3.append(position3[index3])
                                                                counterPref3.append(index)
                                                                countPref3 += 1
                                                        #No RBCs have been put in outEPref3
                                                        else:
                                                            if oe3['sign'] == 1.0:
                                                            #If there are are already RBCs present in outE3
                                                            #Check that there is no overtaking of RBCs
                                                                if len(oe3['rRBC']) > 0:
                                                                    if oe3['rRBC'][0]-position3[index3] < oe3['minDist']:
                                                                        position3[index3]=oe3['rRBC'][0]-oe3['minDist']
                                                                #There are no RBCs present in outE3
                                                                else:
                                                                    #Avoid overshooting whole vessels
                                                                    if position3[index3] > oe3['length']:
                                                                        position3[index3]=oe3['length']
                                                            else:
                                                            #If there are are already RBCs present in outE3
                                                            #Check that there is no overtaking of RBCs
                                                                if len(oe3['rRBC']) > 0:
                                                                    if position3[index3]-oe3['rRBC'][-1] < oe3['minDist']:
                                                                        position3[index3]=oe3['rRBC'][-1]+oe3['minDist']
                                                                    else:
                                                                    #Avoid overshooting whole vessels
                                                                        if position3[index3] < 0:
                                                                            position3[index3]=0
                                                            positionPref3.append(position3[index3])
                                                            counterPref3.append(index)
                                                            countPref3 += 1
                                                    #There is no space in the third outEdge
                                                    else:
                                                    #RBC pushed backwards in Edge 1
                                                        if oe['sign'] == 1.0:
                                                            position1[index1]=positionPref1[-1]-oe['minDist']
                                                            if position1[index1] > 0:
                                                                positionPref1.append(position1[index1])
                                                                counterPref1.append(index)
                                                                countPref1 += 1
                                                            else:
                                                                 pref1Full=1
                                                                 break
                                                        else:
                                                            position1[index1]=positionPref1[-1]+oe['minDist']
                                                            if position1[index1] < oe['length']:
                                                                positionPref1.append(position1[index1])
                                                                counterPref1.append(index)
                                                                countPref1 += 1
                                                            else:
                                                                 pref1Full=1
                                                                 break
                                                #There is no third outEdge
                                                else:
                                                #RBC pushed backwards in Edge 1
                                                    if oe['sign'] == 1.0:
                                                        position1[index1]=positionPref1[-1]-oe['minDist']
                                                        if position1[index1] > 0:
                                                            positionPref1.append(position1[index1])
                                                            counterPref1.append(index)
                                                            countPref1 += 1
                                                        else:
                                                            pref1Full=1
                                                            break
                                                    else:
                                                        position1[index1]=positionPref1[-1]+oe['minDist']
                                                        if position1[index1] < oe['length']:
                                                            positionPref1.append(position1[index1])
                                                            counterPref1.append(index)
                                                            countPref1 += 1
                                                        else:
                                                            pref1Full=1
                                                            break
                                        #If the RBC fits into outEPref1
                                        else:
                                            positionPref1.append(position1[index1])
                                            counterPref1.append(index)
                                            countPref1 += 1
                                    #There are not yet any new RBCs in outEdgePref
                                    else:
                                        if oe['sign'] == 1.0:
                                            #If there are are already RBCs present in outE1
                                            #Check that there is no overtaking of RBCs
                                            if len(oe['rRBC']) > 0:
                                                if oe['rRBC'][0]-position1[index1] < oe['minDist']:
                                                    position1[index1]=oe['rRBC'][0]-oe['minDist']
                                            #There are no RBCs present in outE
                                            else:
                                                #Avoid overshooting whole vessels
                                                if position1[index1] > oe['length']:
                                                    position1[index1]=oe['length']
                                        else:
                                            if len(oe['rRBC']) > 0:
                                                if position1[index1]-oe['rRBC'][-1] <oe['minDist']:
                                                    position1[index1]=oe['rRBC'][-1]+oe['minDist']
                                            else:
                                                if position1[index1] < 0:
                                                    position1[index1]=0
                                        positionPref1.append(position1[index1])
                                        counterPref1.append(index)
                                        countPref1 += 1
                                #The RBCs do not fit into the prefered outEdge anymore
                                #Therefore they are either put in outEdge2 or outEdge3
                                elif posNoBifEventsPref2 > countPref2 and pref2Full == 0:
                                    #Check if there are already new RBCs in outEPref2
                                    if positionPref2 != []:
                                        dist2=positionPref2[-1]-position2[index2] if oe2['sign'] == 1.0 \
                                            else position2[index2]-positionPref2[-1]
                                        if dist2 < oe2['minDist']:
                                            #Check if there is a third outEdge
                                            if len(outEdges)>2:
                                                if posNoBifEventsPref3 > countPref3 and pref3Full == 0:
                                                    #Check if there are RBCs in the third outEdge
                                                    if positionPref3 != []:
                                                        dist3=positionPref3[-1]-position3[index3] if oe3['sign'] == 1.0 \
                                                            else position3[index3]-positionPref3[-1]
                                                        #if there is not enough space in the third outEdge
                                                        #Check in which edge the RBC is blocked the shortest time
                                                        if dist3 < oe3['minDist']:
                                                            #Check if another RBCs still fits into the vessel
                                                            space2 =  positionPref2[-1] if oe2['sign'] == 1.0 \
                                                                else oe2['length']-positionPref2[-1]
                                                            if np.floor(space2/oe2['minDist']) > 1:
                                                                timeBlocked2=(oe2['minDist']-dist2)/oe2['v']
                                                            else:
                                                                timeBlocked2=None
                                                                pref2Full=1
                                                            space3 =  positionPref3[-1] if oe3['sign'] == 1.0 \
                                                                else oe3['length']-positionPref3[-1]
                                                            if np.floor(space3/oe3['minDist']) > 1:
                                                                timeBlocked3=(oe3['minDist']-dist3)/oe3['v']
                                                            else:
                                                                timeBlocked3=None
                                                                pref3Full=1
                                                            if pref2Full == 1 and pref3Full == 1:
                                                                break
                                                            #Define newOutEdge
                                                            newOutEdge=0
                                                            if timeBlocked2 == None:
                                                                newOutEdge=3
                                                            elif timeBlocked3 == None:
                                                                newOutEdge=2
                                                            else:
                                                                if timeBlocked2 <= timeBlocked3:
                                                                    newOutEdge=2
                                                                else:
                                                                    newOutEdge=3
                                                            if newOutEdge == 2:
                                                                if oe2['sign'] == 1.0:
                                                                    position2[index2]=positionPref2[-1]-oe2['minDist']
                                                                    if position2[index2] > 0:
                                                                        positionPref2.append(position2[index2])
                                                                        counterPref2.append(index)
                                                                        countPref2 += 1
                                                                    else:
                                                                        print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 19')
                                                                        print(position2[index2])
                                                                        print(np.floor(space2/oe2['minDist']))
                                                                        print(timeBlocked2)
                                                                        print(pref2Full)
                                                                else:
                                                                    position2[index2]=positionPref2[-1]+oe2['minDist']
                                                                    if position2[index2] < oe2['length']:
                                                                        positionPref2.append(position2[index2])
                                                                        counterPref2.append(index)
                                                                        countPref2 += 1
                                                                    else:
                                                                        print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 20')
                                                                        print(position2[index2])
                                                                        print(oe2['length'])
                                                                        print(np.floor(space2/oe2['minDist']))
                                                            elif newOutEdge == 3:
                                                                if oe3['sign'] == 1.0:
                                                                    position3[index3]=positionPref3[-1]-oe3['minDist']
                                                                    if position3[index3] > 0:
                                                                        positionPref3.append(position3[index3])
                                                                        counterPref3.append(index)
                                                                        countPref3 += 1
                                                                    else:
                                                                        print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 21')
                                                                        print(position3[index3])
                                                                        print(np.floor(space3/oe3['minDist']))
                                                                        print(timeBlocked3)
                                                                        print(pref3Full)
                                                                else:
                                                                    position3[index3]=positionPref3[-1]+oe3['minDist']
                                                                    if position3[index3] < oe3['length']:
                                                                        positionPref3.append(position3[index3])
                                                                        counterPref3.append(index)
                                                                        countPref3 += 1
                                                                    else:
                                                                        print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 22')
                                                                        print(position3[index3])
                                                                        print(oe3['length'])
                                                                        print(np.floor(space3/oe3['minDist']))
                                                        #There is enough space in outEdge 3
                                                        else:
                                                            positionPref3.append(position3[index3])
                                                            counterPref3.append(index)
                                                            countPref3 += 1
                                                    #There are no RBCs in outEdgePref3
                                                    else:
                                                        #Check if RBC overshooted the vessel, or runs into the preceding one (which already is in the outEdge)
                                                        if oe3['sign'] == 1.0:
                                                            if len(oe3['rRBC']) > 0:
                                                                if oe3['rRBC'][0]-position3[index3] < oe3['minDist']:
                                                                    position3[index3]=oe3['rRBC'][0]-oe3['minDist']
                                                            else:
                                                                if position3[index3] > oe3['length']:
                                                                    position3[index3]=oe3['length']
                                                        else:
                                                            if len(oe3['rRBC']) > 0:
                                                                if position3[index3]-oe3['rRBC'][-1] < oe3['minDist']:
                                                                    position3[index3]=oe3['rRBC'][-1]+oe3['minDist']
                                                            else:
                                                                if position3[index3] < 0:
                                                                    position3[index3]=0
                                                        positionPref3.append(position3[index3])
                                                        counterPref3.append(index)
                                                        countPref3 += 1
                                                else:
                                                #There is no space in the third outEdge
                                                    if oe2['sign'] == 1.0:
                                                        position2[index2]=positionPref2[-1]-oe2['minDist']
                                                        if position2[index2] > 0:
                                                            positionPref2.append(position2[index2])
                                                            counterPref2.append(index)
                                                            countPref2 += 1
                                                        else:
                                                             pref2Full = 1
                                                             break
                                                    else:
                                                        position2[index2]=positionPref2[-1]+oe2['minDist']
                                                        if position2[index2] < oe2['length']:
                                                            positionPref2.append(position2[index2])
                                                            counterPref2.append(index)
                                                            countPref2 += 1
                                                        else:
                                                             break
                                            #There is no third outEdge
                                            #The RBCs are pushed backwards such that there is no overlap
                                            else:
                                                if oe2['sign'] == 1.0:
                                                    position2[index2]=positionPref2[-1]-oe2['minDist']
                                                    if position2[index2] > 0:
                                                        positionPref2.append(position2[index2])
                                                        counterPref2.append(index)
                                                        countPref2 += 1
                                                    else:
                                                         pref2Full = 1
                                                         break
                                                else:
                                                    position2[index2]=positionPref2[-1]+oe2['minDist']
                                                    if position2[index2] < oe2['length']:
                                                        positionPref2.append(position2[index2])
                                                        counterPref2.append(index)
                                                        countPref2 += 1
                                                    else:
                                                         pref2Full = 1
                                                         break
                                        #There is enough space for the RBCs in the outEdge 2
                                        else:
                                            positionPref2.append(position2[index2])
                                            counterPref2.append(index)
                                            countPref2 += 1
                                    #No RBCs have been put into outEPref2 yet
                                    else:
                                        if oe2['sign'] == 1.0:
                                            if len(oe2['rRBC']) > 0:
                                                if oe2['rRBC'][0]-position2[index2] < oe2['minDist']:
                                                    position2[index2]=oe2['rRBC'][0]-oe2['minDist']
                                            else:
                                                if position2[index2] > oe2['length']:
                                                    position2[index2]=oe2['length']
                                        else:
                                            if len(oe2['rRBC']) > 0:
                                                if position2[index2]-oe2['rRBC'][-1] < oe2['minDist']:
                                                    position2[index2]=oe2['rRBC'][-1]+oe2['minDist']
                                            else:
                                                if position2[index2] < 0:
                                                    position2[index2]=0
                                        positionPref2.append(position2[index2])
                                        counterPref2.append(index)
                                        countPref2 += 1
                                else:
                                    #Check if there is a third outEdge
                                    if len(outEdges)>2:
                                        #Check if there are RBCs in the third outEdge
                                        if posNoBifEventsPref3 > countPref3 and pref3Full == 0:
                                            if positionPref3 != []:
                                                dist3=positionPref3[-1]-position3[index3] if oe3['sign'] == 1.0 \
                                                    else position3[index3]-positionPref3[-1]
                                                #if there is not enough space in the third outEdge
                                                if dist3 < oe3['minDist']:
                                                    #Push RBCs backwards to fit
                                                    if oe3['sign'] == 1.0:
                                                        position3[index3]=positionPref3[-1]-oe3['minDist']
                                                        if position3[index3] > 0:
                                                            positionPref3.append(position3[index3])
                                                            counterPref3.append(index)
                                                            countPref3 += 1
                                                        else:
                                                             pref3Full = 1
                                                             break
                                                    else:
                                                        position3[index3]=positionPref3[-1]+oe3['minDist']
                                                        if position3[index3] < oe3['length']:
                                                            positionPref3.append(position3[index3])
                                                            counterPref3.append(index)
                                                            countPref3 += 1
                                                        else:
                                                             pref3Full = 1
                                                             break
                                                #There is enough space in outEdge 3
                                                else:
                                                    positionPref3.append(position3[index3])
                                                    counterPref3.append(index)
                                                    countPref3 += 1
                                            #No RBCs have been put in outEPref3 yet
                                            else:
                                                if oe3['sign'] == 1.0:
                                                    if len(oe3['rRBC']) > 0:
                                                        if oe3['rRBC'][0]-position3[index3] < oe3['minDist']:
                                                            position3[index3]=oe3['rRBC'][0]-oe3['minDist']
                                                    else:
                                                        if position3[index3] > oe3['length']:
                                                            position3[index3]=oe3['length']
                                                else:
                                                    if len(oe3['rRBC'])>0:
                                                        if position3[index3]-oe3['rRBC'][-1] < oe3['minDist']:
                                                            position3[index3]=oe3['rRBC'][-1]+oe3['minDist']
                                                    else:
                                                        if position3[index3] < 0:
                                                            position3[index3]=0
                                                positionPref3.append(position3[index3])
                                                counterPref3.append(index)
                                                countPref3 += 1
                                        #No space in Pref3
                                        else:
                                            break
                                    #No third out Edge
                                    else:
                                        break
                            #Add rbcs to outEPref 
                            #It has been looped over all overshoot RBCs and the number of possible overshoots hase been corrected 
                            if len(outEdges) > 2:
                                if countPref2+countPref1+countPref3 != overshootsNo:
                                    overshootsNo = countPref2+countPref1+countPref3
                            else:
                                if countPref2+countPref1 != overshootsNo:
                                    overshootsNo = countPref2+countPref1
                            #Add RBCs to outEPref1
                            if oe.index in borderEdges and e.index not in edgesBox:
                                print('All RBC indices created')
                                print('its a convergent bifurcation')
                                RBCindex=range(RBCindexCurrent+len(positionPref1),RBCindexCurrent,-1)
                                print('RBCindexCurrent')
                                print(RBCindexCurrent)
                                RBCindexCurrent += len(positionPref1)
                                print('RBCindex')
                                print(RBCindex)
                                print('RBCindexCurrent')
                                print(RBCindexCurrent)
                                edgesBoxBool=1
                            elif e.index in edgesBox and oe.index in edgesBox:
                                print('divergent bifurcation')
                                RBCindex=[]
                                print(e['RBCindex'])
                                print(e['sign'])
                                print(len(positionPref1))
                                print(counterPref1)
                                for i in range(len(positionPref1)):
                                    RBCindex.append(e['RBCindex'][counterPref1[i]])
                                if e['sign'] == 1:
                                    RBCindex=RBCindex[::-1]
                                print('RBCindex')
                                print(RBCindex)
                                edgesBoxBool=1
                            else:
                                edgesBoxBool=0
                            if oe['sign'] == 1.0:
                                oe['rRBC']=np.concatenate([positionPref1[::-1], oe['rRBC']])
                                if edgesBoxBool:
                                    oe['RBCindex']=np.concatenate([RBCindex,oe['RBCindex']])
                                    print('oe[RBCindex]')
                                    print(oe['RBCindex'])
                            else:
                                oe['rRBC']=np.concatenate([oe['rRBC'],positionPref1])
                                if edgesBoxBool:
                                    oe['RBCindex']=np.concatenate([oe['RBCindex'],RBCindex[::-1]])
                                    print('oe[RBCindex]')
                                    print(oe['RBCindex'])
                            #Add rbcs to outEPref2       
                            if oe2.index in borderEdges and e.index not in edgesBox:
                                print('divergent bifurcation 2')
                                print('All RBC indices created')
                                RBCindex2=range(RBCindexCurrent+len(positionPref2),RBCindexCurrent,-1)
                                print('RBCindexCurrent')
                                print(RBCindexCurrent)
                                RBCindexCurrent += len(positionPref2)
                                print('RBCindex')
                                print(RBCindex2)
                                print('RBCindexCurrent')
                                print(RBCindexCurrent)
                                edgesBoxBool2=1
                            elif e.index in edgesBox and oe2.index in edgesBox:
                                print('divergent bifurcation 2')
                                RBCindex2=[]
                                print(oe2['RBCindex'])
                                print(len(positionPref2))
                                print(counterPref2)
                                for i in range(len(positionPref2)):
                                    RBCindex2.append(e['RBCindex'][counterPref2[i]])
                                if e['sign'] == 1:
                                    RBCindex2=RBCindex2[::-1]
                                print('RBCindex')
                                print(RBCindex2)
                                edgesBoxBool2=1
                            else:
                                edgesBoxBool2=0
                            if oe2['sign'] == 1.0:
                                oe2['rRBC']=np.concatenate([positionPref2[::-1], oe2['rRBC']])
                                if edgesBoxBool2:
                                    oe2['RBCindex']=np.concatenate([RBCindex2,oe2['RBCindex']])
                                    print('oe[RBCindex]')
                                    print(oe2['RBCindex'])
                            else:
                                oe2['rRBC']=np.concatenate([oe2['rRBC'],positionPref2])
                                if edgesBoxBool2:
                                    oe2['RBCindex']=np.concatenate([oe2['RBCindex'],RBCindex2[::-1]])
                                    print('oe[RBCindex]')
                                    print(oe2['RBCindex'])
                            if len(outEdges) >2:
                            #Add rbcs to outEPref3       
                                if oe3.index in borderEdges and e.index in edgesBox:
                                    print('All RBC indices created')
                                    RBCindex3=range(RBCindexCurrent+len(positionPref3),RBCindexCurrent,-1)
                                    print('RBCindexCurrent')
                                    print(RBCindexCurrent)
                                    RBCindexCurrent += len(positionPref3)
                                    print('RBCindex')
                                    print(RBCindex3)
                                    print('RBCindexCurrent')
                                    print(RBCindexCurrent)
                                    edgesBoxBool3=1
                                elif e.index in edgesBox and oe3.index in edgesBox:
                                    print(e3['RBCindex'])
                                    print(len(positionPref3))
                                    print(counterPref3)
                                    RBCindex3=[]
                                    for i in range(len(positionPref3)):
                                        RBCindex3.append(e['RBCindex'][counterPref3[i]])
                                    if e['sign'] == 1:
                                        RBCindex3=RBCindex3[::-1]
                                    print('RBCindex')
                                    print(RBCindex3)
                                    edgesBoxBool3=1
                                else:
                                    edgesBoxBool3=0
                                if oe3['sign'] == 1.0:
                                    oe3['rRBC']=np.concatenate([positionPref3[::-1], oe3['rRBC']])
                                    if edgesBoxBool3:
                                        oe3['RBCindex']=np.concatenate([RBCindex3,oe3['RBCindex']])
                                        print('oe[RBCindex]')
                                        print(oe3['RBCindex'])
                                else:
                                    oe3['rRBC']=np.concatenate([oe3['rRBC'],positionPref3])
                                    if edgesBoxBool3:
                                        oe3['RBCindex']=np.concatenate([oe3['RBCindex'],RBCindex3[::-1]])
                                        print('oe[RBCindex]')
                                        print(oe3['RBCindex'])
                            #Remove RBCs from old Edge
                            if overshootsNo > 0:
                                if sign == 1.0:
                                    e['rRBC']=e['rRBC'][:-overshootsNo]
                                    if e['RBCindex'] != None:
                                        print('divergent bifurcation')
                                        print('Remove RBCs')
                                        print(e['RBCindex'])
                                        e['RBCindex']=e['RBCindex'][:-overshootsNo]
                                        print(e['RBCindex'])
                                else:
                                    e['rRBC']=e['rRBC'][overshootsNo::]
                                    if e['RBCindex'] != None:
                                        print('divergent bifurcation')
                                        print('Remove RBCs')
                                        print(e['RBCindex'])
                                        e['RBCindex']=e['RBCindex'][overshootsNo::]
                                        print(e['RBCindex'])
                            #Recheck if the distance between the newly introduces RBCs is still big enough
                            ##TODO:  not really needed needed
                            #if len(oe['rRBC']) > 0:
                            #    if len(positionPref1) == len(oe['rRBC']):
                            #        if oe['sign'] == 1.0:
                            #            iter=len(positionPref1)-2
                            #            if oe['rRBC'][-1] > oe['length']:
                            #               oe['rRBC'][-1]=oe['length']
                            #        else:
                            #            iter=len(positionPref1)-1
                            #            if oe['rRBC'][0] < 0:
                            #                oe['rRBC'][0]=0
                            #    else:
                            #        if oe['sign']==1.0:
                            #            iter=len(positionPref1)-1
                            #        else:
                            #            iter=len(positionPref1)
                            #    if oe['sign']==1.0:
                            #        for i in range(iter,-1,-1):
                            #            if oe['rRBC'][i+1]-oe['rRBC'][i]+ eps < oe['minDist'] \
                            #                or oe['rRBC'][i+1]-oe['rRBC'][i]+ eps < 0:
                            #                print('ERROR CHECK 3 should not be necessary')
                            #                print(oe['rRBC'][i+1]-oe['rRBC'][i])
                            #                print(oe['minDist'])
                            #                print(oe['rRBC'][i+1]-oe['rRBC'][i]-oe['minDist'])
                            #                print(eps*10000)
                            #                oe['rRBC'][i]=oe['rRBC'][i+1]-oe['minDist']
                            #    else:
                            #        for i in range(-1*iter,0,1):
                            #            if oe['rRBC'][i]-oe['rRBC'][i-1]+eps < oe['minDist'] \
                            #                or oe['rRBC'][i]-oe['rRBC'][i-1] + eps< 0:
                            #                print('ERROR 4 CHECK should not be necessary')
                            #                print(oe['rRBC'][i]-oe['rRBC'][i-1])
                            #                print(oe['minDist'])
                            #                print(oe['rRBC'][i]-oe['rRBC'][i-1]-oe['minDist'])
                            #                print(eps*10000)
                            #                oe['rRBC'][i]=oe['rRBC'][i-1]+oe['minDist']
                            #if len(oe2['rRBC']) > 0:
                            #    if len(positionPref2) == len(oe2['rRBC']):
                            #        if oe2['sign'] == 1.0:
                            #            iter=len(positionPref2)-2
                            #            if oe2['rRBC'][-1] > oe2['length']:
                            #                oe2['rRBC'][-1]=oe2['length']
                            #        else:
                            #            iter=len(positionPref2)-1
                            #            if oe2['rRBC'][0] < 0:
                            #                oe2['rRBC'][0]=0
                            #    else:
                            #        if oe2['sign']==1.0:
                            #            iter=len(positionPref2)-1
                            #        else:
                            #            iter=len(positionPref2)
                            #    if oe2['sign']==1.0:
                            #        for i in range(iter,-1,-1):
                            #            if oe2['rRBC'][i+1]-oe2['rRBC'][i] + eps < oe2['minDist'] \
                            #                or oe2['rRBC'][i+1]-oe2['rRBC'][i] +eps < 0:
                            #                print('ERROR CHECK 5 should not be necessary')
                            #                print(oe2['rRBC'][i+1]-oe2['rRBC'][i])
                            #                print(oe2['minDist'])
                            #                print(oe2['rRBC'][i+1]-oe2['rRBC'][i]-oe2['minDist'])
                            #                print(eps*10000)
                            #                oe2['rRBC'][i]=oe2['rRBC'][i+1]-oe2['minDist']
                            #    else:
                            #        for i in range(-1*iter,0,1):
                            #            if oe2['rRBC'][i]-oe2['rRBC'][i-1] +eps < oe2['minDist'] \
                            #                or oe2['rRBC'][i]-oe2['rRBC'][i-1] + eps < 0:
                            #                print('ERROR 6 CHECK should not be necessary')
                            #                print(oe2['rRBC'][i]-oe2['rRBC'][i-1])
                            #                print(oe2['minDist'])
                            #                print(oe2['rRBC'][i]-oe2['rRBC'][i-1]-oe2['minDist'])
                            #                print(eps*10000)
                            #                oe2['rRBC'][i]=oe2['rRBC'][i-1]+oe2['minDist']
                            ##outEPref3
                            #if len(outEdges)>2:
                            #    if len(oe3['rRBC']) > 0:
                            #        if len(positionPref3) == len(oe3['rRBC']):
                            #            if oe3['sign'] == 1.0:
                            #                iter=len(positionPref3)-2
                            #                if oe3['rRBC'][-1] > oe3['length']:
                            #                    oe3['rRBC'][-1]=oe3['length']
                            #            else:
                            #                iter=len(positionPref3)-1
                            #                if oe3['rRBC'][0] < 0:
                            #                    oe3['rRBC'][0]=0
                            #        else:
                            #            if oe3['sign']==1.0:
                            #                iter=len(positionPref3)-1
                            #            else:
                            #                iter=len(positionPref3)
                            #        if oe3['sign']==1.0:
                            #            for i in range(iter,-1,-1):
                            #                if oe3['rRBC'][i+1]-oe3['rRBC'][i] + eps < oe3['minDist'] \
                            #                    or oe3['rRBC'][i+1]-oe3['rRBC'][i] + eps < 0:
                            #                    print('ERROR CHECK 7 should not be necessary')
                            #                    print(oe3['rRBC'][i+1]-oe3['rRBC'][i])
                            #                    print(oe3['minDist'])
                            #                    print(oe3['rRBC'][i+1]-oe3['rRBC'][i]-oe3['minDist'])
                            #                    print(eps*10000)
                            #                    oe3['rRBC'][i]=oe3['rRBC'][i+1]-oe3['minDist']
                            #        else:
                            #            for i in range(-1*iter,0,1):
                            #                if oe3['rRBC'][i]-oe3['rRBC'][i-1] + eps < oe3['minDist'] \
                            #                    or oe3['rRBC'][i]-oe3['rRBC'][i-1] + eps < 0:
                            #                    print('ERROR 8 CHECK should not be necessary')
                            #                    print(oe3['rRBC'][i]-oe3['rRBC'][i-1])
                            #                    print(oe3['minDist'])
                            #                    print(oe3['rRBC'][i]-oe3['rRBC'][i-1]-oe3['minDist'])
                            #                    print(eps*10000)
                            #                    oe3['rRBC'][i]=oe3['rRBC'][i-1]+oe3['minDist']
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        noStuckRBCs=len(bifRBCsIndex)-overshootsNo
                        for i in range(noStuckRBCs):
                            index=-1*(i+1) if sign == 1.0 else i
                            e['rRBC'][index]=e['length']-i*e['minDist'] if sign == 1.0 else 0+i*e['minDist']
                        #Recheck if the distance between the newly introduces RBCs is still big enough 
                        if len(e['rRBC']) >1:
                            moved = 0
                            count = 0
                            if sign == 1.0:
                                for i in range(-1,-1*(len(e['rRBC'])),-1):
                                    index=i-1
                                    if e['rRBC'][i] < e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]-e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs and moved == 0:
                                        break
                            else:
                                for i in range(len(e['rRBC'])-1):
                                    index=i+1
                                    if e['rRBC'][i] > e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]+e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs+1 and moved == 0:
                                        break
    #-------------------------------------------------------------------------------------------
                #if vertex is convergent vertex
                    elif G.vs[vi]['vType'] == 4:
                        #print('at convergent vertex')
                        bifRBCsIndex1=bifRBCsIndex
                        noBifEvents1=noBifEvents
                        outE=G.vs[vi]['outflowE'][0]
                        oe = G.es[outE]
                        inflowEdges=G.vs[vi]['inflowE']
                        k=0
                        for i in inflowEdges:
                            if i == e.index:
                                inE1=e.index
                            else:
                                if k == 0:
                                    inE2=i
                                    k = 1
                                else:
                                    inE3=i
                        e2=G.es[inE2]
                        #Move RBCs in second inEdge (if that has not been done already)
                        if inE2 not in convEdges2:
                            convEdges2.append(inE2)
                            #Check if httBC exists
                            if e2['httBC'] is not None:
                                boolHttEdge2 = 1
                                rRBC = []
                                RBCindex = []
                                lrbc = e2['minDist']
                                htt = e2['httBC']
                                length = e2['length']
                                nMaxNew=e2['nMax']-len(e2['rRBC'])
                                if len(e2['rRBC']) > 0:
                                    #if cum_length > distToFirst:
                                    posFirst=e2['rRBC'][0] if e2['sign']==1.0 else e2['length']-e2['rRBC'][-1]
                                    cum_length = posFirst
                                else:
                                    cum_length = e2['posFirst_last'] + e2['v_last'] * dt
                                    posFirst = cum_length
                                while cum_length >= lrbc and nMaxNew > 0:
                                    if len(e2['keep_rbcs']) != 0:
                                        if posFirst - e2['keep_rbcs'][0] >= 0:
                                            rRBC.append(posFirst - e2['keep_rbcs'][0])
                                            nMaxNew += -1
                                            posFirst=posFirst - e2['keep_rbcs'][0]
                                            cum_length = posFirst
                                            e2['keep_rbcs']=[]
                                            e2['posFirst_last']=posFirst
                                            e2['v_last']=e2['v']
                                            if e.index in edgesBox:
                                                RBCindexCurrent += 1
                                                RBCindex.append(RBCindexCurrent)
                                        else:
                                            if len(e2['rRBC']) > 0:
                                                e2['posFirst_last'] = posFirst
                                                e2['v_last']=e2['v']
                                            else:
                                                e2['posFirst_last'] += e2['v_last'] * dt
                                            break
                                    else:
                                        #number of RBCs randomly chosen to average htt
                                        number=np.exp(self._mu+self._sigma*np.random.randn(1)[0])
                                        #self._spacing.append(number)
                                        spacing = lrbc+lrbc*number
                                        if posFirst - spacing >= 0:
                                            rRBC.append(posFirst - spacing)
                                            nMaxNew += -1
                                            posFirst=posFirst - spacing
                                            cum_length = posFirst
                                            e2['posFirst_last']=posFirst
                                            e2['v_last']=e2['v']
                                            if e.index in edgesBox:
                                                RBCindexCurrent += 1
                                                RBCindex.append(RBCindexCurrent)
                                        else:
                                            e2['keep_rbcs']=[spacing]
                                            e2['v_last']=e2['v']
                                            if len(rRBC) == 0:
                                                e2['posFirst_last']=posFirst
                                            else:
                                                e2['posFirst_last']=rRBC[-1]
                                            break
                                rRBC = np.array(rRBC)
                                RBCindex=np.array(RBCindex)
                                if len(rRBC) >= 1.:
                                    if e2['sign'] == 1:
                                        e2['rRBC'] = np.concatenate([rRBC[::-1], e2['rRBC']])
                                        if e2['RBCindex'] != None:
                                            e2['RBCindex']=np.concatenate([RBCindex[::-1], e2['RBCindex']])
                                    else:
                                        e2['rRBC'] = np.concatenate([e2['rRBC'], length-rRBC])
                                        if e2['RBCindex'] != None:
                                            e2['RBCindex']=np.concatenate([e2['RBCindex'],RBCindex])
                                    vertexUpdate.append(e.target)
                                    vertexUpdate.append(e.source)
                                    edgeUpdate.append(e2.index)
                            #If RBCs are present move all RBCs in inEdge2
                            if len(e2['rRBC']) > 0:
                                e2['rRBC'] = e2['rRBC'] + e2['v'] * dt * e2['sign']
                                bifRBCsIndex2=[]
                                nRBC2=len(e2['rRBC'])
                                if e2['sign'] == 1.0:
                                    if e2['rRBC'][-1] > e2['length']:
                                        for i,j in enumerate(e2['rRBC'][::-1]):
                                            if j > e2['length']:
                                                bifRBCsIndex2.append(nRBC2-1-i)
                                            else:
                                                break
                                    bifRBCsIndex2=bifRBCsIndex2[::-1]
                                else:
                                    if e2['rRBC'][0] < 0:
                                        for i,j in enumerate(e2['rRBC']):
                                            if j < 0:
                                                bifRBCsIndex2.append(i)
                                            else:
                                                break
                                noBifEvents2=len(bifRBCsIndex2)
                            else:
                                bifRBCsIndex2=[]
                                noBifEvents2=0
                            sign2=e2['sign']
                        else:
                            noBifEvents2=0
                            bifRBCsIndex2=[]
                            sign2=e2['sign']
                        #Check if there is a third inEdge
                        if len(inflowEdges) > 2:
                            e3=G.es[inE3]
                            if inE3 not in convEdges2:
                                convEdges2.append(inE3)
                                #Check if httBC exists
                                if e3['httBC'] is not None:
                                    boolHttEdge3 = 1
                                    rRBC = []
                                    RBCindex = []
                                    lrbc = e3['minDist']
                                    htt = e3['httBC']
                                    length = e3['length']
                                    nMaxNew=e3['nMax']-len(e3['rRBC'])
                                    if len(e3['rRBC']) > 0:
                                        #if cum_length > distToFirst:
                                        posFirst=e3['rRBC'][0] if e3['sign']==1.0 else e3['length']-e3['rRBC'][-1]
                                        cum_length = posFirst
                                    else:
                                        cum_length = e3['posFirst_last'] + e3['v_last'] * dt
                                        posFirst = cum_length
                                    while cum_length >= lrbc and nMaxNew > 0:
                                        if len(e3['keep_rbcs']) != 0:
                                            if posFirst - e3['keep_rbcs'][0] >= 0:
                                                rRBC.append(posFirst - e3['keep_rbcs'][0])
                                                nMaxNew += -1
                                                posFirst=posFirst - e3['keep_rbcs'][0]
                                                cum_length = posFirst
                                                e3['keep_rbcs']=[]
                                                e3['posFirst_last']=posFirst
                                                e3['v_last']=e3['v']
                                                if e.index in edgesBox:
                                                    RBCindexCurrent += 1
                                                    RBCindex.append(RBCindexCurrent)
                                            else:
                                                if len(e3['rRBC']) > 0:
                                                    e3['posFirst_last'] = posFirst
                                                    e3['v_last']=e3['v']
                                                else:
                                                    e3['posFirst_last'] += e3['v_last'] * dt
                                                break
                                        else:
                                            #number of RBCs randomly chosen to average htt
                                            number=np.exp(self._mu+self._sigma*np.random.randn(1)[0])
                                            #self._spacing.append(number)
                                            spacing = lrbc+lrbc*number
                                            if posFirst - spacing >= 0:
                                                rRBC.append(posFirst - spacing)
                                                nMaxNew += -1
                                                posFirst=posFirst - spacing
                                                cum_length = posFirst
                                                e3['posFirst_last']=posFirst
                                                e3['v_last']=e3['v']
                                                if e.index in edgesBox:
                                                    RBCindexCurrent += 1
                                                    RBCindex.append(RBCindexCurrent)
                                            else:
                                                e3['keep_rbcs']=[spacing]
                                                e3['v_last']=e3['v']
                                                if len(rRBC) == 0:
                                                    e3['posFirst_last']=posFirst
                                                else:
                                                    e3['posFirst_last']=rRBC[-1]
                                                break
                                    rRBC = np.array(rRBC)
                                    RBCindex=np.array(RBCindex)
                                    if len(rRBC) >= 1.:
                                        if e3['sign'] == 1:
                                            e3['rRBC'] = np.concatenate([rRBC[::-1], e3['rRBC']])
                                            if e3['RBCindex'] != None:
                                                e3['RBCindex']=np.concatenate([RBCindex[::-1], e3['RBCindex']])
                                        else:
                                            e3['rRBC'] = np.concatenate([e3['rRBC'], length-rRBC])
                                            if e3['RBCindex'] != None:
                                                e3['RBCindex']=np.concatenate([e3['RBCindex'],RBCindex])
                                        vertexUpdate.append(e.target)
                                        vertexUpdate.append(e.source)
                                        edgeUpdate.append(e3.index)
                                #If RBCs are present move all RBCs in inEdge3
                                if len(e3['rRBC']) > 0:
                                    e3['rRBC'] = e3['rRBC'] + e3['v'] * dt * e3['sign']
                                    bifRBCsIndex3=[]
                                    nRBC3=len(e3['rRBC'])
                                    if e3['sign'] == 1.0:
                                        if e3['rRBC'][-1] > e3['length']:
                                            for i,j in enumerate(e3['rRBC'][::-1]):
                                                if j > e3['length']:
                                                    bifRBCsIndex3.append(nRBC3-1-i)
                                                else:
                                                    break
                                        bifRBCsIndex3=bifRBCsIndex3[::-1]
                                    else:
                                        if e3['rRBC'][0] < 0:
                                            for i,j in enumerate(e3['rRBC']):
                                                if j < 0:
                                                    bifRBCsIndex3.append(i)
                                                else:
                                                    break
                                    noBifEvents3=len(bifRBCsIndex3)
                                else:
                                    bifRBCsIndex3=[]
                                    noBifEvents3=0
                                sign3=e3['sign']
                            else:
                                bifRBCsIndex3=[]
                                sign3=e3['sign']
                                noBifEvents3=0
                        else:
                            bifRBCsIndex3=[]
                            noBifEvents3=0
                        #Calculate distance to first RBC in outEdge
                        if len(G.es[outE]['rRBC']) > 0:
                            distToFirst=G.es[outE]['rRBC'][0] if G.es[outE]['sign'] == 1.0 else G.es[outE]['length']-G.es[outE]['rRBC'][-1]
                        else:
                            distToFirst=G.es[outE]['length']
                        posNoBifEvents=int(np.floor(distToFirst/G.es[outE]['minDist']))
                        if posNoBifEvents + len(G.es[outE]['rRBC']) > G.es[outE]['nMax']:
                            posNoBifEvents = G.es[outE]['nMax'] - len(G.es[outE]['rRBC'])
                        #If bifurcations are possible check how many overshoots there are at the inEdges
                        if posNoBifEvents > 0:
                            #overshootDist starts with the RBC which overshoots the least
                            overshootDist1=[e['rRBC'][bifRBCsIndex1]-[e['length']]*noBifEvents1 if sign == 1.0
                                else [0]*noBifEvents1-e['rRBC'][bifRBCsIndex1]][0]
                            if sign != 1.0:
                                overshootDist1 = overshootDist1[::-1]
                            overshootTime1=np.array(overshootDist1 / ([e['v']]*noBifEvents1))
                            dummy1=np.array([1]*len(overshootTime1))
                            if noBifEvents2 > 0:
                                #overshootDist starts with the RBC which overshoots the least
                                overshootDist2=[e2['rRBC'][bifRBCsIndex2]-[e2['length']]*noBifEvents2 if sign2 == 1.0
                                    else [0]*noBifEvents2-e2['rRBC'][bifRBCsIndex2]][0]
                                if sign2 != 1.0:
                                    overshootDist2 = overshootDist2[::-1]
                                overshootTime2=np.array(overshootDist2)/ np.array([e2['v']]*noBifEvents2)
                                dummy2=np.array([2]*len(overshootTime2))
                            else:
                                overshootDist2=[]
                                overshootTime2=[]
                                dummy2=[]
                            if len(inflowEdges) > 2:
                                if noBifEvents3 > 0:
                                    overshootDist3=[e3['rRBC'][bifRBCsIndex3]-[e3['length']]*noBifEvents3 if sign3 == 1.0
                                        else [0]*noBifEvents3-e3['rRBC'][bifRBCsIndex3]][0]
                                    if sign3 != 1.0:
                                        overshootDist3 = overshootDist3[::-1]
                                    overshootTime3=np.array(overshootDist3)/ np.array([e3['v']]*noBifEvents3)
                                    dummy3=np.array([3]*len(overshootTime3))
                                else:
                                    overshootDist3=[]
                                    overshootTime3=[]
                                    dummy3=[]
                            else:
                                overshootDist3=[]
                                overshootTime3=[]
                                dummy3=[]
                            #Define which RBC arrive first, second, .. at convergent bifurcation
                            overshootTimes=zip(np.concatenate([overshootTime1,overshootTime2,overshootTime3]),np.concatenate([dummy1,dummy2,dummy3]))
                            overshootTimes.sort()
                            overshootTime=[]
                            inEdge=[]
                            count1=0
                            count2=0
                            count3=0
                            if posNoBifEvents > len(overshootTimes):
                                overshootsNo=int(len(overshootTimes))
                            else:
                                overshootsNo=int(posNoBifEvents)
                            #position rbcs based on when they appear at bifurcation
                            for i in range(-1*overshootsNo,0):
                                overshootTime.append(overshootTimes[i][0])
                                inEdge.append(overshootTimes[i][1])
                            #Numbers of RBCs from corresponding inEdge
                            count1=inEdge.count(1)
                            count2=inEdge.count(2)
                            count3=inEdge.count(3)
                            #position starts with least overshooting RBC and ends with highest overshooting RBC
                            position=np.array(overshootTime)*np.array([G.es[outE]['v']]*overshootsNo)
                            #Check if RBCs are to close to each other
                            #Check if the RBCs runs into an old one in the vessel
                            #(only position of the leading RBC is changed)
                            if len(G.es[outE]['rRBC']) > 0:
                                if G.es[outE]['sign'] == 1.0:
                                    if position[-1] > G.es[outE]['rRBC'][0]-G.es[outE]['minDist']:
                                        position[-1]=G.es[outE]['rRBC'][0]-G.es[outE]['minDist']
                                else:
                                    if G.es[outE]['length']-position[-1] < G.es[outE]['rRBC'][-1]+G.es[outE]['minDist']:
                                        position[-1]=G.es[outE]['length']-(G.es[outE]['rRBC'][-1]+G.es[outE]['minDist'])
                            else:
                                #Check if the RBCs overshooted the vessel
                                if position[-1] > G.es[outE]['length']:
                                    position[-1]=G.es[outE]['length']
                            #Position of the following RBCs is changed, such that they do not overlap
                            for i in range(-1,-1*(count1+count2+count3),-1):
                                if position[i]-position[i-1] < G.es[outE]['minDist'] or \
                                    position[i-1] > position[i]:
                                    position[i-1]=position[i]-G.es[outE]['minDist']
                            #if first RBC did not yet move enough less than the possible no of RBCs fit into the outEdge
                            for i in range(count1+count2+count3):
                                if position[i] < 0:
                                    if inEdge[i] == 1:
                                        count1 += -1
                                    elif inEdge[i] == 2:
                                        count2 += -1
                                    elif inEdge[i] == 3:
                                        count3 += -1
                                else:
                                    break
                            if len(position) != len(position[i::]):
                                position=position[i::]
                                inEdge=inEdge[i::]
                            #Add rbcs to outE 
                            if len(inflowEdges) > 2:
                                if oe.index in borderEdges and e.index not in edgesBox and e2.index not in edgesBox and e3.index not in edgesBox:
                                    print('All RBC indices created')
                                    print('its a convergent bifurcation')
                                    RBCindex=range(RBCindexCurrent+len(position),RBCindexCurrent,-1)
                                    print('RBCindexCurrent')
                                    print(RBCindexCurrent)
                                    RBCindexCurrent += len(position)
                                    print('RBCindex')
                                    print(RBCindex)
                                    print('RBCindexCurrent')
                                    print(RBCindexCurrent)
                                    edgesBoxBool=1
                                elif oe.index in edgesBox:
                                    if e.index in edgesBox:
                                        if e['sign'] == 1:
                                            RBCindex1=e['RBCindex'][-1*count1::]
                                        else:
                                            RBCindex1=e['RBCindex'][0:count1]
                                            RBCindex1=RBCindex1[::-1]
                                    else:
                                        RBCindex1=range(RBCindexCurrent+count1,RBCindexCurrent,-1)
                                        RBCindexCurrent += count1
                                    if e2.index in edgesBox:
                                        if e2['sign'] == 1:
                                            RBCindex2=e2['RBCindex'][-1*count2::]
                                        else:
                                            RBCindex2=e2['RBCindex'][0:count2]
                                            RBCindex2=RBCindex2[::-1]
                                    else:
                                        RBCindex2=range(RBCindexCurrent+count2,RBCindexCurrent,-1)
                                        RBCindexCurrent += count2
                                    if e3.index in edgesBox:
                                        if e3['sign'] == 1:
                                            RBCindex3=e3['RBCindex'][-1*count3::]
                                        else:
                                            RBCindex3=e['RBCindex'][0:count3]
                                            RBCindex3=RBCindex3[::-1]
                                    else:
                                        RBCindex3=range(RBCindexCurrent+count3,RBCindexCurrent,-1)
                                        RBCindexCurrent += count3
                                    print('its a convergent bifurcation')
                                    print(RBCindex1)
                                    print(RBCindex2)
                                    print(RBCindex3)
                                    RBCindex=[]
                                    index=[0,0,0]
                                    for i in range(len(position)):
                                        iE=inEdge[i]-1
                                        if inEdge[i] == 1:
                                            RBCindexList=RBCindex1
                                        elif inEdge[i] == 2:
                                            RBCindexList=RBCindex2
                                        else:
                                            RBCindexList=RBCindex3
                                        RBCindex.append(RBCindexList[-(index[iE]+1)])
                                        index[iE] = index[iE] + 1
                                    edgesBoxBool=1
                                    print('RBCindex')
                                    print(RBCindex)
                                else:
                                    edgesBoxBool=0
                            else:
                                if oe.index in borderEdges and e.index not in edgesBox and e2.index not in edgesBox:
                                    print('All RBC indices created')
                                    print('its a convergent bifurcation')
                                    RBCindex=range(RBCindexCurrent+len(position),RBCindexCurrent,-1)
                                    print('RBCindexCurrent')
                                    print(RBCindexCurrent)
                                    RBCindexCurrent += len(position)
                                    print('RBCindex')
                                    print(RBCindex)
                                    print('RBCindexCurrent')
                                    print(RBCindexCurrent)
                                    edgesBoxBool=1
                                elif oe.index in edgesBox:
                                    if e.index in edgesBox:
                                        if e['sign'] == 1:
                                            RBCindex1=e['RBCindex'][-1*count1::]
                                        else:
                                            RBCindex1=e['RBCindex'][0:count1]
                                            RBCindex1=RBCindex1[::-1]
                                    else:
                                        RBCindex1=range(RBCindexCurrent+count1,RBCindexCurrent,-1)
                                        RBCindexCurrent += count1
                                    if e2.index in edgesBox:
                                        if e2['sign'] == 1:
                                            RBCindex2=e2['RBCindex'][-1*count2::]
                                        else:
                                            RBCindex2=e2['RBCindex'][0:count2]
                                            RBCindex2=RBCindex2[::-1]
                                    else:
                                        RBCindex2=range(RBCindexCurrent+count2,RBCindexCurrent,-1)
                                        RBCindexCurrent += count2
                                    print('its a convergent bifurcation')
                                    print(RBCindex1)
                                    print(RBCindex2)
                                    RBCindex=[]
                                    index=[0,0]
                                    for i in range(len(position)):
                                        iE=inEdge[i]-1
                                        if inEdge[i] == 1:
                                            RBCindexList=RBCindex1
                                        elif inEdge[i] == 2:
                                            RBCindexList=RBCindex2
                                        RBCindex.append(RBCindexList[-(index[iE]+1)])
                                        index[iE] = index[iE] + 1
                                    edgesBoxBool=1
                                    print('RBCindex')
                                    print(RBCindex)
                                else:
                                    edgesBoxBool=0
			    #Add RBCs
                            if oe['sign'] == 1.0:
                                oe['rRBC']=np.concatenate([position, oe['rRBC']])
                                if edgesBoxBool:
                                    oe['RBCindex']=np.concatenate([RBCindex,oe['RBCindex']])
                                    print('oe[RBCindex]')
                                    print(oe['RBCindex'])
                            else:
                                position = [oe['length']]*len(position) - position[::-1]
                                oe['rRBC']=np.concatenate([oe['rRBC'],position])
                                if edgesBoxBool:
                                    oe['RBCindex']=np.concatenate([oe['RBCindex'],RBCindex[::-1]])
                                    print('oe[RBCindex]')
                                    print(oe['RBCindex'])
                            #Remove RBCs from old Edge 1
                            if count1 > 0:
                                if sign == 1.0:
                                    e['rRBC']=e['rRBC'][:-count1]
                                    if e['RBCindex'] != None:
                                        print('its a convergent bifurcation')
                                        print('Remove RBCs')
                                        print(e['RBCindex'])
                                        e['RBCindex']=e['RBCindex'][:-count1]
                                        print(e['RBCindex'])
                                else:
                                    e['rRBC']=e['rRBC'][count1::]
                                    if e['RBCindex'] != None:
                                        print('its a convergent bifurcation')
                                        print('Remove RBCs')
                                        print(e['RBCindex'])
                                        e['RBCindex']=e['RBCindex'][count1::]
                                        print(e['RBCindex'])
                            if noBifEvents2 > 0 and count2 > 0:
                                #Remove RBCs from old Edge 2
                                if sign2 == 1.0:
                                    e2['rRBC']=e2['rRBC'][:-count2]
                                    if e2['RBCindex'] != None:
                                        print('its a convergent bifurcation')
                                        print('Remove RBCs 2')
                                        print(e2['RBCindex'])
                                        e2['RBCindex']=e2['RBCindex'][:-count2]
                                        print(e2['RBCindex'])
                                else:
                                    e2['rRBC']=e2['rRBC'][count2::]
                                    if e2['RBCindex'] != None:
                                        print('its a convergent bifurcation')
                                        print('Remove RBCs 2')
                                        print(e2['RBCindex'])
                                        e2['RBCindex']=e2['RBCindex'][count2::]
                                        print(e2['RBCindex'])
                            if len(inflowEdges) > 2:
                                if noBifEvents3 > 0 and count3 > 0:
                                    #Remove RBCs from old Edge 3
                                    if sign3 == 1.0:
                                        e3['rRBC']=e3['rRBC'][:-count3]
                                        if e3['RBCindex'] != None:
                                            print('its a convergent bifurcation')
                                            print('Remove RBCs 3')
                                            print(e3['RBCindex'])
                                            e3['RBCindex']=e3['RBCindex'][:-count3]
                                            print(e3['RBCindex'])
                                    else:
                                        e3['rRBC']=e3['rRBC'][count3::]
                                        if e3['RBCindex'] != None:
                                            print('its a convergent bifurcation')
                                            print('Remove RBCs 3')
                                            print(e3['RBCindex'])
                                            e3['RBCindex']=e3['RBCindex'][count3::]
                                            print(e3['RBCindex'])
                            overshootsNo = count1 + count2 + count3
                        else:
                            count1=0
                            count2=0
                            count3=0
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        #InEdge 1
                        noStuckRBCs1=len(bifRBCsIndex1)-count1
                        for i in range(noStuckRBCs1):
                            index=-1*(i+1) if sign == 1.0 else i
                            e['rRBC'][index]=e['length']-i*e['minDist'] if sign == 1.0 else 0+i*e['minDist']
                        #Recheck if the distance between the newly introduces RBCs is still big enough 
                        if len(e['rRBC']) >1.0:
                            moved = 0
                            count = 0
                            if sign == 1.0:
                                for i in range(-1,-1*(len(e['rRBC'])),-1):
                                    index=i-1
                                    if e['rRBC'][i] < e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]-e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs1 and moved == 0:
                                        break
                            else:
                                for i in range(len(e['rRBC'])-1):
                                    index=i+1
                                    if e['rRBC'][i] > e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]+e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs1+1 and moved == 0:
                                        break
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        #InEdge 2
                        noStuckRBCs2=len(bifRBCsIndex2)-count2
                        for i in range(noStuckRBCs2):
                            index=-1*(i+1) if sign2 == 1.0 else i
                            e2['rRBC'][index]=e2['length']-i*e2['minDist'] if sign2 == 1.0 else 0+i*e2['minDist']
                        #Recheck if the distance between the newly introduces RBCs is still big enough 
                        if len(e2['rRBC']) >1:
                            moved = 0
                            count = 0
                            if sign2 == 1.0:
                                for i in range(-1,-1*(len(e2['rRBC'])),-1):
                                    index=i-1
                                    if e2['rRBC'][i] < e2['rRBC'][index] or abs(e2['rRBC'][i]-e2['rRBC'][index]) < e2['minDist']:
                                        e2['rRBC'][index]=e2['rRBC'][i]-e2['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs2 and moved == 0:
                                        break
                            else:
                                for i in range(len(e2['rRBC'])-1):
                                    index=i+1
                                    if e2['rRBC'][i] > e2['rRBC'][index] or abs(e2['rRBC'][i]-e2['rRBC'][index]) < e2['minDist']:
                                        e2['rRBC'][index]=e2['rRBC'][i]+e2['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs2+1 and moved == 0:
                                        break
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        #InEdge 3
                        if len(inflowEdges) > 2:
                            noStuckRBCs3=len(bifRBCsIndex3)-count3
                            for i in range(noStuckRBCs3):
                                index=-1*(i+1) if sign3 == 1.0 else i
                                e3['rRBC'][index]=e3['length']-i*e3['minDist'] if sign3 == 1.0 else 0+i*e3['minDist']
                            #Recheck if the distance between the newly introduces RBCs is still big enough 
                            if len(e3['rRBC']) >1:
                                moved = 0
                                count = 0
                                if sign3 == 1.0:
                                    for i in range(-1,-1*(len(e3['rRBC'])),-1):
                                        index=i-1
                                        if e3['rRBC'][i] < e3['rRBC'][index] or abs(e3['rRBC'][i]-e3['rRBC'][index]) < e3['minDist']:
                                            e3['rRBC'][index]=e3['rRBC'][i]-e3['minDist']
                                            moved = 1
                                        else:
                                            moved = 0
                                        count += 1
                                        if count >= noStuckRBCs3 and moved == 0:
                                            break
                                else:
                                    for i in range(len(e3['rRBC'])-1):
                                        index=i+1
                                        if e3['rRBC'][i] > e3['rRBC'][index] or abs(e3['rRBC'][i]-e3['rRBC'][index]) < e3['minDist']:
                                            e3['rRBC'][index]=e3['rRBC'][i]+e3['minDist']
                                            moved = 1
                                        else:
                                            moved = 0
                                        count += 1
                                        if count >= noStuckRBCs3+1 and moved == 0:
                                            break
         #------------------------------------------------------------------------------------------
                    #if vertex is double connecting vertex
                    elif G.vs[vi]['vType'] == 6:
                        #print('at double connecting vertex')
                        bifRBCsIndex1=bifRBCsIndex
                        noBifEvents1=noBifEvents
                        outE=G.vs[vi]['outflowE'][0]
                        inflowEdges=G.vs[vi]['inflowE']
                        for i in inflowEdges:
                            if i == e.index:
                                inE1=e.index
                            else:
                                inE2=i
                        e2=G.es[inE2]
                        if inE2 not in convEdges2:
                            convEdges2.append(inE2)
                            #Check if httBC exists
                            if e2['httBC'] is not None:
                                boolHttEdge2 = 1
                                rRBC = []
                                RBCindex = []
                                lrbc = e2['minDist']
                                htt = e2['httBC']
                                length = e2['length']
                                nMaxNew=e2['nMax']-len(e2['rRBC'])
                                if len(e2['rRBC']) > 0:
                                    #if cum_length > distToFirst:
                                    posFirst=e2['rRBC'][0] if e2['sign']==1.0 else e2['length']-e2['rRBC'][-1]
                                    e2['posFirst_last']=posFirst
                                    e2['v_last'] = e2['v']
                                    cum_length = posFirst
                                else:
                                    cum_length = e2['posFirst_last'] + e2['v_last'] * dt
                                    posFirst = cum_length
                                    e2['posFirst_last']=posFirst
                                    e2['v_last']=e2['v']
                                while cum_length >= lrbc and nMaxNew > 0:
                                    if len(e2['keep_rbcs']) != 0:
                                        if posFirst - e2['keep_rbcs'][0] >= 0:
                                            rRBC.append(posFirst - e2['keep_rbcs'][0])
                                            nMaxNew += -1
                                            posFirst=posFirst - e2['keep_rbcs'][0]
                                            cum_length = posFirst
                                            e2['keep_rbcs']=[]
                                            e2['posFirst_last']=posFirst
                                            e2['v_last']=e2['v']
                                            if e.index in edgesBox:
                                                RBCindexCurrent += 1
                                                RBCindex.append(RBCindexCurrent)
                                        else:
                                            if len(e2['rRBC']) > 0:
                                                e2['posFirst_last'] = posFirst
                                                e2['v_last']=e2['v']
                                            else:
                                                e2['posFirst_last'] += e2['v_last'] * dt
                                            break
                                    else:
                                        #number of RBCs randomly chosen to average htt
                                        number=np.exp(e2['logNormal'][0]+e2['logNormal'][1]*np.random.randn(1)[0])
                                        #self._spacing.append(number)
                                        spacing = lrbc+lrbc*number
                                        if posFirst - spacing >= 0:
                                            rRBC.append(posFirst - spacing)
                                            nMaxNew += -1
                                            posFirst=posFirst - spacing
                                            cum_length = posFirst
                                            e2['posFirst_last']=posFirst
                                            e2['v_last']=e2['v']
                                            if e.index in edgesBox:
                                                RBCindexCurrent += 1
                                                RBCindex.append(RBCindexCurrent)
                                        else:
                                            e2['keep_rbcs']=[spacing]
                                            e2['v_last']=e2['v']
                                            if len(rRBC) == 0:
                                                e2['posFirst_last']=posFirst
                                            else:
                                                e2['posFirst_last']=rRBC[-1]
                                            break
                                rRBC = np.array(rRBC)
                                RBCindex=np.array(RBCindex)
                                if len(rRBC) >= 1.:
                                    if e2['sign'] == 1:
                                        e2['rRBC'] = np.concatenate([rRBC[::-1], e2['rRBC']])
                                        if e2['RBCindex'] != None:
                                            e2['RBCindex']=np.concatenate([RBCindex[::-1], e2['RBCindex']])
                                    else:
                                        e2['rRBC'] = np.concatenate([e2['rRBC'], length-rRBC])
                                        if e2['RBCindex'] != None:
                                            e2['RBCindex']=np.concatenate([e2['RBCindex'],RBCindex])
                                    vertexUpdate.append(e.target)
                                    vertexUpdate.append(e.source)
                                    edgeUpdate.append(e2.index)
                            #If RBCs are present move all RBCs in inEdge2
                            if len(e2['rRBC']) > 0:
                                e2['rRBC'] = e2['rRBC'] + e2['v'] * dt * e2['sign']
                                bifRBCsIndex2=[]
                                nRBC2=len(e2['rRBC'])
                                if e2['sign'] == 1.0:
                                    if e2['rRBC'][-1] > e2['length']:
                                        for i,j in enumerate(e2['rRBC'][::-1]):
                                            if j > e2['length']:
                                                bifRBCsIndex2.append(nRBC2-1-i)
                                            else:
                                                break
                                    bifRBCsIndex2=bifRBCsIndex2[::-1]
                                else:
                                    if e2['rRBC'][0] < 0:
                                        for i,j in enumerate(e2['rRBC']):
                                            if j < 0:
                                                bifRBCsIndex2.append(i)
                                            else:
                                                break
                                noBifEvents2=len(bifRBCsIndex2)
                            else:
                                noBifEvents2=0
                                bifRBCsIndex2=[]
                        else:
                            bifRBCsIndex2=[]
                            noBifEvents2=0
                        sign2=e2['sign']
                        #Define outEdges
                        outEdges=G.vs[vi]['outflowE']
                        outE=outEdges[0]
                        outE2=outEdges[1]
                        #Differ between capillaries and non-capillaries
                        if G.vs[vi]['isCap']:
                            preferenceList = [x[1] for x in sorted(zip(np.array(G.es[outEdges]['flow'])/np.array(G.es[outEdges]['crosssection']), outEdges), reverse=True)]
                        else:
                            preferenceList = [x[1] for x in sorted(zip(G.es[outEdges]['flow'], outEdges), reverse=True)]
                        #Define prefered OutEdges
                        outEPref=preferenceList[0]
                        outEPref2=preferenceList[1]
                        oe=G.es[outEPref]
                        oe2=G.es[outEPref2]
                        #Calculate distance to first RBC
                        if len(oe['rRBC']) > 0:
                            distToFirst=oe['rRBC'][0] if oe['sign'] == 1.0 else oe['length']-oe['rRBC'][-1]
                        else:
                            distToFirst=oe['length']
                        if len(oe2['rRBC']) > 0:
                            distToFirst2=oe2['rRBC'][0] if oe2['sign'] == 1.0 else oe2['length']-oe2['rRBC'][-1]
                        else:
                            distToFirst2=oe2['length']
                        #Check how many RBCs are allowed by nMax
                        posNoBifEventsPref=np.floor(distToFirst/oe['minDist'])
                        if posNoBifEventsPref + len(oe['rRBC']) > oe['nMax']:
                            posNoBifEventsPref = oe['nMax'] - len(oe['rRBC'])
                        posNoBifEventsPref2=np.floor(distToFirst2/oe2['minDist'])
                        if posNoBifEventsPref2 + len(oe2['rRBC']) > oe2['nMax']:
                            posNoBifEventsPref2 = oe2['nMax'] - len(oe2['rRBC'])
                        #Check how many RBCs fit into the new Vessel
                        posNoBifEvents=int(posNoBifEventsPref+posNoBifEventsPref2)
                        #Calculate number of bifEvents
                        #If bifurcations are possible check how many overshoots there are at the inEdges
                        if posNoBifEvents > 0:
                            overshootDist1=[e['rRBC'][bifRBCsIndex1]-[e['length']]*noBifEvents1 if sign == 1.0
                                else [0]*noBifEvents1-e['rRBC'][bifRBCsIndex1]][0]
                            if sign != 1.0:
                                overshootDist1 = overshootDist1[::-1]
                            overshootTime1=np.array(overshootDist1 / ([e['v']]*noBifEvents1))
                            dummy1=np.array([1]*len(overshootTime1))
                            if noBifEvents2 > 0:
                                overshootDist2=[e2['rRBC'][bifRBCsIndex2]-[e2['length']]*noBifEvents2 if sign2 == 1.0
                                    else [0]*noBifEvents2-e2['rRBC'][bifRBCsIndex2]][0]
                                if sign2 != 1.0:
                                    overshootDist2 = overshootDist2[::-1]
                                overshootTime2=np.array(overshootDist2)/ np.array([e2['v']]*noBifEvents2)
                                dummy2=np.array([2]*len(overshootTime2))
                            else:
                                overshootDist2=[]
                                overshootTime2=[]
                                dummy2=[]
                            overshootTimes=zip(np.concatenate([overshootTime1,overshootTime2]),np.concatenate([dummy1,dummy2]))
                            overshootTimes.sort()
                            overshootTime=[]
                            inEdge=[]
                            #Count RBCs moving from inEdge1 and inEdge2
                            if posNoBifEvents > len(overshootTimes):
                                overshootsNo=int(len(overshootTimes))
                            else:
                                overshootsNo=int(posNoBifEvents)
                            for i in range(-1*overshootsNo,0):
                                overshootTime.append(overshootTimes[i][0])
                                inEdge.append(overshootTimes[i][1])
                            if oe['sign'] == 1.0:
                                position1=np.array(overshootTime)*np.array([oe['v']]*overshootsNo)
                            else:
                                position1=np.array([oe['length']]*overshootsNo)-np.array(overshootTime[::-1])* \
                                    np.array([oe['v']]*overshootsNo)
                            if oe2['sign'] == 1.0:
                                position2=np.array(overshootTime)*np.array([oe2['v']]*overshootsNo)
                            else:
                                position2=np.array([oe2['length']]*overshootsNo)-np.array(overshootTime[::-1])* \
                                    np.array([oe2['v']]*overshootsNo)
                            #To begin with it is tried if all RBCs fit into the prefered outEdge. The time of arrival at the RBCs is taken into account
                            #RBCs which would be too close together are put into the other edge
                            #postion2/position3 is used if there is not enough space in the prefered outEdge and hence the RBC is moved to the other outEdge
                            #actual position of RBCs in the edges
                            positionPref2=[]
                            positionPref1=[]
                            inEPref1=[]
                            inEPref2=[]
                            indexPref1=[]
                            indexPref2=[]
                            #number of RBCs in the Edges
                            countPref1=0
                            countPref2=0
                            pref1Full = 0
                            pref2Full = 0
		            count1 = 0
		            count2 = 0
                            #Loop over all movable RBCs
                            for i in range(overshootsNo):
                                index=-1*(i+1)
                                index1=-1*(i+1) if oe['sign'] == 1.0 else i
                                index2=-1*(i+1) if oe2['sign'] == 1.0 else i
                                #posRBC1=position1[index1] if oe['sign'] == 1 else oe['length']-position1[index1]
                                #posRBC2=position2[index2] if oe2['sign'] == 1 else oe2['length']-position2[index2]
                                #check if RBC still fits into Prefered OutE
                                if posNoBifEventsPref > countPref1 and pref1Full == 0:
                                    #Check if there are RBCs present in outEPref
                                    if positionPref1 != []:
                                        #Check if distance to preceding RBC is big enough
                                        dist1=positionPref1[-1]-position1[index1] if oe['sign'] == 1.0 \
                                            else position1[index1]-positionPref1[-1]
                                        #The distance is not big enough check if RBC fits into outEPref2
                                        if dist1 < oe['minDist']:
                                            #if RBCs are present in the outEdgePref2
                                            if posNoBifEventsPref2 > countPref2 and pref2Full == 0:
                                                if positionPref2 != []:
                                                    dist2=positionPref2[-1]-position2[index2] if oe2['sign'] == 1.0 \
                                                        else position2[index2]-positionPref2[-1]
                                                    #Check if there is enough space in 2nd outEdge
                                                    #in case there is not enough space, check where the RBC is blocked the least amount of time
                                                    if dist2 < oe2['minDist']:
                                                            #Check if another RBCs still fits into the vessel
                                                            space1 =  positionPref1[-1] if oe['sign'] == 1.0 \
                                                                else oe['length']-positionPref1[-1]
                                                            if np.floor(space1/oe['minDist']) > 1:
                                                                timeBlocked1=(oe['minDist']-dist1)/oe['v']
                                                            else:
                                                                timeBlocked1=None
                                                                pref1Full=1
                                                            space2 =  positionPref2[-1] if oe2['sign'] == 1.0 \
                                                                else oe2['length']-positionPref2[-1]
                                                            if np.floor(space2/oe2['minDist']) > 1:
                                                                timeBlocked2=(oe2['minDist']-dist2)/oe2['v']
                                                            else:
                                                                timeBlocked2=None
                                                                pref2Full=1
                                                            if pref1Full == 1 and pref2Full == 1:
                                                                break
                                                            #Define newOutEdge
                                                            newOutEdge=0
                                                            if timeBlocked1 == None:
                                                                newOutEdge=2
                                                            elif timeBlocked2 == None:
                                                                newOutEdge=1
                                                            else:
                                                                if timeBlocked1 <= timeBlocked2:
                                                                    newOutEdge=1
                                                                else:
                                                                    newOutEdge=2
                                                            if newOutEdge == 1:
                                                                if oe['sign'] == 1.0:
                                                                    position1[index1]=positionPref1[-1]-oe['minDist']
                                                                    if position1[index1] > 0:
                                                                        positionPref1.append(position1[index1])
                                                                        inEPref1.append(inEdge[index])
                                                                        indexPref1.append(i)
                                                                        countPref1 += 1
                                                                        if inEdge[index] == 1:
                                                                            count1 += 1
                                                                        elif inEdge[index] == 2:
                                                                            count2 += 1
                                                                    else:
                                                                        print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 23')
                                                                        print(position1[index1])
                                                                        print(np.floor(space1/oe['minDist']))
                                                                        print(timeBlocked1)
                                                                        print(pref1Full)
                                                                else:
                                                                    position1[index1]=positionPref1[-1]+oe['minDist']
                                                                    if position1[index1] < oe['length']:
                                                                        positionPref1.append(position1[index1])
                                                                        inEPref1.append(inEdge[index])
                                                                        indexPref1.append(i)
                                                                        countPref1 += 1
                                                                        if inEdge[index] == 1:
                                                                            count1 += 1
                                                                        elif inEdge[index] == 2:
                                                                            count2 += 1
                                                                    else:
                                                                        print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 24')
                                                                        print(position1[index1])
                                                                        print(oe['length'])
                                                                        print(np.floor(space1/oe['minDist']))
                                                            elif newOutEdge == 2:
                                                                if oe2['sign'] == 1.0:
                                                                    position2[index2]=positionPref2[-1]-oe2['minDist']
                                                                    if position2[index2] > 0:
                                                                        positionPref2.append(position2[index2])
                                                                        inEPref2.append(inEdge[index])
                                                                        indexPref2.append(i)
                                                                        countPref2 += 1
                                                                        if inEdge[index] == 1:
                                                                            count1 += 1
                                                                        elif inEdge[index] == 2:
                                                                            count2 += 1
                                                                    else:
                                                                        print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 25')
                                                                        print(position2[index2])
                                                                        print(np.floor(space2/oe2['minDist']))
                                                                        print(timeBlocked2)
                                                                        print(pref2Full)
                                                                else:
                                                                    position2[index2]=positionPref2[-1]+oe2['minDist']
                                                                    if position2[index2] < oe2['length']:
                                                                        positionPref2.append(position2[index2])
                                                                        inEPref2.append(inEdge[index])
                                                                        indexPref2.append(i)
                                                                        countPref2 += 1
                                                                        if inEdge[index] == 1:
                                                                            count1 += 1
                                                                        elif inEdge[index] == 2:
                                                                            count2 += 1
                                                                    else:
                                                                        print('WARNING RBC has been pushed outside SHOULD NOT HAPPEN 26')
                                                                        print(position2[index2])
                                                                        print(oe2['length'])
                                                                        print(np.floor(space2/oe2['minDist']))
                                                    #there is enough space for the RBC in outEPref2
                                                    else:
                                                        positionPref2.append(position2[index2])
                                                        inEPref2.append(inEdge[index])
                                                        indexPref2.append(i)
                                                        countPref2 += 1
                                                        if inEdge[index] == 1:
                                                            count1 += 1
                                                        elif inEdge[index] == 2:
                                                            count2 += 1
                                                #there are no RBCs in outEdge2
                                                else:
                                                    if oe2['sign'] == 1.0:
                                                        if len(oe2['rRBC'])>0:
                                                            if oe2['rRBC'][0]-position2[index2] < oe2['minDist']:
                                                                position2[index2]=oe2['rRBC'][0]-oe2['minDist']
                                                        else:
                                                            if position2[index2] > oe2['length']:
                                                                position2[index2]=oe2['length']
                                                    else:
                                                        if len(oe2['rRBC'])>0:
                                                            if position2[index2]-oe2['rRBC'][-1] < oe2['minDist']:
                                                                position2[index2]=oe2['rRBC'][-1]+oe2['minDist']
                                                        else:
                                                            if position2[index2] < 0:
                                                                position2[index2]=0
                                                    positionPref2.append(position2[index2])
                                                    inEPref2.append(inEdge[index])
                                                    indexPref2.append(i)
                                                    countPref2 += 1
                                                    if inEdge[index] == 1:
                                                        count1 += 1
                                                    elif inEdge[index] == 2:
                                                        count2 += 1
                                            #There is no space in outEPref2 --> RBC is pushed backwards in outEPref1
                                            else:
                                                if oe['sign'] == 1.0:
                                                    position1[index1]=positionPref1[-1]-oe['minDist']
                                                    if position1[index1] > 0:
                                                        positionPref1.append(position1[index1])
                                                        inEPref1.append(inEdge[index])
                                                        indexPref1.append(i)
                                                        countPref1 += 1
                                                        if inEdge[index] == 1:
                                                            count1 += 1
                                                        elif inEdge[index] == 2:
                                                            count2 += 1
                                                    else:
                                                        pref1Full = 1
                                                        break
                                                else:
                                                    position1[index1]=positionPref1[-1]+oe['minDist']
                                                    if position1[index1] < oe['length']:
                                                        positionPref1.append(position1[index1])
                                                        inEPref1.append(inEdge[index])
                                                        indexPref1.append(i)
                                                        countPref1 += 1
                                                        if inEdge[index] == 1:
                                                            count1 += 1
                                                        elif inEdge[index] == 2:
                                                            count2 += 1
                                                    else:
                                                        pref1Full = 1
                                                        break
                                        #If the RBC fits into outEPref1
                                        else:
                                           positionPref1.append(position1[index1])
                                           inEPref1.append(inEdge[index])
                                           indexPref1.append(i)
                                           countPref1 += 1
                                           if inEdge[index] == 1:
                                               count1 += 1
                                           elif inEdge[index] == 2:
                                               count2 += 1
                                    #There are not yet any new RBCs in outEdgePref
                                    else:
                                        if oe['sign'] == 1.0:
                                            if len(oe['rRBC'])>0:
                                                if oe['rRBC'][0]-position1[index1] < oe['minDist']:
                                                    position1[index1]=oe['rRBC'][0]-oe['minDist']
                                            else:
                                                if position1[index1] > oe['length']:
                                                    position1[index1]=oe['length']
                                        else:
                                            if len(oe['rRBC'])>0:
                                                if position1[index1]-oe['rRBC'][-1] < oe['minDist']:
                                                    position1[index1]=oe['rRBC'][-1]+oe['minDist']
                                            else:
                                                if position1[index1] < 0:
                                                    position1[index1]=0
                                        positionPref1.append(position1[index1])
                                        inEPref1.append(inEdge[index])
                                        indexPref1.append(i)
                                        countPref1 += 1
                                        if inEdge[index] == 1:
                                            count1 += 1
                                        elif inEdge[index] == 2:
                                            count2 += 1
                                #The RBCs do not fit into the prefered outEdge anymore
                                #Therefore they are either put in outEdge2 or outEdge3
                                elif posNoBifEventsPref2 > countPref2 and pref2Full == 0:
                                    #Check if there are already new RBCs in outEPref2
                                    if positionPref2 != []:
                                        dist2=positionPref2[-1]-position2[index2] if oe2['sign'] == 1.0 \
                                            else position2[index2]-positionPref2[-1]
                                        if dist2 < oe2['minDist']:
                                            #RBCs are pushed backewards in Pref2
                                            if oe2['sign'] == 1.0:
                                                position2[index2]=positionPref2[-1]-oe2['minDist']
                                                if position2[index2] > 0:
                                                    positionPref2.append(position2[index2])
                                                    inEPref2.append(inEdge[index])
                                                    indexPref2.append(i)
                                                    countPref2 += 1
                                                    if inEdge[index] == 1:
                                                        count1 += 1
                                                    elif inEdge[index] == 2:
                                                        count2 += 1
                                                else:
                                                    pref2Full = 1
                                                    break
                                            else:
                                                position2[index2]=positionPref2[-1]+oe2['minDist']
                                                if position2[index2] < oe2['length']:
                                                    positionPref2.append(position2[index2])
                                                    inEPref2.append(inEdge[index])
                                                    indexPref2.append(i)
                                                    countPref2 += 1
                                                    if inEdge[index] == 1:
                                                        count1 += 1
                                                    elif inEdge[index] == 2:
                                                        count2 += 1
                                                else:
                                                    pref2Full = 1
                                                    break
                                        #There is enough space for the RBCs in the outEdge 2
                                        else:
                                            positionPref2.append(position2[index2])
                                            inEPref2.append(inEdge[index])
                                            indexPref2.append(i)
                                            countPref2 += 1
                                            if inEdge[index] == 1:
                                                count1 += 1
                                            elif inEdge[index] == 2:
                                                count2 += 1
                                    #There are not yet any RBCs in outEdge 2
                                    else:
                                        if oe2['sign'] == 1.0:
                                            if len(oe2['rRBC']) > 0:
                                                if oe2['rRBC'][0]-position2[index2] < oe2['minDist']:
                                                    position2[index2]=oe2['rRBC'][0]-oe2['minDist']
                                            else:
                                                if position2[index2] > oe2['length']:
                                                    position2[index2]=oe2['length']
                                        else:
                                            if len(oe2['rRBC']) > 0:
                                                if position2[index2]-oe2['rRBC'][-1] < oe2['minDist']:
                                                    position2[index2]=oe2['rRBC'][-1]+oe2['minDist']
                                            else:
                                                if position2[index2] < 0:
                                                    position2[index2]=0
                                        positionPref2.append(position2[index2])
                                        inEPref2.append(inEdge[index])
                                        indexPref2.append(i)
                                        countPref2 += 1
                                        if inEdge[index] == 1:
                                            count1 += 1
                                        elif inEdge[index] == 2:
                                            count2 += 1
                                #There is no more space for further RBCs
                                else:
                                    break
                            #Add rbcs to outEPref
                            if countPref2+countPref1 != overshootsNo:
                                overshootsNo = countPref2+countPref1
                            #Define RBC index
                            dummyOut1=[1]*len(positionPref1)
                            dummyOut2=[2]*len(positionPref2)
                            indicesAndInEdges=zip(np.concatenate([indexPref1,indexPref2]),np.concatenate([inEPref1,inEPref2]),np.concatenate([dummyOut1,dummyOut2])) 
                            indicesAndInEdges.sort()
                            RBCindices=[[],[]]
                            index=[0,0]
                            if e.index not in edgesBox and e2.index not in edgesBox:
                                if oe.index in borderEdges or oe2.index in borderEdges:
                                    print('it s a double connecting bifurcation')
                                    if oe.index in borderEdges:
                                        print('All RBC indices created')
                                        RBCindex=range(RBCindexCurrent+len(positionPref1),RBCindexCurrent,-1)
                                        print('RBCindexCurrent')
                                        print(RBCindexCurrent)
                                        RBCindexCurrent += len(positionPref1)
                                        print('RBCindex')
                                        print(RBCindex)
                                        print('RBCindexCurrent')
                                        print(RBCindexCurrent)
                                        edgesBoxBool=1
                                        if oe2.index in internalEdges:
                                            print('ERROR one border and one internal Edge')
                                    else:
                                        RBCindex=[]
                                    if oe2.index in borderEdges:
                                        print('All RBC indices created')
                                        RBCindex2=range(RBCindexCurrent+len(positionPref2),RBCindexCurrent,-1)
                                        print('RBCindexCurrent')
                                        print(RBCindexCurrent)
                                        RBCindexCurrent += len(positionPref2)
                                        print('RBCindex')
                                        print(RBCindex)
                                        print('RBCindexCurrent')
                                        print(RBCindexCurrent)
                                        edgesBoxBool2=1
                                        if oe1.index in internalEdges:
                                            print('ERROR one border and one internal Edge')
                                    else:
                                        RBCindex2=[]
                                    print('RBCindex')
                                    print(RBCindex)
                                    print('RBCindex 2')
                                    print(RBCindex2)
                            elif oe.index in edgesBox and oe2.index in edgesBox and e.index in edgesBox and e2.index in edgesBox:
                                print('it s a double connecting bifurcation')
                                print(e['RBCindex']) 
                                print(e2['RBCindex']) 
                                for i in range(len(indicesAndInEdges)):
                                    #which out Edge
                                    if indicesAndInEdges[i][2] == 1:
                                        outEdge=oe
                                        oeIndex=0
                                        positionPref=positionPref1
                                    else:
                                        outEdge=oe2
                                        oeIndex=1
                                        positionPref=positionPref2
                                    #which in Edge
                                    if indicesAndInEdges[i][1] == 1:
                                        inEdge=e
                                        ieIndex=0
                                    else:
                                       inEdge=e2
                                       ieIndex=1
                                    if inEdge['sign'] == 1:
                                        RBCindices[oeIndex].append(inEdge['RBCindex'][-(index[ieIndex]+1)])
                                    else:
                                        RBCindices[oeIndex].append(inEdge['RBCindex'][index[ieIndex]])
                                    index[ieIndex] = index[ieIndex] + 1
                                RBCindex=RBCindices[0]
                                RBCindex2=RBCindices[1]
                                print('RBCindex')
                                print(RBCindex)
                                print('RBCindex 2')
                                print(RBCindex2)
                            if oe.index in edgesBox:
                                edgesBoxBool=1
                            else:
                                edgesBoxBool=0
                            if oe2.index in edgesBox:
                                edgesBoxBool2=1
                            else:
                                edgesBoxBool2=0
                            #Add rbcs to outE
                            if oe['sign'] == 1.0:
                                oe['rRBC']=np.concatenate([positionPref1[::-1], oe['rRBC']])
                                if edgesBoxBool:
                                    oe['RBCindex']=np.concatenate([RBCindex,oe['RBCindex']])
                                    print('oe[RBCindex]')
                                    print(oe['RBCindex'])
                            else:
                                oe['rRBC']=np.concatenate([oe['rRBC'],positionPref1])
                                if edgesBoxBool:
                                    oe['RBCindex']=np.concatenate([oe['RBCindex'],RBCindex[::-1]])
                                    print('First indexs to be placed')
                                    print('oe[RBCindex]')
                                    print(oe['RBCindex'])
                            #Add rbcs to outEPref2       
                            if oe2['sign'] == 1.0:
                                oe2['rRBC']=np.concatenate([positionPref2[::-1], oe2['rRBC']])
                                if edgesBoxBool2:
                                    oe2['RBCindex']=np.concatenate([RBCindex2,oe2['RBCindex']])
                                    print('oe[RBCindex]')
                                    print(oe2['RBCindex'])
                            else:
                                oe2['rRBC']=np.concatenate([oe2['rRBC'],positionPref2])
                                if edgesBoxBool2:
                                    oe2['RBCindex']=np.concatenate([oe2['RBCindex'],RBCindex2[::-1]])
                                    print('oe[RBCindex]')
                                    print(oe2['RBCindex'])
                            #Remove RBCs from old Edge 1
                            if count1 > 0:
                                if sign == 1.0:
                                    e['rRBC']=e['rRBC'][:-count1]
                                    if e['RBCindex'] != None:
                                        print('it s a double connecting bifurcation')
                                        print('Remove RBCs')
                                        print(e['RBCindex'])
                                        e['RBCindex']=e['RBCindex'][:-count1]
                                        print(e['RBCindex'])
                                else:
                                    e['rRBC']=e['rRBC'][count1::]
                                    if e['RBCindex'] != None:
                                        print('it s a double connecting bifurcation')
                                        print('Remove RBCs')
                                        print(e['RBCindex'])
                                        e['RBCindex']=e['RBCindex'][count1::]
                                        print(e['RBCindex'])
                            if noBifEvents2 > 0 and count2 > 0:
                                #Remove RBCs from old Edge 2
                                if sign2 == 1.0:
                                    e2['rRBC']=e2['rRBC'][:-count2]
                                    if e2['RBCindex'] != None:
                                        print('it s a double connecting bifurcation')
                                        print('Remove RBCs 2')
                                        print(e2['RBCindex'])
                                        e2['RBCindex']=e2['RBCindex'][:-count2]
                                        print(e2['RBCindex'])
                                else:
                                    e2['rRBC']=e2['rRBC'][count2::]
                                    if e2['RBCindex'] != None:
                                        print('it s a double connecting bifurcation')
                                        print('Remove RBCs 2')
                                        print(e2['RBCindex'])
                                        e2['RBCindex']=e2['RBCindex'][count2::]
                                        print(e2['RBCindex'])
                            #Recheck if the distance between the newly introduces RBCs is still big enough
                            ##TODO:  not really needed needed
                            #if len(oe['rRBC']) > 0:
                            #    if len(positionPref1) == len(oe['rRBC']):
                            #        if oe['sign'] == 1.0:
                            #            iter=len(positionPref1)-2
                            #            if oe['rRBC'][-1] > oe['length']:
                            #                oe['rRBC'][-1]=oe['length']
                            #        else:
                            #            iter=len(positionPref1)-1
                            #            if oe['rRBC'][0] < 0:
                            #                oe['rRBC'][0]=0
                            #    else:
                            #        if oe['sign']==1.0:
                            #            iter=len(positionPref1)-1
                            #        else:
                            #            iter=len(positionPref1)
                            #    if oe['sign']==1.0:
                            #        for i in range(iter,-1,-1):
                            #            if oe['rRBC'][i+1]-oe['rRBC'][i] + eps < oe['minDist'] \
                            #                or oe['rRBC'][i+1]-oe['rRBC'][i] + eps < 0:
                            #                print('ERROR CHECK 9 should not be necessary')
                            #                print(oe['rRBC'][i+1]-oe['rRBC'][i])
                            #                print(oe['minDist'])
                            #                print(oe['rRBC'][i+1]-oe['rRBC'][i]-oe['minDist'])
                            #                print(eps*10000)
                            #                oe['rRBC'][i]=oe['rRBC'][i+1]-oe['minDist']
                            #    else:
                            #        for i in range(-1*iter,0,1):
                            #            if oe['rRBC'][i]-oe['rRBC'][i-1] + eps < oe['minDist'] \
                            #                or oe['rRBC'][i]-oe['rRBC'][i-1] + eps < 0:
                            #                print('ERROR CHECK 10 should not be necessary')
                            #                print(oe['rRBC'][i]-oe['rRBC'][i-1])
                            #                print(oe['minDist'])
                            #                print(oe['rRBC'][i]-oe['rRBC'][i-1]-oe['minDist'])
                            #                print(eps*10000)
                            #                oe['rRBC'][i]=oe['rRBC'][i-1]+oe['minDist']
                            #if len(oe2['rRBC']) > 0:
                            #    if len(positionPref2) == len(oe2['rRBC']):
                            #        if oe2['sign'] == 1.0:
                            #            iter=len(positionPref2)-2
                            #            if oe2['rRBC'][-1] > oe2['length']:
                            #                oe2['rRBC'][-1]=oe2['length']
                            #        else:
                            #            iter=len(positionPref2)-1
                            #            if oe2['rRBC'][0] < 0:
                            #                oe2['rRBC'][0]=0
                            #    else:
                            #        if oe2['sign']==1.0:
                            #            iter=len(positionPref2)-1
                            #        else:
                            #            iter=len(positionPref2)
                            #    if oe2['sign']==1.0:
                            #        for i in range(iter,-1,-1):
                            #            if oe2['rRBC'][i+1]-oe2['rRBC'][i] + eps < oe2['minDist'] \
                            #                or oe2['rRBC'][i+1]-oe2['rRBC'][i] + eps < 0:
                            #                print('ERROR CHECK 11 should not be necessary')
                            #                print(oe2['rRBC'][i+1]-oe2['rRBC'][i])
                            #                print(oe2['minDist'])
                            #                print(oe2['rRBC'][i+1]-oe2['rRBC'][i]-oe2['minDist'])
                            #                print(eps*10000)
                            #                oe2['rRBC'][i]=oe2['rRBC'][i+1]-oe2['minDist']
                            #    else:
                            #        for i in range(-1*iter,0,1):
                            #            if oe2['rRBC'][i]-oe2['rRBC'][i-1] + eps < oe2['minDist'] \
                            #                or oe2['rRBC'][i]-oe2['rRBC'][i-1] + eps < 0:
                            #                print('ERROR 12 CHECK should not be necessary')
                            #                print(oe2['rRBC'][i]-oe2['rRBC'][i-1])
                            #                print(oe2['minDist'])
                            #                print(oe2['rRBC'][i]-oe2['rRBC'][i-1]-oe2['minDist'])
                            #                print(eps*10000)
                            #                oe2['rRBC'][i]=oe2['rRBC'][i-1]+oe2['minDist']
                        #OutEdges are currently blocked, no bifurcation events possible
                        else:
                            countPref1=0
                            countPref2=0
                            count1=0
                            count2=0
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        #InEdge 1
                        noStuckRBCs1=len(bifRBCsIndex1)-count1
                        for i in range(noStuckRBCs1):
                            index=-1*(i+1) if sign == 1.0 else i
                            e['rRBC'][index]=e['length']-i*e['minDist'] if sign == 1.0 else 0+i*e['minDist']
                        #Recheck if the distance between the newly introduces RBCs is still big enough 
                        if len(e['rRBC']) >1.0:
                            moved = 0
                            if sign == 1.0:
                                count = 0
                                for i in range(-1,-1*(len(e['rRBC'])),-1):
                                    index=i-1
                                    if e['rRBC'][i] < e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]-e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs1 and moved == 0:
                                        break
                            else:
                                count = 0
                                for i in range(len(e['rRBC'])-1):
                                    index=i+1
                                    if e['rRBC'][i] > e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]+e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs1+1 and moved == 0:
                                        break
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        #InEdge 2
                        if noBifEvents2 > 0:
                            noStuckRBCs2=len(bifRBCsIndex2)-count2
                            for i in range(noStuckRBCs2):
                                index=[-1*(i+1) if sign2 == 1.0 else i]
                                e2['rRBC'][index]=[e2['length']-i*e2['minDist'] if sign2 == 1.0 else 0+i*e2['minDist']]
                            #Recheck if the distance between the newly introduces RBCs is still big enough 
                            if len(e2['rRBC']) >1:
                                moved = 0
                                if sign2 == 1.0:
                                    count = 0
                                    for i in range(-1,-1*(len(e2['rRBC'])),-1):
                                        index=i-1
                                        if e2['rRBC'][i] < e2['rRBC'][index] or abs(e2['rRBC'][i]-e2['rRBC'][index]) < e2['minDist']:
                                            e2['rRBC'][index]=e2['rRBC'][i]-e2['minDist']
                                            moved = 1
                                        else:
                                            moved = 0
                                        count += 1
                                        if count >= noStuckRBCs2 and moved == 0:
                                            break
                                else:
                                    count = 0
                                    for i in range(len(e2['rRBC'])-1):
                                        index=i+1
                                        if e2['rRBC'][i] > e2['rRBC'][index] or abs(e2['rRBC'][i]-e2['rRBC'][index]) < e2['minDist']:
                                            e2['rRBC'][index]=e2['rRBC'][i]+e2['minDist']
                                            moved = 1
                                        else:
                                            moved = 0
                                        count += 1
                                        if count >= noStuckRBCs2+1 and moved == 0:
                                            break
            #-------------------------------------------------------------------------------------------
                    if overshootsNo != 0:
                        vertexUpdate.append(e.target)
                        vertexUpdate.append(e.source)
                        for i in edgesInvolved:
                            edgeUpdate.append(i)
                        if self._analyzeBifEvents:
                            if G.vs['vType'][vi] == 3 or G.vs['vType'][vi] == 5:
                                rbcMoved += overshootsNo
                            elif G.vs['vType'][vi] == 6:
                                rbcMoved += count1 + count2
                            elif G.vs['vType'][vi] == 4:
                                rbcMoved += count1 + count2
                                if len(inflowEdges) > 2:
                                    if count3 > 0:
                                        rbcMoved += count3
                        if self._analyzeBifEvents:
                            if G.vs['vType'][vi] == 3 or G.vs['vType'][vi] == 5:
                                rbcsMovedPerEdge.append(overshootsNo)
                                edgesWithMovedRBCs.append(e.index)
                            elif G.vs['vType'][vi] == 6:
                                if count1 > 0:
                                    rbcsMovedPerEdge.append(count1)
                                    edgesWithMovedRBCs.append(e.index)
                                if count2 > 0:
                                    edgesWithMovedRBCs.append(e2.index)
                                    rbcsMovedPerEdge.append(count2)
                            elif G.vs['vType'][vi] == 4:
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
                    #for i in edgesInvolved:
                    #    if len(G.es['rRBC'][i]) > 0:
                    #        if G.es['rRBC'][i][0] < 0:
                    #            print('BIGERROR')
                    #            print(G.es['rRBC'][i][0])
                    #            print(overshootsNo)
                    #            print(vi)
                    #            print(G.vs['vType'][vi])
                    #        if G.es['rRBC'][i][-1] > G.es['length'][i]:
                    #            print('BIGERROR 2')
                    #            print(G.es['rRBC'][i][-1])
                    #            print(G.es['length'][i])
                    #            print(overshootsNo)
                    #            print(vi)
                    #            print(G.vs['vType'][vi])

        #-------------------------------------------------------------------------------------------
        self._vertexUpdate=np.unique(vertexUpdate)
        self._edgeUpdate=np.unique(edgeUpdate)
        nRBC=G.es['nRBC']
        G.es['nRBC'] = [len(e['rRBC']) for e in G.es]
        if self._analyzeBifEvents:
            self._rbcsMovedPerEdge.append(rbcsMovedPerEdge)
            self._edgesWithMovedRBCs.append(edgesWithMovedRBCs)
            self._rbcMoveAll.append(rbcMoved)
        self._G=G
        self._RBCindexCurrent=RBCindexCurrent

    #--------------------------------------------------------------------------

    def _particleTracking(self):
        """Takes a snapshot of relevant current data and adds it to the sample
        database.
        INPUT: None
        OUTPUT: None, data added to self._sampledict
        """
        RBCdict = self._RBCdict
        G = self._G
        edgesBox=self._allEdgesBox
        tSample=self._tSample
        
        for i in edgesBox:
            for j in range(len(G.es['rRBC'][i])):
                index=G.es['RBCindex'][i][j]
                if index not in RBCdict.keys():
                    RBCdict[index]=[]
                    RBCdict[index].append([]) #time
                    RBCdict[index].append([]) #edge
                    RBCdict[index].append([]) #position
                RBCdict[index][0].append(tSample)
                RBCdict[index][1].append(i)
                RBCdict[index][2].append(G.es['rRBC'][i][j])

        for eprop in ['flow','v','sign']:
            if eprop not in RBCdict.keys():
                RBCdict[eprop]=[]
                for i in range(len(edgesBox)):
                    RBCdict[eprop].append([])
            for i in range(len(edgesBox)):
                RBCdict[eprop][i].append(G.es[eprop][edgesBox[i]])
        if 'time' not in RBCdict.keys():
             RBCdict['time']=[]
        RBCdict['time'].append(self._tSample)

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
        G=self._G
        if len(G.es(nMax_eq=0)) > 0:
            sys.exit("BIGERROR nMax=0 exists --> check vessel lengths") 
        tPlot = self._tPlot # deepcopy, since type is float
        tSample = self._tSample # deepcopy, since type is float
        filenamelist = self._filenamelist
        self._dt=dtfix
        timelist = self._timelist
	#filenamelistAvg = self._filenamelistAvg
	timelistAvg = self._timelistAvg

        if 'init' in kwargs.keys():
            init=kwargs['init']
        else:
            init=self._init

        SampleDetailed=False
        if 'SampleDetailed' in kwargs.keys():
            SampleDetailed=kwargs['SampleDetailed']

        doSampling, doPlotting = [False, False]

        if 'plotPrms' in kwargs.keys():
            pStart, pStop, pStep = kwargs['plotPrms']
            doPlotting = True
            if init == True:
                tPlot = 0.0
                filenamelist = []
                timelist = []
            else:
                tPlot=G['iterFinalPlot']
                pStart = G['iterFinalPlot']+pStart+pStep
                pStop = G['iterFinalPlot']+pStop

        if 'samplePrms' in kwargs.keys():
            sStart, sStop, sStep = kwargs['samplePrms']
            doSampling = True
            if init == True:
                self._tSample = 0.0
                self._sampledict = {}
                #self._transitTimeDict = {}
                #filenamelistAvg = []
                timelistAvg = []
            else:
                self._tSample = G['iterFinalSample']
                sStart = G['iterFinalSample']+sStart+sStep
                sStop = G['iterFinalSample']+sStop

        t1 = ttime.time()
        if init:
            self._t = 0.0
            BackUpTStart=0.1*time
            #BackUpTStart=0.0005*time
            BackUpT=0.1*time
            #BackUpT=0.0005*time
            BackUpCounter=0
        else:
            self._t = G['dtFinal']
            self._tSample=G['iterFinalSample']
            BackUpT=0.1*time
            print('Simulation starts at')
            print(self._t)
            print('First BackUp should be done at')
            time = G['dtFinal']+time
            BackUpCounter=G['BackUpCounter']+1
            BackUpTStart=G['dtFinal']+BackUpT
            print(BackUpTStart)
            print('BackUp should be done every')
            print(BackUpT)

        #Convert 'pBC' ['mmHG'] to default Units
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']*self._scaleToDef

        tSample = self._tSample
        start_timeTot=ttime.time()
        t=self._t
        iteration=0
        while True:
            if t >= time:
                break
            iteration += 1
            start_time=ttime.time()
            self._update_eff_resistance_and_LS(None, self._vertexUpdate, False)
            print('Matrix updated')
            self._solve(method, **kwargs)
            print('Matrix solved')
            self._G.vs['pressure'] = deepcopy(self._x)
            print('Pressure copied')
            self._update_flow_and_velocity()
            print('Flow updated')
            self._update_flow_sign()
            print('Flow sign updated')
            self._verify_mass_balance()
            print('Mass balance verified updated')
            self._update_out_and_inflows_for_vertices()
            print('In and outflows updated')
            self._particleTracking()
            print('Particle Tracking')
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
                self._t=t
                self._tSample=tSample
                self._sample()
                filenameDetailed ='G_iteration_'+str(iteration)+'.pkl'
                #Convert deaultUnits to ['mmHG']
                #for 'pBC' and 'pressure'
                for v in G.vs:
                    if v['pBC'] != None:
                        v['pBC']=v['pBC']/self._scaleToDef
                    v['pressure']=v['pressure']/self._scaleToDef
                vgm.write_pkl(G,filenameDetailed)
                #Convert 'pBC' ['mmHG'] to default Units
                for v in G.vs:
                    if v['pBC'] != None:
                        v['pBC']=v['pBC']*self._scaleToDef
                    v['pressure']=v['pressure']*self._scaleToDef
            else:
                if doSampling and tSample >= sStart and tSample <= sStop:
                    print('DO sampling')
                    stdout.flush()
                    self._t=t
                    self._tSample=tSample
                    print('start sampling')
                    stdout.flush()
                    self._sample()
                    sStart = tSample + sStep
                    print('sampling DONE')
                    if t > BackUpTStart:
                        print('BackUp should be done')
                        print(BackUpCounter)
                        stdout.flush()
                        G['dtFinal']=t
                        G['iterFinalSample']=tSample
                        G['BackUpCounter']=BackUpCounter
                        if self._analyzeBifEvents:
                            G['rbcsMovedPerEdge']=self._rbcsMovedPerEdge
                            G['edgesMovedRBCs']=self._edgesWithMovedRBCs
                            G['rbcMovedAll']=self._rbcMoveAll
                        #G['spacing']=self._spacing
                        filename1='sampledict_BackUp_'+str(BackUpCounter)+'.pkl'
                        filename2='G_BackUp'+str(BackUpCounter)+'.pkl'
                        self._sample_average()
                        print(filename1)
                        print(filename2)
                        #Convert deaultUnits to 'pBC' ['mmHG']
                        for v in G.vs:
                            if v['pBC'] != None:
                                v['pBC']=v['pBC']/self._scaleToDef
                            v['pressure']=v['pressure']/self._scaleToDef
                        #g_output.write_pkl(self._sampledict,filename1)
                        #vgm.write_pkl(G,filename2)
                        self._sampledict = {}
                        self._sampledict['averagedCount']=G['averagedCount']
                        #if 'edgeIndex' not in self._RBCdict.keys():
                        #    self._RBCdict['edgeIndex']=self._allEdgesBox
                        #vgm.write_pkl(self._RBCdict,'RBCdict_BackUp_'+str(BackUpCounter)+'.pkl')
                        print('RBCdict written')
                        #Convert 'pBC' ['mmHG'] to default Units
                        for v in G.vs:
                            if v['pBC'] != None:
                                v['pBC']=v['pBC']*self._scaleToDef
                            v['pressure']=v['pressure']*self._scaleToDef
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

        self._update_eff_resistance_and_LS(None, None, False)
        self._solve(method, **kwargs)
        self._G.vs['pressure'] = deepcopy(self._x)
        print('Pressure copied')
        self._update_flow_and_velocity()
        self._update_flow_sign()
        self._update_out_and_inflows_for_vertices()
        self._verify_mass_balance()
        print('Mass balance verified updated')
        self._t=t
        self._tSample=tSample
        stdout.flush()

        G['dtFinal']=t
        if self._analyzeBifEvents:
            G['rbcsMovedPerEdge']=self._rbcsMovedPerEdge
            G['edgesMovedRBCs']=self._edgesWithMovedRBCs
            G['rbcMovedAll']=self._rbcMoveAll
        #G['iterFinalPlot']=tPlot
        G['iterFinalSample']=tSample
        G['BackUpCounter']=BackUpCounter
        #G['spacing']=self._spacing
        filename1='sampledict_BackUp_'+str(BackUpCounter)+'.pkl'
        filename2='G_BackUp'+str(BackUpCounter)+'.pkl'
        #if doPlotting:
        #    filename= 'iter_'+str(int(round(tPlot+1)))+'.vtp'
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
                    v['pBC']=v['pBC']/self._scaleToDef
                v['pressure']=v['pressure']/self._scaleToDef
            self._sample_average()
            g_output.write_pkl(self._sampledict, 'sampledict.pkl')
            #COMMENTED
            #g_output.write_pkl(self._sampledict,filename1)
	    #g_output.write_pkl(self._transitTimeDict, 'TransitTimeDict.pkl')
            #g_output.write_pvd_time_series('sequenceSampling.pvd',
	    #				   filenamelistAvg, timelistAvg)
        #G['spacing']=self._spacing
        vgm.write_pkl(G, 'G_final.pkl')
        #COMMENTED
        #vgm.write_pkl(G,filename2)
        if 'edgeIndex' not in self._RBCdict.keys():
            self._RBCdict['edgeIndex']=self._allEdgesBox
        vgm.write_pkl(self._RBCdict,'RBCdict.pkl')
        # Since Physiology has been rewritten using Cython, it cannot be
        # pickled. This class holds a Physiology object as a member and
        # consequently connot be pickled either.
        #g_output.write_pkl(self, 'LSHTD.pkl')
        #self._timelist = timelist[:]
        #self._filenamelist = filenamelist[:]
	#self._filenamelistAvg = filenamelistAvg[:]
	#self._timelistAvg = timelistAvg[:]

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
                rsource = G.vs[e.source]['r']
                dvec = G.vs[e.target]['r'] - G.vs[e.source]['r']
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
        scaleToDef=self._scaleToDef
        #Convert default units to ['mmHG']
        pressure=np.array([1/scaleToDef]*G.vcount())*G.vs['pressure']
        
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
            avCount=sampledict['averagedCount']
        else:
            avCount = 0
        avCountNew=len(sampledict['time'])
        avCountE=np.array([avCount]*G.ecount())
        avCountNewE=np.array([avCountNew]*G.ecount())
        for eprop in ['flow', 'v', 'htt', 'htd','nRBC','effResistance']:
            if eprop+'_avg' in G.es.attribute_names():
                G.es[eprop + '_avg'] = (avCountE*G.es[eprop+'_avg']+ \
                    avCountNewE*np.average(sampledict[eprop], axis=0))/(avCountE+avCountNewE)
            else:
                G.es[eprop + '_avg'] = np.average(sampledict[eprop], axis=0)
            #if not [eprop + '_avg'] in sampledict.keys():
            #    sampledict[eprop + '_avg']=[]
            sampledict[eprop + '_avg']=G.es[eprop + '_avg']
        avCountV=np.array([avCount]*G.vcount())
        avCountNewV=np.array([avCountNew]*G.vcount())
        for vprop in ['pressure']:
            if vprop+'_avg' in G.vs.attribute_names():
                G.vs[vprop + '_avg'] = (avCountV*G.vs[vprop+'_avg']+ \
                    avCountNewV*np.average(sampledict[vprop], axis=0))/(avCountV+avCountNewV)
            else:
                G.vs[vprop + '_avg'] = np.average(sampledict[vprop], axis=0)
            #if not [vprop + '_avg'] in sampledict.keys():
            #    sampledict[vprop + '_avg']=[]
            sampledict[vprop + '_avg']=G.vs[vprop + '_avg']
        sampledict['averagedCount']=avCount + avCountNew
        G['averagedCount']=avCount + avCountNew
        #g_output.write_vtp(G, sampleAvgFilename, False)


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
            x = abs(AA.solve(self._b, tol=self._eps/10000000000000000000, accel='cg')) # abs required, as (small) negative pressures may arise
        elif method == 'iterative2':
         # Set linear solver
             ml = rootnode_solver(A, smooth=('energy', {'degree':2}), strength='evolution' )
             M = ml.aspreconditioner(cycle='V')
             # Solve pressure system
             #x,info = gmres(A, self._b, tol=self._eps, maxiter=50, M=M, x0=self._x)
             #x,info = gmres(A, self._b, tol=self._eps/10000000000000, maxiter=50, M=M)
             x,info = gmres(A, self._b, tol=self._eps/10000000000, maxiter=50, M=M)
             if info != 0:
                 print('SOLVEERROR in Solving the Matrix')
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
        for i in range(G.vcount()):
            if G.vs[i]['flowSum'] > 1e-4 and i not in G['av'] and i not in G['vv']:
                print('')
                print(i)
                print(G.vs['flowSum'][i])
                #print(self._res[i])
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
        invivo=self._invivo
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
                v['pBC']=v['pBC']*self._scaleToDef

        for i, v in enumerate(G.vs):
            if v['pBC'] is None:
                pdiff = [v['pressure'] - n['pressure']
                         for n in G.vs[G.neighbors(i)]]
                if min(pdiff) > 0:
                    localMaxima.append((i, max(pdiff)))         
        #Convert defaultUnits to 'pBC' ['mmHG']
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']/self._scaleToDef

        return localMaxima

    #--------------------------------------------------------------------------
    
    def _residual_norm(self):
        """Computes the norm of the current residual.
        """
        return np.linalg.norm(self._A * self._x - self._b)

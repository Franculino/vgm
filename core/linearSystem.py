from __future__ import division, print_function
from copy import deepcopy
import pyamg
import igraph as ig
import numpy as np
from pyamg import smoothed_aggregation_solver, rootnode_solver, util
from scipy import finfo, ones, zeros
from scipy.sparse import lil_matrix, linalg, coo_matrix
from scipy.sparse.linalg import gmres
from physiology import Physiology
import units
import g_output
import vascularGraph
import pdb
import time as ttime
import vgm

__all__ = ['LinearSystem']

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class LinearSystem(object):
    def __init__(self, G, **kwargs):
        """Constructs the linear system A x = b where the matrix A contains the 
        conductance information of the vascular graph, the vector b specifies 
        the boundary conditions and the vector x holds the pressures at the 
        vertices (for which the system needs to be solved).
        pBC should be given in mmHG and pressure will be output in mmHg

        INPUT: G: Vascular graph in iGraph format.(the pBC should be given in mmHg)
               withRBC: boolean if a fixed distribution of RBCs should be considered
                       if 'htt' is an edge attribute the current distribution is used.
                       Otherwise the value to be assinged for all vessels should be given.
                       For arteries and veins a empirical value which is a function of
                       the diameter is assigned.
               resistanceLength: boolean if diameter is not considered for the restistance
                       and hence the resistance is only a function of the vessel length
        OUTPUT: A: Matrix A of the linear system, holding the conductance 
                   information.
                b: Vector b of the linear system, holding the boundary 
                   conditions.
        """
        self._G = G
        self._P = Physiology(G['defaultUnits'])
        self._muPlasma = self._P.dynamic_plasma_viscosity()
        #Check if a arbirtrary distribution of RBCs should be considered
        if kwargs.has_key('withRBC'):
            if kwargs['withRBC']!=0:
                self._withRBC = kwargs['withRBC']
            else:
                self._withRBC = 0
        else:
            self._withRBC = 0

        if kwargs.has_key('invivo'):
            if kwargs['invivo']!=0:
                self._invivo = kwargs['invivo']
            else:
                self._invivo = 0
        else:
            self._invivo = 0

        if kwargs.has_key('resistanceLength'):
            if kwargs['resistanceLength']==1:
                self._resistanceLength = 1
                print('Diameter not considered for calculation of resistance')
            else:
                self._resistanceLength = 0
        else:
            self._resistanceLength = 0

        if self._withRBC != 0:
            if self._withRBC < 1.:
                if 'htt' not in G.es.attribute_names():
                    G.es['htt']=[self._withRBC]*G.ecount()
                    self._withRBC = 1
                else:
                    httNone = G.es(htt_eq=None).indices
                    if len(httNone) > 0:
                        G.es[httNone]['htt']=[self._withRBC]*len(httNone)

        self.update(G)
        self._eps = np.finfo(float).eps
        
    #--------------------------------------------------------------------------    
        
    def update(self, newGraph=None):
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
                         previous self._G. (Optional, default=None.)
        OUTPUT: A: Matrix A of the linear system, holding the conductance 
                   information.
                b: Vector b of the linear system, holding the boundary 
                   conditions.
        """
        htt2htd = self._P.tube_to_discharge_hematocrit
        nurel = self._P.relative_apparent_blood_viscosity
        if newGraph is not None:
            self._G = newGraph
            
        G = self._G
        if not G.vs[0].attributes().has_key('pBC'):
            G.vs[0]['pBC'] = None
        if not G.vs[0].attributes().has_key('rBC'):
            G.vs[0]['rBC'] = None        

        #Convert 'pBC' ['mmHG'] to default Units
        pBCneNone=G.vs(pBC_ne=None).indices
        for i in pBCneNone:
            v=G.vs[i]
            v['pBC']=v['pBC']*vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])
        
        nVertices = G.vcount()
        b = np.zeros(nVertices)            
        A = lil_matrix((nVertices,nVertices),dtype=float)

        # Compute nominal and specific resistance:
        self._update_nominal_and_specific_resistance()

        #if with RBCs compute effective resistance
        if self._withRBC:
            if 'htd' not in G.es.attribute_names():
                dischargeHt = [min(htt2htd(e, d, self._invivo), 1.0) for e,d in zip(G.es['htt'],G.es['diameter'])]
            else:
                dischargeHt = G.es['htd']
            #G.es['effResistance'] =[ res * nurel(max(4.0,d),min(dHt,0.6),self._invivo) for res,dHt,d in zip(G.es['resistance'], \
            #    dischargeHt,G.es['diameter'])]
            G.es['effResistance'] =[ res * nurel(max(3.5,d),min(dHt,0.6),self._invivo) for res,dHt,d in zip(G.es['resistance'], \
                dischargeHt,G.es['diameter'])]
            G.es['conductance']=1/np.array(G.es['effResistance'])
        else: 
	    # Compute conductance
            for e in G.es:
	            e['conductance']=1/e['resistance']
        
            #if not bound_cond is None:
            #    self._conductance = [max(min(c, bound_cond[1]), bound_cond[0])
            #                     for c in G.es['conductance']]
            #else:
            #    self._conductance = G.es['conductance']
        self._conductance = G.es['conductance']

        for vertex in G.vs:        
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
                    conductance = G.es[edge]['conductance']
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
    
    #--------------------------------------------------------------------------

    def solve(self, method, **kwargs):
        """Solves the linear system A x = b for the vector of unknown pressures
        x, either using a direct solver or an iterative AMG solver. From the
        pressures, the flow field is computed.
        INPUT: method: This can be either 'direct' or 'iterative'
               **kwargs 
               precision: The accuracy to which the ls is to be solved. If not 
                          supplied, machine accuracy will be used.
               maxiter: The maximum number of iterations. The default value for
                        the iterative solver is 250.
        OUTPUT: None - G is modified in place.
        """
                
        b = self._b
        G = self._G
        htt2htd = self._P.tube_to_discharge_hematocrit
        
        A = self._A.tocsr()
        if method == 'direct':
            linalg.use_solver(useUmfpack=True)
            x = linalg.spsolve(A, b)
        elif method == 'iterative':
            if kwargs.has_key('precision'):
                eps = kwargs['precision']
            else:
                eps = self._eps
            if kwargs.has_key('maxiter'):
                maxiter = kwargs['maxiter']
            else:
                maxiter = 250
            AA = pyamg.smoothed_aggregation_solver(A, max_levels=10, max_coarse=500)
            x = abs(AA.solve(self._b, x0=None, tol=eps, accel='cg', cycle='V', maxiter=maxiter))
            # abs required, as (small) negative pressures may arise
        elif method == 'iterative2':
         # Set linear solver
             ml = rootnode_solver(A, smooth=('energy', {'degree':2}), strength='evolution' )
             M = ml.aspreconditioner(cycle='V')
             # Solve pressure system
             #x,info = gmres(A, self._b, tol=self._eps, maxiter=1000, M=M)
             x,info = gmres(A, self._b, tol=10*self._eps,M=M)
             if info != 0:
                 print('ERROR in Solving the Matrix')
                 print(info)

        G.vs['pressure'] = x
        self._x = x
        conductance = self._conductance
        G.es['flow'] = [abs(G.vs[edge.source]['pressure'] -   \
                            G.vs[edge.target]['pressure']) *  \
                        conductance[i] for i, edge in enumerate(G.es)]
        for v in G.vs:
            v['pressure']=v['pressure']/vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])
        if self._withRBC:
            if 'htd' not in G.es.attribute_names():
                dischargeHt = [min(htt2htd(e['htt'], e['diameter'], self._invivo), 1.0) for e in G.es]
            else:
                dischargeHt = G.es['htd']
            if 'htt' not in G.es.attribute_names():
                G.es['v']=[e['flow']/(0.25*np.pi*e['diameter']**2) for e in G.es]
            else:
	            G.es['v']=[Htd/e['htt']*e['flow']/(0.25*np.pi*e['diameter']**2) for Htd,e in zip(dischargeHt,G.es)]
        else:
            for e in G.es:
	            e['v']=e['flow']/(0.25*np.pi*e['diameter']**2)
        
        #Convert 'pBC' from default Units to mmHg
        pBCneNone=G.vs(pBC_ne=None).indices
        if 'diamCalcEff' in G.es.attribute_names():
            del(G.es['diamCalcEff'])

        #if 'effResistance' in G.es.attribute_names():
        #    del(G.es['effResistance'])

        #if 'conductance' in G.es.attribute_names():
        #    del(G.es['conductance'])

        #if 'resistance' in G.es.attribute_names():
        #    del(G.es['resistance'])

        G.vs[pBCneNone]['pBC']=np.array(G.vs[pBCneNone]['pBC'])*(1/vgm.units.scaling_factor_du('mmHg',G['defaultUnits']))

        vgm.write_pkl(G, 'G_final.pkl')
        vgm.write_vtp(G, 'G_final.vtp',False)

	#Write Output
	sampledict={}
        for eprop in ['flow', 'v']:
            if not eprop in sampledict.keys():
                sampledict[eprop] = []
            sampledict[eprop].append(G.es[eprop])
        for vprop in ['pressure']:
            if not vprop in sampledict.keys():
                sampledict[vprop] = []
            sampledict[vprop].append(G.vs[vprop])

	g_output.write_pkl(sampledict, 'sampledict.pkl')
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
            if G.vs[i]['flowSum'] > self._eps:
                print('')
                print(i)
                print(G.vs['flowSum'][i])
                #print(self._res[i])
                print('ERROR')
                for j in G.adjacent(i):
                    print(G.es['flow'][j])

    #--------------------------------------------------------------------------

    def _verify_p_consistency(self):
        """Checks for local pressure maxima at non-pBC vertices.
        INPUT: None.
        OUTPUT: A list of local pressure maxima vertices and the maximum 
                pressure difference to their respective neighbors."""
        G = self._G
        localMaxima = []
        for i, v in enumerate(G.vs):
            if v['pBC'] is None:
                pdiff = [v['pressure'] - n['pressure']
                         for n in G.vs[G.neighbors(i)]]
                if min(pdiff) > 0:
                    localMaxima.append((i, max(pdiff)))         
        return localMaxima

    #--------------------------------------------------------------------------
    
    def _residual_norm(self):
        """Computes the norm of the current residual.
        """
        return np.linalg.norm(self._A * self._x - self._b)

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

        if esequence is None:
            es = G.es
        else:
            es = G.es(esequence)

        if self._resistanceLength:
            G.es['specificResistance'] = [1]*G.ecount()
        else:
            G.es['specificResistance'] = [128 * self._muPlasma / (np.pi * d**4)
                                        for d in G.es['diameter']]

        G.es['resistance'] = [l * sr for l, sr in zip(G.es['length'],
                                                G.es['specificResistance'])]

    #-----------------------------------------------------------------------------

from __future__ import division, print_function
from copy import deepcopy
import numpy as np
from pyamg import rootnode_solver
from scipy import finfo
from scipy.sparse import lil_matrix, linalg
from scipy.sparse.linalg import gmres
from physiology import Physiology
import units
import g_output
import vascularGraph
import vgm

__all__ = ['LinearSystemTimeCourse']

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class LinearSystemTimeCourse(object):
    def __init__(self, G, withRBC = 0, invivo = 0, dMin_empirical = 3.5, htdMax_empirical = 0.6, verbose = True,
            diameterOverTime=[],**kwargs):
        """
        Computes the flow and pressure field of a vascular graph without RBC tracking.
        It can be chosen between pure plasma flow, constant hematocrit or a given htt/htd
        distribution.
        The pressure boundary conditions (pBC) should be given in mmHG and pressure will be output in mmHg

        INPUT: G: Vascular graph in iGraph format.(the pBC should be given in mmHg)
               invivo: boolean if the invivo or invitro empirical functions are used (default = 0)
               withRBC: = 0: no RBCs, pure plasma Flow (default)
                        0 < withRBC < 1 & 'htt' not in edgeAttributes: the given value is assigned as htt to all edges.
                        0 < withRBC < 1 & 'htt' in edgeAttributes: the given value is assigned as htt to all edges where htt = None.
                        NOTE: Htd will be computed from htt and used to compute the resistance.
                            If htd is already in the edge attributes, it will be overwritten.
                dMin_empiricial: lower limit for the diameter that is used to compute nurel (effective viscosity). The aim of the limit
                        is to avoid using the empirical equations in a range where no data exists (default = 3.5).
                htdMax_empirical: upper limit for htd that is used to compute nurel (effective viscosity). The aim of the limit
                        is to avoid using the empirical equations in a range where no data exists (default = 0.6). Maximum has to be 1.
                verbose: Bool if WARNINGS and setup information (INFO) is printed
                diameterOverTime: list of the diameterChanges over time. The length of the list is the number of time steps with diameter change.
                    each diameter change should be provided as a tuple, e.g. two diameterChanges at the same time, a single diameter change afterwards.
                    [[[edgeIndex1, edgeIndex1],[newDiameter1, newDiameter2]],[[edgeIndex3], [newDiameter3]]]
        OUTPUT: None, the edge properties htt is assgined and the function update is executed (see description for more details)
        """
        self._G = G
        nVertices = G.vcount()
        self._b = np.zeros(nVertices)            
        self._A = lil_matrix((nVertices,nVertices),dtype=float)
        self._eps = np.finfo(float).eps
        self._P = Physiology(G['defaultUnits'])
        self._muPlasma = self._P.dynamic_plasma_viscosity()
        self._withRBC = withRBC
        self._invivo = invivo
        self._verbose = verbose
        self._dMin_empirical = dMin_empirical
        self._htdMax_empirical = htdMax_empirical
        self._diameterOverTime = diameterOverTime
        self._timeSteps = len(diameterOverTime)
        self._scalingFactor = vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])

        if self._verbose:
            print('INFO: The limits for the compuation of the effective viscosity are set to')
            print('Minimum diameter %.2f' %self._dMin_empirical)
            print('Maximum discharge %.2f' %self._htdMax_empirical)

        if self._withRBC != 0:
            if self._withRBC < 1.:
                if 'htt' not in G.es.attribute_names():
                    G.es['htt']=[self._withRBC]*G.ecount()
                else:
                    httNone = G.es(htt_eq=None).indices
                    if len(httNone) > 0:
                        G.es[httNone]['htt']=[self._withRBC]*len(httNone)
                    else:
                        if self._verbose:
                            print('WARNING: htt is already an edge attribute. \n Existing values are not overwritten!'+\
                                    '\n If new values should be assigned htt has to be deleted beforehand!')
            else:
                print('ERROR: 0 < withRBC < 1')

        if 'rBC' not in G.vs.attribute_names():
            G.vs['rBC'] = [None]*G.vcount()

        if 'pBC' not in G.vs.attribute_names():
            G.vs['pBC'] = [None]*G.vcount()

        #Convert 'pBC' ['mmHG'] to default Units
        for v in G.vs(pBC_ne=None):
            v['pBC']=v['pBC']*self._scalingFactor

        if len(G.vs(pBC_ne=None)) > 0:
            if self._verbose:
                print('INFO: Pressure boundary conditions changed from mmHg --> default Units')

        self.update()
        
    #--------------------------------------------------------------------------    
        
    def update(self,esequence=None):
        """Constructs the linear system A x = b where the matrix A contains the 
        conductance information of the vascular graph, the vector b specifies 
        the boundary conditions and the vector x holds the pressures at the 
        vertices (for which the system needs to be solved). 
        INPUT: esequence: list of edges which need to be updated. Default=None, i.e. all edges will be updated
        OUTPUT: matrix A and vector b
        """
        G = self._G
        A = self._A
        b = self._b
        htt2htd = self._P.tube_to_discharge_hematocrit
        nurel = self._P.relative_apparent_blood_viscosity
        
        # Compute nominal and specific resistance:
        self._update_nominal_and_specific_resistance(esequence=esequence)

        #edge and vertex List that need to be updated
        if esequence is None:
            es = G.es
            vertexList = G.vs
        else:
            es = G.es(esequence)
            vertexList = []
            for edge in esequence:
                e = G.es[edge]
                vertexList.append(e.source)
                vertexList.append(e.target)
            vertexList = [int(v) for v in np.unique(vertexList)]
            vertexList = G.vs[vertexList]

        #if with RBCs compute effective resistance
        if self._withRBC:
            es['htd'] = [min(htt2htd(htt, d, self._invivo), 1.0) for htt,d in zip(es['htt'],es['diameter'])]
            es['effResistance'] =[ e['resistance'] * nurel(max(self._dMin_empirical,e['diameter']),\
                    min(e['htd'],self._htdMax_empirical),self._invivo) for e in es]
            es['conductance']=1/np.array(es['effResistance'])
        else:
            es['conductance'] = [1/e['resistance'] for e in es]
       
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
                for edge in G.incident(i,'all'):
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

    def solve(self, method):
        """Solves the linear system A x = b for the vector of unknown pressures
        x, either using a direct solver (obsolete) or an iterative GMRES solver. From the
        pressures, the flow field is computed.
        INPUT: method: This can be either 'direct' or 'iterative2'
        OUTPUT: None - G is modified in place.
                G_final.pkl & G_final.vtp: are save as output
                sampledict.pkl: is saved as output
        """
        b = self._b
        A = self._A
        G = self._G
        htt2htd = self._P.tube_to_discharge_hematocrit
        
        A = self._A.tocsr()
        if method == 'direct':
            linalg.use_solver(useUmfpack=True)
            x = linalg.spsolve(A, b)
        elif method == 'iterative2':
             ml = rootnode_solver(A, smooth=('energy', {'degree':2}), strength='evolution' )
             M = ml.aspreconditioner(cycle='V')
             # Solve pressure system
             x,info = gmres(A, self._b, tol=1000*self._eps, maxiter=200, M=M)
             if info != 0:
                 print('ERROR in Solving the Matrix')
                 print(info)

        G.vs['pressure'] = x
        G.es['flow'] = [abs(G.vs[edge.source]['pressure'] - G.vs[edge.target]['pressure']) *  \
                        edge['conductance'] for edge in G.es]

        #Default Units - mmHg for pressure
        G.vs['pressure'] = [v['pressure']/self._scalingFactor for v in G.vs]

        if self._withRBC:
	        G.es['v']=[e['htd']/e['htt']*e['flow']/(0.25*np.pi*e['diameter']**2) for e in G.es]
        else:
	        G.es['v']=[e['flow']/(0.25*np.pi*e['diameter']**2) for e in G.es]
        
    #--------------------------------------------------------------------------
    def _update_nominal_and_specific_resistance(self, esequence=None):
        """Updates the nominal and specific resistance of a given edge 
        sequence.
        INPUT: esequence: Sequence of edge indices which have to be updated. If not provided, all 
                   edges are updated.
        OUTPUT: None, the edge properties 'resistance' and 'specificResistance'
                are updated (or created).
        """
        G = self._G

        if esequence is None:
            es = G.es
        else:
            es = G.es(esequence)

        es['specificResistance'] = [128 * self._muPlasma / (np.pi * d**4)
                                        for d in es['diameter']]

        es['resistance'] = [l * sr for l, sr in zip(es['length'],
                                                es['specificResistance'])]

    #--------------------------------------------------------------------------

    def evolve(self):
        """ The flow field is recomputed for changing vessel diameters over time. The changing vessel
        diameters have been provided as input in _init_ (diameterOverTime). 
        OUTPUT: None - G is modified in place.
                sampledict.pkl: which saves the pressure for all vertices over time and
                    flow, diameter and RBC velocity for all edges over time.
        """
        G = self._G
        diameterOverTime = self._diameterOverTime
        flow_time_edges=[]
        diameter_time_edges=[]
        v_time_edges=[]
        pressure_time_edges=[]

        #First iteration for initial diameter distribution
        self.solve('iterative2')
        flow_time_edges.append(G.es['flow'])
        diameter_time_edges.append(G.es['diameter'])
        v_time_edges.append(G.es['v'])
        pressure_time_edges.append(G.vs['pressure'])

        for timeStep in range(self._timeSteps):
            edgeSequence = diameterOverTime[timeStep][0]
            G.es[edgeSequence]['diameter'] = diameterOverTime[timeStep][1]
            self.update(esequence=edgeSequence)
            self.solve('iterative2')
            flow_time_edges.append(G.es['flow'])
            diameter_time_edges.append(G.es['diameter'])
            v_time_edges.append(G.es['v'])
            pressure_time_edges.append(G.vs['pressure'])
            vgm.write_pkl(G,'G_'+str(timeStep)+'.pkl')

        flow_edges_time = np.transpose(flow_time_edges)
        diameter_edges_time = np.transpose(diameter_time_edges)
        v_edges_time = np.transpose(v_time_edges)
        pressure_edges_time = np.transpose(pressure_time_edges)

        #Write Output
        sampledict={}
        sampledict['flow']=flow_edges_time
        sampledict['diameter']=diameter_edges_time
        sampledict['v']=v_edges_time
        sampledict['pressure']=pressure_edges_time
        g_output.write_pkl(sampledict, 'sampledict.pkl')

        #Convert 'pBC' from default Units to mmHg
        pBCneNone=G.vs(pBC_ne=None)
        pBCneNone['pBC']=np.array(pBCneNone['pBC'])*(1/self._scalingFactor)
        if len(pBCneNone) > 0:
            if self._verbose:
                print('INFO: Pressure boundary conditions changed from default Units --> mmHg')

    #-----------------------------------------------------------------------------

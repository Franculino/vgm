from __future__ import division
from sys import stdout
import igraph as ig
import itertools
#from pyamg import ruge_stuben_solver, smoothed_aggregation_solver
from pyamg import smoothed_aggregation_solver, rootnode_solver, util
import numpy as np
from scipy.sparse.linalg import gmres
from scipy import array, finfo, ones, sparse, zeros
from scipy.sparse import lil_matrix, linalg
import copy
from physiology import Physiology
import g_output
import vgm

__all__ = ['LinearSystemPries']
log = vgm.LogDispatcher.create_logger(__name__)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class LinearSystemPries(object):
    """Solves a VascularGraph for pressure and flow. The influence of red blood
    cells is taken explicitly into account.
    This is an iterative method consisting of two main parts: the linear
    analysis and the rheological analysis. In the linear analysis part (which
    gives this class its name), a linear system Ax = b is constructed from
    current vessel conduction values and solved to yield pressure and flow.
    In the rheological analysis part, a hematocrit redistribution is performed
    based on current flow values and the empirical relations found by Pries et
    al. (1990).
    Note that the method as a whole is non-linear.
    """
    def __init__(self, G, invivo=True,assert_pBCs=True, resetHtd=True,**kwargs):
        """Initializes a LinearSystemPries instance.
        INPUT: G: Vascular graph in iGraph format.
               invivo: Boolean, whether the physiological blood characteristics 
                       are calculated using the invivo (=True) or invitro (=False)
                       equations
               assert_pBCs: (Optional, default=True.) Boolean whether or not to
                            check components for correct pressure boundary
                            conditions.
               resetHtd: (Optional, default=True.) Boolean whether or not to
                         reset the discharge hematocrit of the VascularGraph at
                         initialization. It may be useful to preserve the
                         original htd-distribution, if only a minor change from
                         the current state is to be expected (faster to obtain
                         the solution).
	       **kwargs:
		    httBC: tube hematocrit boundary condition at inflow (edge)
		    htdBC: discharge hematocrit boundary condition at inflow (vertex)
                    plasmaType: if it is not given, the default value is used. option two: --> francesco: plasma value of francescos simulations
                    species: what type of animal we are dealing with --> relevant for the rbc volume that is used, default is rat
		    
        OUTPUT: None
        """
        self._G = G
        self._invivo=invivo
        self._P = Physiology(G['defaultUnits'])
        self._eps = 1e-7 #finfo(float).eps * 1e4
	htt2htd=self._P.tube_to_discharge_hematocrit

        if kwargs.has_key('httBC'):
            for vi in G['av']:
                for ei in G.adjacent(vi):
                    G.es[ei]['httBC']=kwargs['httBC']
		    htdBC = htt2htd(kwargs['httBC'],G.es[ei]['diameter'],invivo)
	        G.vs[vi]['htdBC']=htdBC
	if kwargs.has_key('htdBC'):
	    for vi in G['av']:
		G.vs[vi]['htdBC']=kwargs['htdBC']	

	if kwargs.has_key('plasmaType'):
            self._plasmaType=kwargs['plasmaType']
        else:
            self._plasmaType='default'

	if kwargs.has_key('species'):
            self._species=kwargs['species']
        else:
            self._species='rat'

        # Discharge hematocrit boundary conditions:
        if not 'htdBC' in G.vs.attribute_names():
            for vi in G['av']:
                htdlist = []
                for ei in G.adjacent(vi):
                    if 'httBC' in G.es.attribute_names():
                        if G.es[ei]['httBC'] != None:
		            htdBC = htt2htd(G.es[ei]['httBC'],G.es[ei]['diameter'],invivo)
	                    G.vs[vi]['htdBC']=htdBC
                        else:
                            for ei in G.adjacent(vi):
                                htdlist.append(self._P.discharge_hematocrit(
                                                       G.es[ei]['diameter'], 'a'))
                                G.vs[vi]['htdBC'] = np.mean(htdlist)
                    else:
                        for ei in G.adjacent(vi):
                            htdlist.append(self._P.discharge_hematocrit(
                                                   G.es[ei]['diameter'], 'a'))
                            G.vs[vi]['htdBC'] = np.mean(htdlist)

        #Convert 'pBC' ['mmHg'] to default Units
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']*vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])

        # Initial RBC flow, hematocrit, conductance, pressure and flow:
        G.vs['pressure'] = [0.0 for v in G.vs]
        G.es['rbcFlow'] = [0.0 for e in G.es]
        if resetHtd:
            G.es['htd'] = [0.0 for e in G.es]
        if not G.vs[0].attributes().has_key('pBC'):
            G.vs[0]['pBC'] = None
        if not G.vs[0].attributes().has_key('rBC'):
            G.vs[0]['rBC'] = None
        nVertices = G.vcount()
        self._b = zeros(nVertices)
        self._A = lil_matrix((nVertices,nVertices),dtype=float)
        self._update_conductance_and_LS(G, assert_pBCs)
        self._linear_analysis('iterative2')
        self._rheological_analysis(None, True, 1.0)
        self._linear_analysis('iterative2')

    #--------------------------------------------------------------------------

    def _update_conductance_and_LS(self, newGraph=None, assert_pBCs=True):
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
               assert_pBCs: (Optional, default=True.) Boolean whether or not to
                            check components for correct pressure boundary
                            conditions.
        OUTPUT: A: Matrix A of the linear system, holding the conductance
                   information.
                b: Vector b of the linear system, holding the boundary
                   conditions.
        """

        if newGraph is not None:
            self._G = newGraph

        G = self._G
        P = self._P
        invivo=self._invivo
        cond = P.conductance
        nublood = P.dynamic_blood_viscosity

        G.es['conductance'] = [cond(e['diameter'], e['length'],
                               nublood(e['diameter'], invivo,discharge_ht=e['htd'],plasmaType=self._plasmaType))
                               for e in G.es]
        G.es['conductance'] = [max(min(c, 1e5), 1e-5)
                               for c in G.es['conductance']]


        A = self._A
        b = self._b

        if assert_pBCs:
            # Ensure that the problem is well posed in terms of BCs.
            # This takes care of unconnected nodes as well as connected
            # components of the graph that have not been assigned a minimum of
           # one pressure boundary condition:
            for component in G.components():
                if all(map(lambda x: x is None, G.vs(component)['pBC'])):
                    i = component[0]
                    G.vs[i]['pBC'] = 0.0

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
	self._G = G

    #--------------------------------------------------------------------------

    def _update_rbc_flow(self, limiter=0.5):
        """Traverses all vertices ordered by pressure from high to low.
        Distributes the red cell flow of the mother vessel to the daughter
        vessels according to an empirical relation.
        Note that plasma and RBC flow are conserved at bifurcations, however,
        discharge hematocrit is not a conservative quantity.
        INPUT: limiter: Limits change from one iteration level to the next, if
                        < 1.0, at limiter == 1.0 the change is unmodified.
        OUTPUT: None, edge properties 'rbcFlow' and 'htd' are modified
                in-place.
        """
        # This limit ensures that the hematocrit stays within the physically
        # possible bounds:
        htdLimit = 0.99

        # Short notation:
        G = self._G
        eps = self._eps
        pse = self._P.phase_separation_effect
        #pse = self._P.phase_separation_effect_step

        # Copy current htd:
	oldRbcFlow = copy.deepcopy(G.es['rbcFlow'])
        G.es['rbcFlow'] = [0.0 for e in G.es]
        G.es['htd'] = [0.0 for e in G.es]

        # Vertices sorted by pressure:
        pSortedVertices = sorted([(v['pressure'], v.index) for v in G.vs],
                                 reverse=True)

        # Loop through vertices, distributing discharge hematocrit:
        for vertexPressure, vertex in pSortedVertices:
            # Determine in- and outflow edges:
            outEdges = []
            inEdges = []
            for neighbor, edge in zip(G.neighbors(vertex, 'all'), G.adjacent(vertex, 'all')):
                if G.vs[neighbor]['pressure'] < vertexPressure - eps:
                    outEdges.append(edge)
                elif G.vs[neighbor]['pressure'] > vertexPressure + eps:
                    inEdges.append(edge)

            # The rbc flow is computed from htdBC and the red cells entering
            # from the mother vessels (if any exist). In case of multiple
            # mother vessels, each is distributed according to the empirical
            # relation. If there are more than two daughter edges, the
            # emperical relation is applied to all possible pairings and the
            # final fractional flow to each daughter is computed in a
            # hierarchical fashion (see below).
            trimmedOutEdges = copy.deepcopy(outEdges)
            for outEdge in outEdges:
                if G.es[outEdge]['flow'] <= eps:
                    trimmedOutEdges.remove(outEdge)
                elif not (G.vs[vertex]['htdBC'] is None):
                    G.es[outEdge]['rbcFlow'] = G.vs[vertex]['htdBC'] * \
                                               G.es[outEdge]['flow']
                    G.es[outEdge]['htd'] = G.vs[vertex]['htdBC']
                    trimmedOutEdges.remove(outEdge)

            # Only edges without hematocrit BC are included in the distribution
            # algorithm:
            outEdges = copy.deepcopy(trimmedOutEdges)
	    NoOutEdges=len(outEdges)

            if len(outEdges) == 0:
                continue
            elif len(outEdges) == 1:
                outEdge = outEdges[0]
                rbcFlowIn = 0.0
		FlowIn = 0.0
                for inEdge in inEdges:
                    rbcFlowIn += G.es[inEdge]['rbcFlow']
		    FlowIn += G.es[inEdge]['flow']
		if len(inEdges) == 0:
		    G.es[outEdge]['htd']=G.es[outEdge]['htdBC']
		    G.es[outEdge]['rbcFlow']=G.es[outEdge]['flow']*G.es[outEdge]['htd']
		else:
                    G.es[outEdge]['rbcFlow'] = rbcFlowIn
		    G.es[outEdge]['htd'] = min(rbcFlowIn/FlowIn,htdLimit)
                if G.es[outEdge]['htd'] < 0:
                    print('ERROR 1 htd smaller than 0')
            else:
                rbcFlowIn = 0.0
                edgepairs = list(itertools.combinations(outEdges, 2))
                for inEdge in inEdges:
                    df = G.es[inEdge]['diameter']
                    htdIn = G.es[inEdge]['htd']
                    rbcFlowIn += G.es[inEdge]['rbcFlow']
                    outFractions = dict(zip(outEdges, [[] for e in outEdges]))
                    for edgepair in edgepairs:
                        oe0, oe1 = G.es[edgepair]
                        flowSum = sum(G.es[edgepair]['flow'])
                        if flowSum > 0.0:
			    relativeValue=oe0['flow']/flowSum
			    #stdout.write("\r oe0/flowSum = %f        \n" % relativeValue)
			    #if oe0['flow']/flowSum > 0.49 and oe0['flow']/flowSum < 0.51:
		                #stdout.write("\r NOW        \n")
                            f0 = pse(oe0['flow'] / flowSum,
                                     oe0['diameter'], oe1['diameter'],
                                    df, htdIn)
                            #f0 = pse(oe0['flow'] / flowSum,
                            #         0.7, 0.7, 0.7, 0.64)

                            f1 = 1 - f0
                        else:
                            f0 = 0
                            f1 = 0
                        outFractions[oe0.index].append(f0)
                        outFractions[oe1.index].append(f1)
                    # Sort out-edges from highest to lowest out fraction and
                    # distribute RBC flow accordingly:
                    sortedOutEdges = sorted(zip(map(sum, outFractions.values()),
                                                outFractions.keys()),
                                            reverse=True)
                    remainingFraction = 1.0
                    for i, soe in enumerate(sortedOutEdges[:-1]):
                        outEdge = soe[1]
                        outFractions[outEdge] = remainingFraction * \
                                               sorted(outFractions[outEdge])[i]
                        remainingFraction -= outFractions[outEdge]
                        remainingFraction = max(remainingFraction, 0.0)

                    outFractions[sortedOutEdges[-1][1]] = remainingFraction

		    #Outflow in second outEdge is calculated by the difference of inFlow and OutFlow first
		    #outEdge
		    count = 0 
                    for outEdge in outEdges:
                        G.es[outEdge]['rbcFlow'] += G.es[inEdge]['rbcFlow'] * \
                                                    outFractions[outEdge]
                        #stdout.write("\r outEdge = %g        \n" %outEdge)
                        #stdout.write("\r G.es[outEdge]['rbcFlow'] = %f        \n" %G.es[outEdge]['rbcFlow'])

		        if count == 0:
                            G.es[outEdge]['rbcFlow'] = oldRbcFlow[outEdge] + \
                                (G.es[outEdge]['rbcFlow'] - oldRbcFlow[outEdge]) * limiter
		        elif count == 1:
		            G.es[outEdge]['rbcFlow']=G.es[inEdge]['rbcFlow']-G.es[outEdges[0]]['rbcFlow']
                        #stdout.write("\r LIMITED: G.es[outEdge]['rbcFlow'] = %f        \n" %G.es[outEdge]['rbcFlow'])
                        count += 1


                # Limit change between iteration levels for numerical
                # stability. Note that this is only applied to the diverging
                # bifurcations:
                for outEdge in outEdges:
                    if G.es[outEdge]['flow'] > eps:
                        G.es[outEdge]['htd'] = min(G.es[outEdge]['rbcFlow'] / \
                                                G.es[outEdge]['flow'], htdLimit)

                        if G.es[outEdge]['htd'] < 0:
                            print('ERROR 2 htd smaller than 0')
                        #stdout.write("\r outEdge = %g        \n" %outEdge)
                        #stdout.write("\r outFractions[outEdge] = %f        \n" %outFractions[outEdge])
                        #stdout.write("\r G.es[outEdge]['flow'] = %f        \n" %G.es[outEdge]['flow'])
                        #stdout.write("\r G.es[outEdge]['htd'] = %f        \n" %G.es[outEdge]['htd'])

    #--------------------------------------------------------------------------

    def _linear_analysis(self, method, **kwargs):
        """Performs the linear analysis, in which the pressure and flow fields
        are computed.
        INPUT: method: This can be either 'direct' or 'iterative'
               **kwargs
               precision: The accuracy to which the ls is to be solved. If not
                          supplied, machine accuracy will be used. (This only
                          applies to the iterative solver)
        OUTPUT: The maximum, mean, and median pressure change. Moreover,
                pressure and flow are modified in-place.
        """

        G = self._G
        A = self._A.tocsr()
        if method == 'direct':
            linalg.use_solver(useUmfpack=True)
            x = linalg.spsolve(A, self._b)
        elif method == 'iterative':
            if kwargs.has_key('precision'):
                eps = kwargs['precision']
            else:
                eps = self._eps
            AA = smoothed_aggregation_solver(A, max_levels=10, max_coarse=500)
            x = abs(AA.solve(self._b, x0=None, tol=eps, accel='cg', cycle='V', maxiter=150))
            # abs required, as (small) negative pressures may arise
        elif method == 'iterative2':
         # Set linear solver
             ml = rootnode_solver(A, smooth=('energy', {'degree':2}), strength='evolution' )
             M = ml.aspreconditioner(cycle='V')
             # Solve pressure system
             #x,info = gmres(A, self._b, tol=self._eps, maxiter=50, M=M, x0=self._x)
             #x,info = gmres(A, self._b, tol=self._eps/10000000000000, maxiter=50, M=M)
             x,info = gmres(A, self._b, tol=self._eps/10000, maxiter=50, M=M)
             if info != 0:
                 print('SOLVEERROR in Solving the Matrix')

        pdiff = map(abs, [(p - xx) / p if p > 0 else 0.0
                          for p, xx in zip(G.vs['pressure'], x)])
        maxPDiff = max(pdiff)
        meanPDiff = np.mean(pdiff)
        medianPDiff = np.median(pdiff)
        log.debug(np.nonzero(np.array(pdiff) == maxPDiff)[0])

        G.vs['pressure'] = x
        G.es['flow'] = [abs(G.vs[edge.source]['pressure'] -   \
                            G.vs[edge.target]['pressure']) *  \
                        edge['conductance'] for edge in G.es]

	self._maxPDiff=maxPDiff
        self._meanPDiff=meanPDiff
        self._medianPDiff=medianPDiff

        return maxPDiff, meanPDiff, medianPDiff

    #--------------------------------------------------------------------------

    def _rheological_analysis(self, newGraph=None, assert_pBCs=True,
                              limiter=0.5):
        """Performs the rheological analysis, in which the discharge hematocrit
        and apparent viscosity (and thus the new conductances of the vessels)
        are computed.
        INPUT: newGraph: Vascular graph in iGraph format to replace the
                         previous self._G. (Optional, default=None.)
               assert_pBCs: (Optional, default=True.) Boolean whether or not to
                            check components for correct pressure boundary
                            conditions.
               limiter: Limits change from one iteration level to the next, if
                        > 1.0, at limiter == 1.0 the change is unmodified.
        OUTPUT: None, edge properties 'htd' and 'conductance' are modified
                in-place.
        """
        self._update_rbc_flow(limiter)
        self._update_conductance_and_LS(newGraph, assert_pBCs)

    #--------------------------------------------------------------------------

    def solve(self, method, precision=None, maxIterations=1e4, limiter=0.5,
              **kwargs):
        """Solve for pressure, flow, hematocrit and conductance by iterating
        over linear and rheological analysis. Stop when either the desired
        accuracy has been achieved or the maximum number of iterations have
        been performed.
        INPUT: method: This can be either 'direct' or 'iterative'
               precision: Desired precision, measured in the maximum amount by
                          which the pressure may change from one iteration
                          level to the next. If the maximum change is below
                          threshold, the iteration is aborted.
               maxIterations: The maximum number of iterations to perform.
               limiter: Limits change from one iteration level to the next, if
                        > 1.0, at limiter == 1.0 the change is unmodified.
               **kwargs
               precisionLS: The accuracy to which the ls is to be solved. If
                            not supplied, machine accuracy will be used. (This
                            only applies to the iterative solver)
        OUTPUT: None, the vascular graph is modified in-place.
        """
        G = self._G
        P = self._P
	invivo=self._invivo
        if precision is None: precision = self._eps
        iterationCount = 0
        maxPDiff = 1e200
        filename = 'iter_'+str(iterationCount)+'.vtp'
        g_output.write_vtp(G, filename, False)
        filenames = [filename]

        convergenceHistory = []
        while iterationCount < maxIterations and maxPDiff > precision:
            stdout.write("\rITERATION %g \n" %iterationCount)
            self._rheological_analysis(None, False, limiter=0.5)
            maxPDiff, meanPDiff, medianPDiff = self._linear_analysis(method, **kwargs)
	    #maxPDiff=self._maxPDiff
	    #meanPDiff=self._meanPDiff
	    #medianPDiff=self._medianPDiff
            log.info('Iteration %i of %i' % (iterationCount+1, maxIterations))
            log.info('maximum pressure change: %.2e' % maxPDiff)
            log.info('mean pressure change: %.2e' % meanPDiff)
            log.info('median pressure change: %.2e\n' % medianPDiff)
            convergenceHistory.append((maxPDiff, meanPDiff, medianPDiff))
            iterationCount += 1
            G.es['htt'] = [P.discharge_to_tube_hematocrit(e['htd'], e['diameter'],invivo)
                           for e in G.es]
            vrbc = P.rbc_volume(self._species)
            G.es['nMax'] = [np.pi * e['diameter']**2 / 4 * e['length'] / vrbc
                            for e in G.es]
            G.es['minDist'] = [e['length'] / e['nMax'] for e in G.es]
            G.es['nRBC'] =[e['htt']*e['length']/e['minDist'] for e in G.es]
            filename = 'iter_'+str(iterationCount)+'.vtp'
            g_output.write_vtp(G, filename, False)
            filenames.append(filename)
        if iterationCount >= maxIterations:
            stdout.write("\rMaximum number of iterations reached\n")
        elif maxPDiff <= precision :
            stdout.write("\rPrecision limit is reached\n")
        self.integrity_check()
        G.es['htt'] = [P.discharge_to_tube_hematocrit(e['htd'], e['diameter'],invivo)
                       for e in G.es]
        vrbc = P.rbc_volume(self._species)
        G.es['nMax'] = [np.pi * e['diameter']**2 / 4 * e['length'] / vrbc
                        for e in G.es]
        G.es['minDist'] = [e['length'] / e['nMax'] for e in G.es]
	G.es['nRBC'] =[e['htt']*e['length']/e['minDist'] for e in G.es]
	G.es['v']=[4 * e['flow'] * P.velocity_factor(e['diameter'], invivo, tube_ht=e['htt']) /
                     (np.pi * e['diameter']**2) if e['htt'] > 0 else
                     4 * e['flow'] / (np.pi * e['diameter']**2)
                     for e in G.es]

        for v in G.vs:
            v['pressure']=v['pressure']/vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])
        filename = 'iter_final.vtp'
        g_output.write_vtp(G, filename, False)
        filenames.append(filename)
        g_output.write_pvd_time_series('sequence.pvd', filenames)
        vgm.write_pkl(G, 'G_final.pkl')

	stdout.flush()
        return convergenceHistory
        return G

    #--------------------------------------------------------------------------

    def integrity_check(self):
        """Asserts that mass conservation is honored by writing the sum of in-
        and outflow as well as the sum of the in- and outgoing discharge
        hematocrit to the graph vertices as 'flowSum' and 'htdSum'
        INPUT: None
        OUTPUT: None, flow and htd sums added as vertex properties in-place.
        """
        G = self._G
        eps = self._eps
        G.es['plasmaFlow'] = [(1 - e['htd']) * e['flow'] for e in G.es]
        for v in G.vs:
            vertex = v.index
            vertexPressure = v['pressure']
            outEdges = []
            inEdges = []
            for neighbor, edge in zip(G.neighbors(vertex, 'all'),
                                      G.adjacent(vertex, 'all')):
                if G.vs[neighbor]['pressure'] < vertexPressure:
                    outEdges.append(edge)
                elif G.vs[neighbor]['pressure'] >= vertexPressure:
                    inEdges.append(edge)
            v['flowSum'] = sum(G.es[outEdges]['flow']) - \
                           sum(G.es[inEdges]['flow'])
            v['rbcFlowSum'] = sum(G.es[outEdges]['rbcFlow']) - \
                              sum(G.es[inEdges]['rbcFlow'])
            v['plasmaFlowSum'] = sum(G.es[outEdges]['plasmaFlow']) - \
                              sum(G.es[inEdges]['plasmaFlow'])
        for e in G.es:
            e['errorHtd'] = abs(e['rbcFlow'] / e['flow'] - e['htd']) \
                            if e['flow'] > eps else 0.0
        log.info('Max error htd: %.1g' % max(G.es['errorHtd']))

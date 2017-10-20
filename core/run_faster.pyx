# /ython: profile=True
#from __future__ import division, with_statement

from copy import deepcopy
import numpy as np
import cython
cimport numpy as np

__all__ = ['update_timestep','update_flow_and_v']

# -----------------------------------------------------------------------------
def update_timestep(graph,double eps,vi,double dt,eiIn):

    """ Calculate Next timestep
    """

    G = graph
    eiInOldAtDtZero = -1 if dt > 0.0 else eiIn
    blockedEdges = []
    dtmin = 1e25
    tlist = dict([(i, 1e25) for i in xrange(G.ecount())])

    for eIn in G.es:
        ei = eIn.index
        vi = eIn.target if eIn['sign'] == 1 else eIn.source
        outEdges = G.vs[vi]['outflowE']
        # No flow / no RBC are not time limiting:
        #             #if eIn['sign'] ==0 or len(eIn['rRBC']) ==0: --> is sign ever ==0?
        if len(eIn['rRBC']) ==0:
            tlist[ei] = 1e25
            continue
        # Outflow edges are limiting, as RBCs need to be removed:
        elif len(outEdges) == 0:
            #if len(eIn['rRBC']) > 0:
            s = eIn['rRBC'][0] if eIn['sign'] == -1 else eIn['length'] - eIn['rRBC'][-1]
            dt = s / eIn['v']
            tlist[ei] = dt
            continue
 
        # Note that an in-edge that has only blocked out-edges is not
        # blocked itself, as an RBC can still move freely with the flow.
        distToBifurcation = eIn['rRBC'][0] if eIn['sign'] == -1 else eIn['length'] - eIn['rRBC'][-1]

        # If leading RBC at bifurcation, check whether it has free passage
        # to one of the out-edges or whether it is constrained by Ht-limits
        # (note that it can also enter a blocked out-edge):
        if distToBifurcation <= 0.0+eps:
            timeLimits = []
            for outEdge in outEdges:
                e = G.es[outEdge]
                # If outEdge == inEdge of the previous timestep, and that
                # timestep was zero - we have a case of flip-flopping which 
                # needs to be avoided:
                if outEdge == eiInOldAtDtZero:
                    blockedEdges.append(ei)
                    eIn['free'] = []
                    tlist[ei] = dt
                    break
                # If outEdge is devoid of RBCs, it does not constrain flow
                # in inEdge:
                if len(e['rRBC']) == 0:
                    dt = 0.0
                    tlist[ei] = dt
                    break
                else:
                    # Distance from vertex vi to first RBC in outEdge:
                    s = e['rRBC'][0] if e['sign'] == 1 else e['length'] - e['rRBC'][-1]
                    if s < e['minDist'] - eps:
                        timeLimits.append((e['minDist']-s) / e['v'])
                    else:
                        dt = 0.0
                        tlist[ei] = dt
                        break
                # If the RBC causing the blockage cannot be reassigned, compute
                # the time it takes for the first of the free RBCs to reach the 'traffic jam': 
                # TODO: Man koennte eventuel direkt alle differenzen zwischen i und i-1 berechnen 
                # und dann erste die groesser als lrbc+eps ist raussuchen
            if len(timeLimits) == len(outEdges):
                blockedEdges.append(ei)
                e = G.es[ei]
                lrbc = e['minDist']
                e['free'] = [] # The default, i.e. all RBCs are blocked
                rrbc = e['rRBC']
                if e['sign'] == 1:
                    for i in range(len(rrbc)-1, 0, -1):
                        j = i - 1
                        if rrbc[i] - rrbc[j] > lrbc + eps:
                            timeLimits.append((rrbc[i] - rrbc[j] - lrbc) / e['v'])
                            e['free'] = range(i)
                            break
                else:
                    for i in range(len(rrbc)-1):
                        j = i + 1
                        if rrbc[j] - rrbc[i] > lrbc + eps:
                            timeLimits.append((rrbc[j] - rrbc[i] - lrbc) / e['v'])
                            e['free'] = range(j, len(rrbc))
                            break
                dt = min(timeLimits)
                tlist[ei] = dt
                # If leading RBC is not at bifurcation, determine the time it needs to reach it:
        else:
            dt = (eIn['length'] - eIn['rRBC'][-1]) / eIn['v'] \
                if eIn['sign'] == 1 else eIn['rRBC'][0] / eIn['v']
            tlist[ei] = dt
    dtmin, dtminEdgeIndex = sorted(zip(tlist.values(), tlist.keys()))[0]
    # Account for numerical inaccuracy:
    dtmin = max(0.0, dtmin)


    e = G.es[dtminEdgeIndex]
    if e['sign'] == 1:
        vi = e.target
    else:
        vi = e.source

    #if (dtminEdgeIndex in blockedEdges and dtmin < 1e-14):
    #    pdb.set_trace()
    return G,blockedEdges,vi,dtmin,dtminEdgeIndex

# # -----------------------------------------------------------------------------
#
def update_flow_and_v(graph,invivo,vfList,vrbc):
        """Updates the flow and red blood cell velocity in all vessels
        INPUT: None
        OUTPUT: None
        """
        G=graph
        #Aufruf von Physiology macht alles extrem langsam. Wie wird das gut gemacht? Ist das moeglich?
        #P = Physiology(G['defaultUnits'])
        #vf = P.velocity_factor
        pi=np.pi
        G.es['flow'] = [abs(G.vs[e.source]['pressure'] -
                            G.vs[e.target]['pressure']) /res
                            for e,res in zip(G.es,G.es['effResistance'])]
        G.es['vBulk']=[e['flow']/e['crosssection'] for e in G.es]

        # RBC velocity is not defined if tube_ht==0, using plasma velocity
        # instead:
        G.es['v'] = [vBulk * vf if htt > 0 else vBulk
                     for htt,vBulk,vf in zip(G.es['htt'],G.es['vBulk'],vfList)]

        #RBC Flow in mum^3/ms
        #rbcFlowOut is used to calculate the new nRBC, can be limited in case of traffic jams
        #rbcFlow, ist the actual rbcFlowRate resulting from flow*htd
        G.es['rbcFlow'] = [flow*htd for flow,htd in zip(G.es['flow'],G.es['htd'])]
        #RBC Flow 2 in RBCs/ms
        G.es['rbcFlow2'] = [rbcFlow/vrbc for rbcFlow in G.es['rbcFlow']]

        return G


# # -----------------------------------------------------------------------------
def update_rbcFlowIn(graph):
        """Updates the flow and red blood cell velocity in all vessels
        INPUT: None
        OUTPUT: None
        """
        G=graph
        #edges=[]
        #inflow1=[]
        #inflow2=[]
        #for i in G['conV']:
        #    edges.append(G.vs['outflowE'][i][0])
        #    inflow1.append(G.vs['inflowE'][i][0])
        #    inflow2.append(G.vs['inflowE'][i][1])
            #G.es[G.vs['outflowE'][i][0]]['rbcFlowIn']=G.es[G.vs['inflowE'][i][0]]['rbcFlowOut']+G.es[G.vs['inflowE'][i][1]]['rbcFlowOut']
        #G.es[edges]['rbcFlowIn']=np.array(G.es[inflow1]['rbcFlowOut'])+np.array(G.es[inflow2]['rbcFlowOut'])

        #edges2=[]
        #inflow3=[]
        for i in G['connectV']:
            #edges2.append(G.vs['outflowE'][i][0])
            #inflow3.append(G.vs['inflowE'][i][0])
            G.es[G.vs['outflowE'][i][0]]['rbcFlowIn']=G.es[G.vs['inflowE'][i][0]]['rbcFlowOut']
        #G.es[edges2]['rbcFlowIn']=G.es[inflow3]['rbcFlowOut']

        #adjacents=[G.vs[i]['adjacent'][0] for i in G['av']]
        #for i in G['av']:
        #     G.es[G.vs[i]['adjacent'][0]]['rbcFlowIn'] = G.es['flow'][G.vs[i]['adjacent'][0]]*G.es['htdBC'][G.vs[i]['adjacent'][0]]
        #G.es[adjacents]['rbcFlowIn'] = np.array(G.es[adjacents]['flow'])*np.array(G.es[adjacents]['htdBC'])

        return G

# # -----------------------------------------------------------------------------
def propagate(graph,dt,eiIn,vi,eps,blockedEdges,transitTimeDict,inflowTracker,tPlot):
        """This assigns the current bifurcation-RBC to a new edge and
        propagates all RBCs until the next RBC reaches at a bifurcation.
        INPUT: None
        OUTPUT: None
        """

        G=graph
        # Move all RBCs with their current velocity for a time dt (which is
        # when the first RBC will reach a bifurcation or a blocked RBC in the
        # case of a blocked edge):
        update_htt_in=[]
        if dt > 0.0:
            for e in G.es:
                ei = e.index
                displacement = e['v'] * dt
                if ei in blockedEdges:
                    if len(e['free']) > 0:
                        e['rRBC'][e['free']] = e['rRBC'][e['free']] + \
                                               displacement * e['sign']
                else:
                    e['rRBC'] = e['rRBC'] + displacement * e['sign']

                if len(e['tRBC']) > 0:
                    e['tRBC'] += dt

                if e['httBC'] is not None:
                    rRBC = []
                    tRBC = []
                    lrbc = e['minDist']
                    htt = e['httBC']
                    length = e['length']
                    inflowTrackerEi = inflowTracker[ei]
                    cum_length = inflowTrackerEi[0] + displacement

                    if cum_length >= lrbc:
                        nrbc_max = cum_length / lrbc
                        nrbc_max_floor = np.floor(nrbc_max)
                        nrbc = sum(np.random.rand(nrbc_max_floor)<htt)
                        lrbc_modulo = (nrbc_max - nrbc_max_floor) * lrbc
                        start = lrbc_modulo
                        stop = cum_length
                        for i in range(nrbc):
                            pos = np.random.rand() * (stop - ((nrbc-i)*lrbc) - start) + start
                            start = pos + lrbc
                            rRBC.append(pos)
                        rRBC = np.array(rRBC)
                        tRBC = rRBC/e['v']
                        if e['sign'] == 1:
                            e['rRBC'] = np.concatenate([rRBC, e['rRBC']])
                            e['tRBC'] = np.concatenate([tRBC, e['tRBC']])
                        else:
                            e['rRBC'] = np.concatenate([e['rRBC'], length-rRBC[::-1]])
                            e['tRBC'] = np.concatenate([e['tRBC'],tRBC[::-1]])
                        if nrbc > 0:
                            update_htt_in.append(ei)
                            #self._update_tube_hematocrit((ei))
                            inflowTrackerEi[1] = True
                        else:
                            inflowTrackerEi[1] = False
                        inflowTrackerEi[0] = lrbc_modulo
                    else:
                        inflowTrackerEi[0] += displacement
        # Assign the current bifurcation-RBC to a new vessel:
        p = G.vs[vi]['pressure']
        outEdges=G.vs[vi]['outflowE']
        outEdge = None
        if len(outEdges) > 0:
            # Choose out-edge preference according to bifurcation rule:
            if G.vs[vi]['isCap']:
                preferenceList = [x[1] for x in
                                  sorted(zip(G.es[outEdges]['flow'],
                                             outEdges), reverse=True)]
            # Choose out-edge preference based on Kirchhoff:
            else:
                preferenceList = []
                remainingOE = deepcopy(outEdges)
                while len(remainingOE) > 1:
                    oe = remainingOE[0]
                    outflow = sum(G.es[remainingOE]['flow'])
                    flowFractions = [e['flow']/outflow
                                     for e in G.es[remainingOE]]
                    intervals = np.cumsum(flowFractions)
                    rand = np.random.rand()
                    selectedEdge = remainingOE[np.nonzero(np.less(rand, intervals))[0][0]]
                    preferenceList.append(selectedEdge)
                    remainingOE.remove(selectedEdge)
                preferenceList.append(remainingOE[0])
            # Assign out-edge based on preference-list and ht-constraints:
            for e in G.es[preferenceList]:
                if len(e['rRBC']) == 0:
                    outEdge = e
                    break
                else:
                    s = e['rRBC'][0] if e.source == vi \
                        else e['length'] - e['rRBC'][-1]
                    if s >= e['minDist'] - eps:
                        outEdge = e
                        break
        # If a designated out-edge exists, add RBC:
        if outEdge is not None:
            e = outEdge
            if e.source == vi:
                e['rRBC'] = np.concatenate([[0.0 + eps], e['rRBC']])
                #Move 'tRBC' value with RBC to new edge
                if len(G.es[eiIn]['tRBC']) > 0 and (len(G.es[eiIn]['rRBC']) == len(G.es[eiIn]['tRBC'])):
                    if G.es[eiIn]['sign'] == 1:
                        e['tRBC']=np.concatenate((G.es[eiIn]['tRBC'][-1::],e['tRBC']))
                    else:
                        e['tRBC']=np.concatenate((G.es[eiIn]['tRBC'][0:1],e['tRBC']))
            else:
                #Move 'tRBC' value with RBC to new edge
                e['rRBC'] = np.concatenate([e['rRBC'], [e['length'] - eps]])
                if len(G.es[eiIn]['tRBC']) > 0 and (len(G.es[eiIn]['rRBC']) == len(G.es[eiIn]['tRBC'])):
                    if G.es[eiIn]['sign'] == 1:
                        e['tRBC']=np.concatenate((e['tRBC'],G.es[eiIn]['tRBC'][-1::]))
                    else:
                        e['tRBC']=np.concatenate((e['tRBC'],G.es[eiIn]['tRBC'][0:1]))
            #self._update_tube_hematocrit((outEdge.index))
            update_htt_in.append(outEdge.index)

        # Remove RBC from mother vessel and save transit time of RBCs
        time = tPlot + dt
        if len(outEdges) > 0 and outEdge == None:
            blockedEdges.append(eiIn)
        else:
            e = G.es[eiIn]
            if e['sign'] == 1:
                if len(e['tRBC']) > 0 and (len(e['rRBC']) == len(e['tRBC'])):
                    if len(outEdges) == 0:
                        if not 'transitTime' in transitTimeDict.keys():
                            transitTimeDict['transitTime']=[]
                        if not 'time' in transitTimeDict.keys():
                            transitTimeDict['time']=[]
                        transitTimeDict['transitTime'].append(e['tRBC'][-1])
                        transitTimeDict['time'].append(time)
                    e['tRBC']=e['tRBC'][:-1]
                e['rRBC'] = e['rRBC'][:-1]
            else:
                if len(e['tRBC']) > 0 and (len(e['rRBC']) == len(e['tRBC'])):
                    if len(outEdges) == 0:
                        if not 'transitTime' in transitTimeDict.keys():
                            transitTimeDict['transitTime']=[]
                        if not 'time' in transitTimeDict.keys():
                            transitTimeDict['time']=[]
                        transitTimeDict['transitTime'].append(e['tRBC'][0])
                        transitTimeDict['time'].append(time)
                    e['tRBC']=e['tRBC'][1:]
                e['rRBC'] = e['rRBC'][1:]
            #self._update_tube_hematocrit((eiIn))
            update_htt_in.append(eiIn)

        G.es['nRBC'] = [len(e['rRBC']) for e in G.es]

        return G,blockedEdges,transitTimeDict,inflowTracker,update_htt_in

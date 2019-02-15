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
import os
import matplotlib.pyplot as plt
from sys import stdout
from scipy.optimize import root
from scipy import finfo
from scipy.integrate import quad
import matplotlib
import gc
from scipy import finfo, ones, zeros
from scipy.interpolate import griddata

__all__ = ['eBlockLabelingOfAll_A_and_V_and_vice_versa','create_trackRootLabel',
    'adjust_SurfPlun_Labels_2','adjust_SurfPlun_Labels',
     'adjust_vertexLabels_to_edgeLabels','adjust_Labels_forNanEdges','introduce_nkind_labels',
     'artificially_increase_length_of_artificialReduceDegreeEdges_based_on_dummy_simulations','assgin_av_vv_pBC',
    'checkForDoubleVertices','eliminate_loop_vertices','dealWithDoubleEdges',
    'eliminateStandard_Degree5_Vertices','deleteUnlabeled_degree1_Vertices',
    'introduce_nkind_labels','eliminate_loop_vertices','improve_capillary_diameters_by_binFitting',
    'improve_capillary_labeling','introduce_minimum_and_maximum_diameter_for_vessel_types',
    'adjust_length_of_vessel_to_fit_oneRBC','cut_off_sides_of_MVN','cut_off_bottom_MVN']
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def eBlockLabelingOfAll_A_and_V_and_vice_versa(G):
        """ 1. The edge attribute 'eBlock' labels all non-capillary vessels. Hence, all
        vessels with eBlock=1 should be either arterioles (medianLabelAV=3) or venules 
        (medianLabelAV=4). The medianLabelAV label is adjusted based on neighboring edges.
        2. There might also be As and Vs which do not have an eBlock Label. 
        Their medianLabelAV will be changed to 1 (=capillaries)

        INPUT: G: Vascular graph in iGraph format.
        OUTPUT: updatedProperties eBlock and medianLabelAV
        """

        print('Improve labeling based on eBlock')
        eBlock=G.es(eBlock_eq=1,medianLabelAV_ne=4).indices
        eBlock2=[] #All edges which have an eBlock=1 label but which are not labeled as arterioles or venules
        for e in G.es[eBlock]:
            if e['medianLabelAV'] != 3:
                eBlock2.append(e.index)

        print('Number of eBlock edges, which are not labeld as A or V')
        print(len(eBlock2))
        while len(eBlock2) != 0:
            print(len(eBlock2))
            for e in G.es[eBlock2]:
                for v in e.tuple:
                    if 3 in G.es[G.incident(v)]['medianLabelAV']:
                        e['medianLabelAV'] = 3
                    elif 4 in G.es[G.incident(v)]['medianLabelAV']:
                        e['medianLabelAV'] = 4
            eBlock=G.es(eBlock_eq=1,medianLabelAV_ne=4).indices
            eBlock2=[]
            for e in G.es[eBlock]:
                if e['medianLabelAV'] != 3:
                    eBlock2.append(e.index)

        #---> NOW: all eBlock edges should be labeled either as arteriole or as venule

        #There might be arterioles and venules, which do not have an eBlock labeling --> their labelig should be changed to capillary
        eBlock=G.es(eBlock_eq=1).indices
        print('Are there arterioles or venules which are not labeled as eBlock')
        print(len(eBlock) < len(G.es(medianLabelAV_eq=3))+len(G.es(medianLabelAV_eq=4)))
        if len(eBlock) < len(G.es(medianLabelAV_eq=3))+len(G.es(medianLabelAV_eq=4)):
            artsWhichAreNotEBlock=G.es(medianLabelAV_eq=3,eBlock_ne=1).indices
            veinsWhichAreNotEBlock=G.es(medianLabelAV_eq=4,eBlock_ne=1).indices
            G.es[artsWhichAreNotEBlock]['medianLabelAV']=[1]*len(artsWhichAreNotEBlock)
            G.es[artsWhichAreNotEBlock]['medianLabelSurfPlun']=[1]*len(artsWhichAreNotEBlock)
            G.es[veinsWhichAreNotEBlock]['medianLabelAV']=[1]*len(artsWhichAreNotEBlock)
            G.es[veinsWhichAreNotEBlock]['medianLabelSurfPlun']=[1]*len(artsWhichAreNotEBlock)
        
        if len(eBlock) != len(G.es(medianLabelAV_eq=3))+len(G.es(medianLabelAV_eq=4)):
            print('ERROR in eBlock and medianLabelAV')

        return G

#------------------------------------------------------------------------------
def adjust_vertexLabels_to_edgeLabels(G, label='AV', valueLabel=1):
        """ Adjusts the label of the tuples of the edges to the value of the edge.
        The attributes for the edge must be 'medianLabelXX', those for the vertices
        'labelXX'
        INPUT: G: Vascular graph in iGraph format.
               label: the label of the edge, vertices (the 'XX')
               valueLabel: the values which are adjusted
        OUTPUT: updatedProperties labelXX
        """
        edges=G.es(**{'medianLabel'+label+'_eq':valueLabel}).indices
        attribute='label'+label
        for e in G.es[edges]:
            G.vs[e.tuple][attribute]=[valueLabel,valueLabel]

        return G
#------------------------------------------------------------------------------
def adjust_Labels_forNanEdges(G, label='AV'):
        """ eliminates Nan lables of edges and assigns valueToAssign. An Error message
        is outputted if the relabeling of edges was not succesfull
        The attributes for the edge must be 'medianLabelXX', those for the vertices
        'labelXX'
        INPUT: G: Vascular graph in iGraph format.
               label: the label of the edge, vertices (the 'XX')
        OUTPUT: updatedProperties labelXX and medianLabelXX
        """
        edgeNan=[]
        for e in G.es:
            if np.isnan(e['medianLabel'+label]):
                edgeNan.append(e.index)
        
        print('Number of edges which have a medianLabelingAV which is Nan')
        print(len(edgeNan))
        edgeNan2=deepcopy(edgeNan)
        for e in G.es[edgeNan]:
            if len(np.unique(G.vs[e.tuple]['label'+label])) == 1:
               e['medianLabel'+label]=G.vs[e.source]['label'+label]
               edgeNan2.remove(e.index)
            else:
                if 1 in G.vs[e.tuple]['degree']:
                    if G.vs[e.source]['degree'] == 1:
                        G.vs[e.source]['label'+label] = G.vs[target]['label'+label]
                        e['medianLabel'+label] = G.vs[e.target]['label'+label]
                        edgeNan2.remove(e.index)
                    else:
                        G.vs[e.target]['label'+label] = G.vs[e.source]['label'+label]
                        e['medianLabel'+label] = G.vs[e.source]['label'+label]
                        edgeNan2.remove(e.index)
                elif 1 in G.vs[e.tuple]['label'+label]:
                    if 3 in G.vs[e.tuple]['label'+label]:
                        e['medianLabel'+label] = G.vs[e.target]['label'+label]
                        edgeNan2.remove(e.index)
                    elif 4 in G.vs[e.tuple]['label'+label]:
                        e['medianLabel'+label] = G.vs[e.target]['label'+label]
                        edgeNan2.remove(e.index)
                    else:
                        print('ERROR in dealing with Nans in medianLabel'+label+' 2')
                else:
                    print('ERROR in dealing with Nans in medianLabel'+label)
        
        if len(edgeNan2) > 0:
            print('ERROR not all Nan edges could be eliminated')

        return G

#------------------------------------------------------------------------------
def adjust_SurfPlun_Labels(G, label=3):
        """ If there are As or Vs which are not labeled as surf of plun this needs to be 
        changed. This function tries to improve the medianSurfPlun labeling, by checking
        if next to degree 2 vertices appropriate medianSurfPlun labels are found which 
        can be extended.
        INPUT: G: Vascular graph in iGraph format.
               label: the label of the vessel type 
        OUTPUT: updatedProperties medianLabelSurfPlun
        """

        G.vs['degree']=G.degree()
        if len(G.es(medianLabelAV_eq=label)) != len(G.es(medianLabelAV_eq=label,medianLabelSurfPlun_eq=3))+len(G.es(medianLabelAV_eq=label,medianLabelSurfPlun_eq=4)):
            notSurfNotPlun=[]
            for e in G.es[G.es(medianLabelAV_eq=label,medianLabelSurfPlun_ne=3).indices]:
                if e['medianLabelSurfPlun'] != 4:
                    notSurfNotPlun.append(e.index)
            countLoopsWithoutChange=0
            lenNSNP=len(notSurfNotPlun)
            while len(notSurfNotPlun) != 0:
                if len(notSurfNotPlun)==lenNSNP:
                    countLoopsWithoutChange += 1
                else:
                    countLoopsWithoutChange=0
                for e in G.es[notSurfNotPlun]:
                    boolMedianLabelFound=0
                    for v in e.tuple:
                        if G.vs[v]['degree']==2:
                            if 4 in G.es[G.incident(v)]['medianLabelSurfPlun']:
                                e['medianLabelSurfPlun'] = 4
                lenNSNP=len(notSurfNotPlun)
                notSurfNotPlun=[]
                for e in G.es[G.es(medianLabelAV_eq=label,medianLabelSurfPlun_ne=3).indices]:
                    if e['medianLabelSurfPlun'] != 4:
                        notSurfNotPlun.append(e.index)
                if countLoopsWithoutChange >= 1000:
                    break

        return G
#------------------------------------------------------------------------------
def adjust_SurfPlun_Labels_2(G):
        """ There might be vessels labeled as surfPlun even if they are not As or Vs. 
        All those vessels are changed to medianLabelSurfPlun =1
        INPUT: G: Vascular graph in iGraph format.
        OUTPUT: updatedProperties medianLabelSurfPlun
        """

        if len(G.es(medianLabelAV_eq=4))+len(G.es(medianLabelAV_eq=3)) != len(G.es(medianLabelSurfPlun_eq=4))+len(G.es(medianLabelSurfPlun_eq=3)):
            surfWhichAreNotAorV=G.es(medianLabelSurfPlun_eq=3,medianLabelAV_ne=3).indices
            surfChangeLabels=[]
            for e in G.es[surfWhichAreNotAorV]:
                if e['medianLabelAV'] != 4:
                    surfChangeLabels.append(e.index)
            G.es[surfChangeLabels]['medianLabelSurfPlun']=[1]*len(surfChangeLabels)
        
            plunWhichAreNotAorV=G.es(medianLabelSurfPlun_eq=4,medianLabelAV_ne=3).indices
            plunChangeLabels=[]
            for e in G.es[plunWhichAreNotAorV]:
                if e['medianLabelAV'] != 4:
                    plunChangeLabels.append(e.index)
            G.es[plunChangeLabels]['medianLabelSurfPlun']=[1]*len(plunChangeLabels)
        
        if len(G.es(medianLabelAV_eq=4))+len(G.es(medianLabelAV_eq=3)) != len(G.es(medianLabelSurfPlun_eq=4))+len(G.es(medianLabelSurfPlun_eq=3)):
            print('ERROR in improving LabelSurfPlun')

        return G
#------------------------------------------------------------------------------
def create_trackRootLabel(G):
        """ The property trackRoot is created. It is assigned at A and V vertices, where ther is a neighboring C vertex 
        it can happen that trackRoot is assigned at degree 2 vertices. This is not realistic and hence the labelAV is 
        adjusted such, that trackRoot only exists vertices with degree > 2.
        INPUT: G: Vascular graph in iGraph format.
        OUTPUT: property trackRoot is created
        """
        G.vs['trackRoot']=[0]*G.vcount()
        arts=G.vs(labelAV_eq=3).indices
        for i in arts:
            v=G.vs[i]
            if 1 in G.vs[G.neighbors(i)]['labelAV']:
                v['trackRoot'] = 1
        
        veins=G.vs(labelAV_eq=4).indices
        for i in veins:
            v=G.vs[i]
            if 1 in G.vs[G.neighbors(i)]['labelAV']:
                v['trackRoot'] = 1

        probs=G.vs(trackRoot_eq=1,degree_lt=3,labelAV_eq=4).indices
        for i in probs:
            doneVerts=[]
            G.vs[i]['labelAV']=1
            treeVertices2=[i]
            while treeVertices2 != []:
                treeVertices=[]
                for j in treeVertices2:
                    neighbors=G.neighbors(j)
                    adjacent=G.incident(j)
                    for k,l in zip(neighbors,adjacent):
                        if G.vs[k]['labelAV'] == 4 and k not in doneVerts and G.vs[k]['degree'] == 2:
                            treeVertices.append(k)
                            G.es[l]['medianLabelAV']=1
                            G.vs[k]['labelAV']=1
                        elif G.vs[k]['labelAV'] == 4 and k not in doneVerts and G.vs[k]['degree'] == 3:
                            G.es[l]['medianLabelAV']=1
                    doneVerts.append(j)
                treeVertices2=deepcopy(treeVertices)
        
        probs=G.vs(trackRoot_eq=1,degree_lt=3,labelAV_eq=3).indices
        for i in probs:
            doneVerts=[]
            G.vs[i]['labelAV']=1
            treeVertices2=[i]
            while treeVertices2 != []:
                treeVertices=[]
                for j in treeVertices2:
                    neighbors=G.neighbors(j)
                    adjacent=G.incident(j)
                    for k,l in zip(neighbors,adjacent):
                        if G.vs[k]['labelAV'] == 3 and k not in doneVerts and G.vs[k]['degree'] == 2:
                            treeVertices.append(k)
                            G.es[l]['medianLabelAV']=1
                            G.vs[k]['labelAV']=1
                        elif G.vs[k]['labelAV'] == 3 and k not in doneVerts and G.vs[k]['degree'] == 3:
                            G.es[l]['medianLabelAV']=1
                    doneVerts.append(j)
                treeVertices2=deepcopy(treeVertices)

        G.vs['trackRoot']=[0]*G.vcount()
        arts=G.vs(labelAV_eq=3).indices
        for i in arts:
            v=G.vs[i]
            for j in G.vs[G.neighbors(i)]:
                if j['labelAV'] == 1:
                    v['trackRoot'] = 1
        
        veins=G.vs(labelAV_eq=4).indices
        for i in veins:
            v=G.vs[i]
            for j in G.vs[G.neighbors(i)]:
                if j['labelAV'] == 1:
                    v['trackRoot'] = 1
        
        if len(G.vs(trackRoot_eq=1)) != len(G.vs(trackRoot_eq=1,degree_ge=3)):
            print('ERROR in trackRoot labeling')

        return G

#------------------------------------------------------------------------------
def assgin_av_vv_pBC(G,pBCIn=80,pBCOut=20):
        """ 
        Prepares the Graph for simulation: assigns av,vv und pBC. Based on degree and labelAV
        INPUT: G: Vascular graph in iGraph format.
        OUTPUT: properties av, vv and pBC are created
        """
        G.vs['degree']=G.degree()
        deg1=G.vs(degree_eq=1).indices
        deg1A=G.vs(degree_eq=1,labelAV_eq=3).indices
        deg1V=G.vs(degree_eq=1,labelAV_eq=4).indices
        G.vs[deg1A]['pBC']=[pBCIn]*len(deg1A)
        G.vs[deg1V]['pBC']=[pBCOut]*len(deg1V)
        G.vs['av']=[0]*G.vcount()
        G.vs['vv']=[0]*G.vcount()
        G.vs[deg1A]['av']=[1]*len(deg1A)
        G.vs[deg1V]['vv']=[1]*len(deg1V)
        G['av']=G.vs(av_eq=1).indices
        G['vv']=G.vs(vv_eq=1).indices

        return G

#------------------------------------------------------------------------------
def checkForDoubleVertices(G):
        """ 
        Checks if there are vertices with the same coordinates
        INPUT: G: Vascular graph in iGraph format.
        OUTPUT: allDoubles: list wiht all vertices for which a vertex with similar
        coordinates exists
        """
        x=[]
        y=[]
        z=[]
        for i in G.vs:
            x.append(i['r'][0])
            y.append(i['r'][1])
            z.append(i['r'][2])
        
        xUnique=np.unique(x)
        x=np.array(x)
        y=np.array(y)
        z=np.array(z)
        G.vs['z']=z
        
        doubleVertices=[]
        allDoubles=[]
        iter =0
        for i in xUnique:
            searchval = i
            ii = np.where(x == searchval)[0]
            count = 0
            for j in ii:
                coords=[x[j],y[j],z[j]]
                count += 1
                same=[j]
                for k in ii[count::]:
                    if coords[1] == y[k] and coords[2] == z[k]:
                        same.append(k)
                if len(same) > 1 and j not in allDoubles:
                    doubleVertices.append(same)
                    for k in same:
                        allDoubles.append(k)
            iter += 1
        
        if len(doubleVertices) > 0:
            print('There are doubleVertices')
            print(len(doubleVertices))
        
        allDoubles=[]
        for i in doubleVertices:
            for j in i:
                allDoubles.append(j)

        return allDoubles,G
#------------------------------------------------------------------------------
def dealWithDoubleEdges(G):
        """ 
        Checks if there are two edges between the same vertex and deletes one of
        them if this is the case
        INPUT: G: Vascular graph in iGraph format.
        OUTPUT:
        """
        verticesWithDoubleEdges=[]
        print('len double edges')
        for i in range(G.vcount()):
            if len(G.neighbors(i)) != len(np.unique(G.neighbors(i))):
                verticesWithDoubleEdges.append(i)
        
        print(len(verticesWithDoubleEdges))
        
        #one of the double edges is deleted
        for i in verticesWithDoubleEdges:
            neighbors=[]
            incidents=G.incident(i)
            for k,j in enumerate(G.neighbors(i)):
                if j in neighbors:
                    G.delete_edges(incidents[k])
                else:
                    neighbors.append(j)
        
        verticesWithDoubleEdges=[]
        print('len double edges - 2')
        for i in range(G.vcount()):
            if len(G.neighbors(i)) != len(np.unique(G.neighbors(i))):
                verticesWithDoubleEdges.append(i)
        
        print(len(verticesWithDoubleEdges))
        return G
#------------------------------------------------------------------------------
def eliminateStandard_Degree5_Vertices(G):
        """ 
        Splits all degree 5 vertices. A pressure field needs to be available for the splitting
        process. The newly introduced edges receive the edge attribute artificialReduceDegreeEdge=1
        INPUT: G: Vascular graph in iGraph format.
        OUTPUT:
        """
        eps = finfo(float).eps*1000000
        #Degree = 5 withouth double vertices
        probs=G.vs(degree_eq=5,corticalDepth_lt=1300).indices
        #Case1: 1 inflows --> 4 outflows --> split 4 outflows in two times two 
        #Case2: 2 inflows --> 3 outflows --> add edge between in and outflows 
        #Case3: 3 inflows --> 2 outflows --> add edge between in and outflows 
        #Case3: 4 inflows --> 1 outflows --> add edge between in and outflows 
        case=[]
        for i in probs:
            pV=G.vs['pressure'][i]
            inV=[]
            inE=[]
            outV=[]
            outE=[]
            pOuts=[]
            for j,k in zip(G.neighbors(i),G.incident(i)):
                if G.vs[j]['pressure'] > pV+eps:
                    inV.append(j)
                    inE.append(k)
                else:
                    outV.append(j)
                    outE.append(k)
                    pOuts.append(G.vs[j]['pressure'])
            if len(inE)== 0:
                for vertex,edge,pValue in zip(outV,outE,pOuts):
                    if pValue==np.max(pOuts):
                        inV.append(vertex)
                        inE.append(edge)
                        break
                outE.remove(edge)
                outV.remove(vertex)
            if len(inE) == 1 and len(outE) == 4:
                case.append(1)
            elif len(inE) == 2 and len(outE) == 3:
                case.append(2)
            elif len(inE) == 3 and len(outE) == 2:
                case.append(3)
            elif len(inE) == 4 and len(outE) == 1:
                case.append(4)
            else:
                print('WARNING')
                print(i)
                print(inE)
            # CASE 1: 1 inflow 4 outflow
            # -   -         -   - o -
            #   o -     TO    o     -
            #     -             - o -
            #     -                 -
            if case[-1] == 1:
                G.add_vertices(2)
                v1=G.vcount()-1
                v2=G.vcount()-2
                for attr in G.vs.attribute_names():
                    G.vs[v1][attr]=G.vs[i][attr]
                    G.vs[v2][attr]=G.vs[i][attr]
                eAndV=[]
                for j,k in zip(outV,outE):
                    eAndV.append([j,k])
                G.add_edges([(v1,i),(v1,eAndV[0][0]),(v1,eAndV[1][0]),(v2,i),(v2,eAndV[2][0]),(v2,eAndV[3][0])])
                e11=G.ecount()-1
                e12=G.ecount()-2
                e10=G.ecount()-3
                e21=G.ecount()-4
                e22=G.ecount()-5
                e20=G.ecount()-6
                G.es[e10]['artificialReduceDegreeEdge']=1
                G.es[e11]['artificialReduceDegreeEdge']=1
                G.es[e12]['artificialReduceDegreeEdge']=1
                G.es[e20]['artificialReduceDegreeEdge']=1
                G.es[e21]['artificialReduceDegreeEdge']=1
                G.es[e22]['artificialReduceDegreeEdge']=1
                G.es[e10]['length']=G.es[eAndV[0][1]]['length']
                G.es[e10]['diameter']=(G.es[eAndV[0][1]]['diameter'] + G.es[eAndV[1][1]]['diameter'])/2.
                G.es[e10]['medianLabelAV']=G.es[eAndV[0][1]]['medianLabelAV']
                G.es[e10]['medianLabelSurfPlun']=G.es[eAndV[0][1]]['medianLabelSurfPlun']
                G.es[e20]['length']=G.es[eAndV[0][1]]['length']
                G.es[e20]['diameter']=(G.es[eAndV[0][1]]['diameter'] + G.es[eAndV[1][1]]['diameter'])/2.
                G.es[e20]['medianLabelAV']=G.es[eAndV[0][1]]['medianLabelAV']
                G.es[e20]['medianLabelSurfPlun']=G.es[eAndV[0][1]]['medianLabelSurfPlun']
                for attr in G.es.attribute_names():
                    G.es[e11][attr]=G.es[eAndV[3][1]][attr]
                for attr in G.es.attribute_names():
                    G.es[e12][attr]=G.es[eAndV[2][1]][attr]
                for attr in G.es.attribute_names():
                    G.es[e21][attr]=G.es[eAndV[1][1]][attr]
                for attr in G.es.attribute_names():
                    G.es[e22][attr]=G.es[eAndV[0][1]][attr]
                G.delete_edges([eAndV[0][1],eAndV[1][1],eAndV[2][1],eAndV[3][1]])
                G.vs['degree']=G.degree()
            # CASE 2: 2 inflows 3 outflows
            # -   -         -       -
            # - o -     TO  - o - o -
            #     -                 -
            elif case[-1] == 2:
                G.add_vertices(1)
                v1=G.vcount()-1
                for attr in G.vs.attribute_names():
                    G.vs[v1][attr]=G.vs[i][attr]
                eAndV=[]
                for j,k in zip(outV,outE):
                    eAndV.append([j,k])
                G.add_edges([(v1,i),(v1,eAndV[0][0]),(v1,eAndV[1][0]),(v1,eAndV[2][0])])
                e11=G.ecount()-1
                e12=G.ecount()-2
                e13=G.ecount()-3
                e10=G.ecount()-4
                G.es[e10]['artificialReduceDegreeEdge']=1
                G.es[e11]['artificialReduceDegreeEdge']=1
                G.es[e12]['artificialReduceDegreeEdge']=1
                G.es[e13]['artificialReduceDegreeEdge']=1
                G.es[e10]['length']=G.es[eAndV[0][1]]['length']
                G.es[e10]['diameter']=(G.es[eAndV[0][1]]['diameter'] + G.es[eAndV[1][1]]['diameter'])/2.
                G.es[e10]['medianLabelAV']=G.es[eAndV[0][1]]['medianLabelAV']
                G.es[e10]['medianLabelSurfPlun']=G.es[eAndV[0][1]]['medianLabelSurfPlun']
                for attr in G.es.attribute_names():
                    G.es[e11][attr]=G.es[eAndV[2][1]][attr]
                for attr in G.es.attribute_names():
                    G.es[e12][attr]=G.es[eAndV[1][1]][attr]
                for attr in G.es.attribute_names():
                    G.es[e13][attr]=G.es[eAndV[0][1]][attr]
                G.delete_edges([eAndV[0][1],eAndV[1][1],eAndV[2][1]])
                G.vs['degree']=G.degree()
            # CASE 3: 3 inflows 2 outflows
            # -   -         -       -
            # - o -     TO  - o - o -
            # -             -       
            elif case[-1] == 3:
                G.add_vertices(1)
                v1=G.vcount()-1
                for attr in G.vs.attribute_names():
                    G.vs[v1][attr]=G.vs[i][attr]
                eAndV=[]
                for j,k in zip(outV,outE):
                    eAndV.append([j,k])
                G.add_edges([(v1,i),(v1,eAndV[0][0]),(v1,eAndV[1][0])])
                e11=G.ecount()-1
                e12=G.ecount()-2
                e10=G.ecount()-3
                G.es[e10]['artificialReduceDegreeEdge']=1
                G.es[e11]['artificialReduceDegreeEdge']=1
                G.es[e12]['artificialReduceDegreeEdge']=1
                G.es[e10]['length']=G.es[eAndV[0][1]]['length']
                G.es[e10]['diameter']=(G.es[eAndV[0][1]]['diameter'] + G.es[eAndV[1][1]]['diameter'])/2.
                G.es[e10]['medianLabelAV']=G.es[eAndV[0][1]]['medianLabelAV']
                G.es[e10]['medianLabelSurfPlun']=G.es[eAndV[0][1]]['medianLabelSurfPlun']
                for attr in G.es.attribute_names():
                    G.es[e11][attr]=G.es[eAndV[1][1]][attr]
                for attr in G.es.attribute_names():
                    G.es[e12][attr]=G.es[eAndV[0][1]][attr]
                G.delete_edges([eAndV[0][1],eAndV[1][1]])
                G.vs['degree']=G.degree()
            # CASE 4: 4 inflows 1 outflow
            # -   -         - o  - 
            # - o       TO  -      o -
            # -             - o  -  
            # -             -       
            elif case[-1] == 4:
                G.add_vertices(2)
                v1=G.vcount()-1
                v2=G.vcount()-2
                for attr in G.vs.attribute_names():
                    G.vs[v1][attr]=G.vs[i][attr]
                    G.vs[v2][attr]=G.vs[i][attr]
                eAndV=[]
                for j,k in zip(inV,inE):
                    eAndV.append([j,k])
                G.add_edges([(v1,i),(v1,eAndV[0][0]),(v1,eAndV[1][0]),(v2,i),(v2,eAndV[2][0]),(v2,eAndV[3][0])])
                e11=G.ecount()-1
                e12=G.ecount()-2
                e10=G.ecount()-3
                e21=G.ecount()-4
                e22=G.ecount()-5
                e20=G.ecount()-6
                G.es[e10]['artificialReduceDegreeEdge']=1
                G.es[e11]['artificialReduceDegreeEdge']=1
                G.es[e12]['artificialReduceDegreeEdge']=1
                G.es[e20]['artificialReduceDegreeEdge']=1
                G.es[e21]['artificialReduceDegreeEdge']=1
                G.es[e22]['artificialReduceDegreeEdge']=1
                G.es[e10]['length']=G.es[eAndV[0][1]]['length']
                G.es[e10]['diameter']=(G.es[eAndV[0][1]]['diameter'] + G.es[eAndV[1][1]]['diameter'])/2.
                G.es[e10]['medianLabelAV']=G.es[eAndV[0][1]]['medianLabelAV']
                G.es[e10]['medianLabelSurfPlun']=G.es[eAndV[0][1]]['medianLabelSurfPlun']
                G.es[e20]['length']=G.es[eAndV[0][1]]['length']
                G.es[e20]['diameter']=(G.es[eAndV[0][1]]['diameter'] + G.es[eAndV[1][1]]['diameter'])/2.
                G.es[e20]['medianLabelAV']=G.es[eAndV[0][1]]['medianLabelAV']
                G.es[e20]['medianLabelSurfPlun']=G.es[eAndV[0][1]]['medianLabelSurfPlun']
                for attr in G.es.attribute_names():
                    G.es[e11][attr]=G.es[eAndV[3][1]][attr]
                for attr in G.es.attribute_names():
                    G.es[e12][attr]=G.es[eAndV[2][1]][attr]
                for attr in G.es.attribute_names():
                    G.es[e21][attr]=G.es[eAndV[1][1]][attr]
                for attr in G.es.attribute_names():
                    G.es[e22][attr]=G.es[eAndV[0][1]][attr]
                G.delete_edges([eAndV[0][1],eAndV[1][1],eAndV[2][1],eAndV[3][1]])
                G.vs['degree']=G.degree()

        return G
#------------------------------------------------------------------------------
def deleteUnlabeled_degree1_Vertices(G):
        """ 
        Deletes all vertices which are not labeled as A or V an which are degree 1
        INPUT: G: Vascular graph in iGraph format.
        OUTPUT:
        """
        av=G.vs(degree_eq=1,labelAV_eq=3).indices
        G.vs[av]['av']=[1]*len(av)
        vv=G.vs(degree_eq=1,labelAV_eq=4).indices
        G.vs[vv]['vv']=[1]*len(vv)
        
        deg1=G.vs(degree_eq=1).indices
        notLabeledDeg1=[]
        for i in deg1:
            if i not in av and i not in vv:
                notLabeledDeg1.append(i)
        
        print('Not labeled deg1')
        print(len(notLabeledDeg1))
        
        while notLabeledDeg1 != []:
            print(len(notLabeledDeg1))
            G.delete_vertices(notLabeledDeg1)
            G.vs['degree']=G.degree()
            notLabeledDeg1=[]
            av=G.vs(degree_eq=1,labelAV_eq=3).indices
            vv=G.vs(degree_eq=1,labelAV_eq=4).indices
            deg1=G.vs(degree_eq=1).indices
            for i in deg1:
                if i not in av and i not in vv:
                    notLabeledDeg1.append(i)

        return G

#------------------------------------------------------------------------------
def introduce_nkind_labels(G):
        """ 
        introduces the nkind label based on the medianLabelAV and the medianLabelSurfPlun
        INPUT: G: Vascular graph in iGraph format.
        OUTPUT:
        """

        #Assign kind and nkind
        G.es['nkind']=[4]*G.ecount()
        arteries=G.es(labelAV_eq=3).indices
        G.es[arteries]['nkind']=[2]*len(arteries)
        venules=G.es(labelAV_eq=4).indices
        G.es[venules]['nkind']=[3]*len(venules)
        pialArteries=G.es(labelAV_eq=3,labelSurfPlun_eq=3).indices
        G.es[pialArteries]['nkind']=[0]*len(pialArteries)
        pialVenules=G.es(labelAV_eq=4,labelSurfPlun_eq=3).indices
        G.es[pialVenules]['nkind']=[1]*len(pialVenules)
        
        #nkind = {'pa':0, 'pv':1, 'a':2, 'v':3, 'c':4, 'n':5}
        caps=G.es(nkind_eq=4)
        for e in caps:
            G.vs[e.tuple]['nkind'] = [4,4]
        
        pArts=G.es(nkind_eq=0)
        for e in pArts:
            G.vs[e.tuple]['nkind'] = [0,0]
        
        pveins=G.es(nkind_eq=1)
        for e in pveins:
            G.vs[e.tuple]['nkind'] = [1,1]
        
        veins=G.es(nkind_eq=3)
        for e in veins:
            G.vs[e.tuple]['nkind'] = [3,3]

        arts=G.es(nkind_eq=2)
        for e in arts:
            G.vs[e.tuple]['nkind'] = [2,2]

        return G 
#------------------------------------------------------------------------------
def eliminate_loop_vertices(G):
        """ 
        Loop vertices are vertices, which contain a edge which goes from vertex i
        to vertex i. Those Edges as well as the dead ends which result are deleted
        INPUT: G: Vascular graph in iGraph format.
        OUTPUT:
        """

        #check for loop vertices
        probLoops=[]
        for i in range(G.vcount()):
            if i in G.neighbors(i):
                probLoops.append(i)
        
        noDeg1ToPreserve=len(G.vs(degree_eq=1))
        print('number of deg1 vertices')
        print(noDeg1ToPreserve)
        print('Loop Vertices')
        print(len(probLoops))
        G.vs['deg1']=[0]*G.vcount()
        deg1=G.vs(degree_eq=1).indices
        G.vs[deg1]['deg1']=[1]*len(deg1)
        for i in probLoops:
            deleteEdges=[]
            for j,k in zip(G.neighbors(i),G.incident(i)):
                if i == j:
                    deleteEdges.append(k)
            G.delete_edges(deleteEdges)
        
        G.vs['degree']=G.degree()
        
        while len(G.vs(degree_eq=1)) != noDeg1ToPreserve:
            print(len(G.vs(degree_eq=1)))
            G.delete_vertices(list(set(G.vs(degree_eq=1).indices)-set(G.vs(deg1_eq=1).indices)))
            G.vs['degree']=G.degree()
        
        probLoops=[]
        for i in range(G.vcount()):
            if i in G.neighbors(i):
                probLoops.append(i)
        
        if len(probLoops) > 0:
            print('ERROR there are still some loop vertices')

        return G
#------------------------------------------------------------------------------
def artificially_increase_length_of_artificialReduceDegreeEdges_based_on_dummy_simulations(G):
        """ 
        The length of the artificially introduced edges is increase such 
        that the transit time of venules is at least 3ms and of all othter vessels
        1.5 ms.
        INPUT: G: Vascular graph in iGraph format.
        OUTPUT:
        """

        P=vgm.Physiology()
        vrbc=P.rbc_volume('mouse')
        #Dummy simulation --> the goal is to artificially increase the length of vessels such that the cfl criterion is fullfield --> avoiding traffic jams in the vessels
        G.vs['av']=[0]*G.vcount()
        G.vs['vv']=[0]*G.vcount()
        G.vs[G.vs(degree_eq=1,nkind_eq=0).indices]['av']=[1]*len(G.vs(degree_eq=1,nkind_eq=0).indices)
        G.vs[G.vs(degree_eq=1,nkind_eq=2).indices]['av']=[1]*len(G.vs(degree_eq=1,nkind_eq=2).indices)
        G.vs[G.vs(degree_eq=1,nkind_eq=1).indices]['vv']=[1]*len(G.vs(degree_eq=1,nkind_eq=1).indices)
        G.vs[G.vs(degree_eq=1,nkind_eq=3).indices]['vv']=[1]*len(G.vs(degree_eq=1,nkind_eq=3).indices)
        G.vs[G.vs(av_eq=1).indices]['pBC']=[60.0]*len(G.vs(av_eq=1).indices)
        G.vs[G.vs(vv_eq=1).indices]['pBC']=[0.0]*len(G.vs(vv_eq=1).indices)
        del(G.es['htt'])
        LS=vgm.LinearSystem(G,withRBC=0.2,invivo=0)
        LS.solve('iterative2')

        
        for e in G.es[G.es(artificialReduceDegreeEdge_eq=1).indices]:
            if e['length'] < 3*vrbc/(0.25*np.pi*e['diameter']**2):
                e['length'] = 3*vrbc/(0.25*np.pi*e['diameter']**2)
        
        edgesProb=[0] #dummy
        while len(edgesProb) != 0:
            edgesProb=[]
            for e in G.es[G.es(artificialReduceDegreeEdge_eq=1).indices]:
                if e['v'] != 0:
                    if e['nkind'] == 3:
                        if e['length']/e['v'] < 3:
                            edgesProb.append(e.index)
                            e['length'] = e['length']*1.1
                    else:
                        if e['length']/e['v'] < 1.5:
                            edgesProb.append(e.index)
                            e['length'] = e['length']*1.1
            del(G.es['htt'])
            LS=vgm.LinearSystem(G,withRBC=0.2,invivo=0)
            LS.solve('iterative2')
        
        G.add_points(1.,G.es(artificialReduceDegreeEdge_eq=1).indices)
        return G
#------------------------------------------------------------------------------
def create_graphINFO(G):
        """ 
        Creates the outputsummary in GraphINFO/PreprocessingGraph_and_GraphStats.txt.
        Additionally, One histogram of capillary diameters and lengths is created
        INPUT: G: Vascular graph in iGraph format.
        OUTPUT:
        """

        diamLt3=G.es(diameter_lt=3).indices
        
        vesselVol=np.array(G.es['diameter'])**2*0.25*np.pi*np.array(G.es['length'])
        G.es['volume']=vesselVol
        vrbcMouse=49.0
        volTooSmall=G.es(volume_lt=vrbcMouse).indices
        
        dirname='GraphINFO'
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        
        os.chdir(dirname)
        
        new_file=open('PreprocessingGraph_and_GraphStats.txt','w')
        new_file.write('Total number of Edges: \n')
        new_file.write(str(G.ecount())+'\n')
        new_file.write('  \n')
        new_file.write('Total number of Vertices: \n')
        new_file.write(str(G.vcount())+'\n')
        new_file.write('  \n')
        new_file.write('Relative number of capillaries: \n')
        new_file.write(str(100*float(len(G.es(nkind_eq=4)))/G.ecount())+'\n')
        new_file.write('  \n')
        new_file.write('Average diameter of capillaries: \n')
        new_file.write(str(np.mean(G.es[G.es(nkind_eq=4).indices]['diameter']))+'\n')
        new_file.write('  \n')
        new_file.write('Relative number of capillaries with a diameter < 3 mum: \n')
        new_file.write(str(100*float(len(G.es(nkind_eq=4,diameter_lt=3.)))/len(G.es(nkind_eq=4)))+'\n')
        new_file.write('  \n')
        new_file.write('Relative number of vessels with volume < vrbc: \n')
        new_file.write(str(100*float(len(volTooSmall))/G.ecount())+'\n')
        new_file.write('  \n')
        new_file.close()
        #Histogramms
        #Diameters capillaries
        P=vgm.Physiology()
        vrbc=P.rbc_volume('mouse')
        caps=G.es(nkind_eq=4).indices
        diameters=[]
        lengths=[]
        for i in caps:
            diameters.append(G.es[i]['diameter'])
            lengths.append(G.es[i]['length'])
        
        diameters=np.array(diameters)
        fig1=plt.figure(figsize=(6,6))
        ax1=fig1.add_subplot(111)
        bins=[1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.,5.5,6.,6.5,7.,7.5,8.,8.5,9.,9.5,10.]
        heights,bins,patches=ax1.hist(diameters,weights=np.zeros_like(diameters)+1./diameters.size,bins=bins,color=[0.,0.6,0.6])
        ax1.xaxis.set_ticks([1,2,3.0,4.0,5.,6.,7.,8.,9.,10.])
        ax1.yaxis.set_ticks([0.,0.05,0.1,0.15,0.2,0.25])
        ax1.xaxis.set_ticklabels(('1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0'),fontsize=16)
        ax1.yaxis.set_ticklabels(('0','5','10','15','20','25'),fontsize=16)
        plt.xlim([0.5,10.5])
        plt.ylim([0.,0.27])
        plt.ylabel('Frequency [$\%$]',fontsize=16)
        plt.xlabel('Diameter [$\mu m$]',fontsize=16)
        plt.text(7.0,0.23,'mean: %.2f' %np.mean(diameters),fontsize=14)
        plt.text(7.0,0.22,'median: %.2f' %np.median(diameters),fontsize=14)
        plt.text(7.0,0.21,'std: %.2f' %np.std(diameters),fontsize=14)
        plt.text(7.0,0.20,'min: %.2f' %np.min(diameters),fontsize=14)
        plt.text(7.0,0.19,'max: %.2f' %np.max(diameters),fontsize=14)
        fig1.savefig('diameter_histogram_caps.eps',format='eps',dpi=200)
        
        #Lengths capillaries
        lengths=np.array(lengths)
        fig2=plt.figure(figsize=(6,6))
        ax2=fig2.add_subplot(111)
        heights2,bins2,patches2=ax2.hist(lengths,weights=np.zeros_like(lengths)+1./lengths.size,bins=[0,25,50,75,100,125,150,175,200],color=[0.,0.6,0.6])
        ax2.xaxis.set_ticks([0.0,25.0,50.,75.,100.,125.,150,175,200])
        ax2.yaxis.set_ticks([0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35])
        ax2.xaxis.set_ticklabels(('0','25','50','75','100','125','150','175','200'),fontsize=16)
        ax2.yaxis.set_ticklabels(('0','5','10','15','20','25','30','35'),fontsize=16)
        plt.xlim([-5,205])
        plt.ylabel('Frequency [$\%$]',fontsize=16)
        plt.xlabel('Length [$\mu m$]',fontsize=16)
        plt.text(135.0,0.33,'mean: %.2f' %np.mean(lengths),fontsize=14)
        plt.text(135.0,0.31,'median: %.2f' %np.median(lengths),fontsize=14)
        plt.text(135.0,0.29,'std: %.2f' %np.std(lengths),fontsize=14)
        plt.text(135.0,0.27,'min: %.2f' %np.min(lengths),fontsize=14)
        plt.text(135.0,0.25,'max: %.2f' %np.max(lengths),fontsize=14)
        fig2.savefig('Length_histogram_caps.eps',format='eps',dpi=200)
        
        os.chdir('../')
#------------------------------------------------------------------------------
def improve_capillary_diameters_by_binFitting(G,mu,std,lowerBound,upperBound):
        """ Histogram base upscaling approach. 
        Furhtermore, the new diameter distributions are plotted in GraphINFO 
        INPUT: G: Vascular graph in iGraph format.
        mu: mean of the beta distribution
        std: std of the beta distribution
        a: lower bound of the beta distribution
        c: upper bound of the beta distribution
        OUTPUT:
        """

        matplotlib.rc('text', usetex = True)
        matplotlib.rc('font', **{'family':"sans-serif"})
        params = {'text.latex.preamble': [r'\usepackage{siunitx}',
                r'\usepackage{sfmath}', r'\sisetup{detect-family = true}',
                    r'\usepackage{amsmath}']}
        plt.rcParams.update(params)

        G.es['diameterNew']=deepcopy(G.es['diameter'])
        eps = finfo(float).eps * 1e5
        #The diameters are fitted based on a beta distribution. The lower limit for capillary diameters = 2.5, the higher limit = 9.0
        #The goal for the mean value is 4.0 and for the std 1.0
        
        a=0 #equals = 2.5
        c=1 #equals = 9.0
        aReal = lowerBound
        cReal = upperBound
        
        ##Compute Alpha and beta have been computed with matlab see script betaDistribution.m
        muStar_func = lambda mu,std,a,c: (mu-a)/(c-a)
        stdStar_func = lambda std,a,c: std/(c-a)
        
        muStar = muStar_func(mu,std,aReal,cReal)
        stdStar = stdStar_func(std,aReal,cReal)
        
        alpha_func = lambda mu,std: ((1-mu)/(std**2)-(1/mu))*(mu**2)
        beta_func = lambda alpha,mu: alpha*((1/mu)-1)
        
        alpha = alpha_func(muStar,stdStar)
        beta = beta_func(alpha,muStar)

        #old alpha and betas, however they are slightly off --> mu=4.2, std=0.95
        #alpha = 2.0735
        #beta=5.8546
        
        #Define beta general betadistribution [a,c]
        betaFuncDummy = lambda u,alpha,beta: u**(alpha-1)*(1-u)**(beta-1)
        betaFunc = lambda alpha,beta: quad(betaFuncDummy,0,1,args=(alpha,beta))[0]
        betaFuncPDF = lambda x,alpha,beta: 1./betaFunc(alpha,beta)*x**(alpha-1)*(1-x)**(beta-1)
        betaFuncPDF_int = lambda alpha,beta,lim0,lim1: quad(betaFuncPDF,lim0,lim1,args=(alpha,beta))[0]
        
        plotBetaXDummy=np.linspace(0,1,501)
        plotBetaX=np.linspace(2.5,9,501)
        plotBetaX2=[]
        plotBetaY=[]
        for i in range(len(plotBetaXDummy)-1):
            plotBetaY.append(betaFuncPDF_int(alpha,beta,plotBetaXDummy[i],plotBetaXDummy[i+1]))
            plotBetaX2.append(np.mean([plotBetaX[i],plotBetaX[i+1]]))
        
        nBins=500
        width=1./nBins
        widthD=(cReal-aReal)/nBins
        eNo=len(G.es(diameter_le=cReal))
        eNoCap=len(G.es(diameter_le=cReal,nkind_eq=4))
        caps=G.es(diameter_le=cReal).indices
        capsEdges=G.es(diameter_le=cReal,nkind_eq=4).indices
        
        bins=[]
        bins.append(0)
        sumnGoal=0
        dmin=np.min(G.es[caps]['diameterNew'])
        binsD=[]
        binsD.append(aReal)
        plotBetaYtest=[]
        edgesSave=[]
        for i in range(nBins):
            limL=i*width
            limH=(i+1)*width
            bins.append(limH)
            limLD=i*widthD+aReal
            limHD=(i+1)*widthD+aReal
            binsD.append(limHD)
            if limHD >= dmin and limLD < dmin:
                edges=G.es(diameterNew_ge=limLD,diameterNew_lt=dmin).indices
                if len(edges) > 0:
                    factor=dmin-np.min(G.es[edges]['diameterNew'])
                    G.es[edges]['diameterNew'] = np.array(G.es[edges]['diameterNew']) + factor
                nGoal=int(np.ceil(betaFuncPDF_int(alpha,beta,limL,limH)*eNo))
                plotBetaYtest.append(nGoal/np.float(eNoCap))
                sumnGoal += nGoal
                edges=G.es(diameterNew_ge=limLD,diameterNew_lt=limHD).indices
                if len(edges) > nGoal:
                    shiftENo = len(edges)-nGoal
                    sortedE=zip(G.es[edges]['diameterNew'],G.es[edges]['diameter'],edges)
                    sortedE.sort()
                    dLim=sortedE[-1*shiftENo][0]
                    edges2=G.es(diameterNew_ge=dLim).indices
                    factor = limHD-dLim
                    G.es[edges2]['diameterNew']=np.array(G.es[edges2]['diameterNew'])+factor
            else:
                if i == 0:
                    edges=G.es(diameterNew_lt=limLD).indices
                    if len(edges) > 0:
                        factor=limLD-np.min(G.es[edges]['diameterNew'])
                        G.es[edges]['diameterNew'] = np.array(G.es[edges]['diameterNew']) + factor
                nGoal=int(np.ceil(betaFuncPDF_int(alpha,beta,limL,limH)*eNo))
                plotBetaYtest.append(nGoal/np.float(eNoCap))
                sumnGoal += nGoal
                edges=G.es(diameterNew_ge=limLD,diameterNew_lt=limHD).indices
                if len(edges) > nGoal:
                    shiftENo = len(edges)-nGoal #Number of edges that has to be transfered to a large bin
                    sortedE=zip(G.es[edges]['diameterNew'],G.es[edges]['diameter'],edges)
                    sortedE.sort()
                    dLim=sortedE[-1*shiftENo][0] #if diameter > dLim --> transfered to larger bin
                    if dLim == sortedE[-1*(shiftENo+1)][0]:
                        for j in range(shiftENo):
                            if np.abs(sortedE[-1*(j+1)][0]-dLim) < eps:
                                G.es[sortedE[-1*(j+1)][2]]['diameterNew'] = G.es[sortedE[-1*(j+1)][2]]['diameterNew'] + eps
                        dLim = G.es[sortedE[-1*(j+1)][2]]['diameterNew']
                    factor = limHD-dLim
                    edges2=G.es(diameterNew_ge=dLim).indices
                    G.es[edges2]['diameterNew']=np.array(G.es[edges2]['diameterNew'])+factor
                edges=G.es(diameterNew_ge=limLD,diameterNew_lt=limHD).indices
                edgesSave.append(len(edges))
        
        diameters=np.array(G.es[G.es(nkind_eq=4,diameterNew_lt=cReal).indices]['diameterNew'])
        diametersStart=np.array(G.es[G.es(nkind_eq=4,diameterNew_lt=cReal).indices]['diameterNew'])
        fig1=plt.figure(figsize=(6,6))
        ax1=fig1.add_subplot(111)
        heights,bins,patches=ax1.hist(diameters,weights=np.zeros_like(diameters)+1./diameters.size,bins=binsD,color=[0.,0.6,0.6])
        plotBetaY2=np.array(plotBetaY)/(np.max(plotBetaY)/np.max(heights))
        plt.plot(plotBetaX2,plotBetaY2,'r',linewidth=2)
        ax1.xaxis.set_ticks([0,1,2,3.0,4.0,5.,6.,7.,8.,9.,10.])
        ax1.xaxis.set_ticklabels(('0','1','2','3','4','5','6','7','8','9','10'),fontsize=16)
        ax1.yaxis.set_ticks([0.,0.005,0.01,0.015,0.02])
        ax1.yaxis.set_ticklabels(('0.0','0.5','1.0','1.5','2.0'),fontsize=16)
        plt.xlim([0,10])
        plt.ylabel('Frequency [$\%$]',fontsize=16)
        plt.xlabel('Diameter [$\mu m$]',fontsize=16)
        plt.text(0.725,0.95,'mean: %.2f' %np.mean(diameters),fontsize=14,transform=ax1.transAxes)
        plt.text(0.725,0.90,'median: %.2f' %np.median(diameters),fontsize=14,transform=ax1.transAxes)
        plt.text(0.725,0.85,'std: %.2f' %np.std(diameters),fontsize=14,transform=ax1.transAxes)
        plt.text(0.725,0.80,'min: %.2f' %np.min(diameters),fontsize=14,transform=ax1.transAxes)
        plt.text(0.725,0.75,'max: %.2f' %np.max(diameters),fontsize=14,transform=ax1.transAxes)
        plt.subplots_adjust(left=0.15,right=0.95,bottom=0.125,top=0.95,wspace=0.0,hspace=0.45)
        #fig1.savefig('GraphINFO/diameter_histogram_capsBinFittingAfter.eps',format='eps',dpi=200)
        
        diameters=np.array(G.es[G.es(nkind_eq=4,diameterNew_lt=cReal).indices]['diameter'])
        fig1=plt.figure(figsize=(6,6))
        ax1=fig1.add_subplot(111)
        heights,bins,patches=ax1.hist(diameters,weights=np.zeros_like(diameters)+1./diameters.size,bins=binsD,color=[0.,0.6,0.6])
        plt.plot(plotBetaX2,plotBetaY2,'r',linewidth=2)
        ax1.xaxis.set_ticks([0,1,2,3.0,4.0,5.,6.,7.,8.,9.,10.])
        ax1.xaxis.set_ticklabels(('0','1','2','3','4','5','6','7','8','9','10'),fontsize=16)
        ax1.yaxis.set_ticks([0.,0.005,0.01,0.015,0.02])
        ax1.yaxis.set_ticklabels(('0.0','0.5','1.0','1.5','2.0'),fontsize=16)
        plt.xlim([0,10])
        plt.ylabel('Frequency [$\%$]',fontsize=16)
        plt.xlabel('Diameter [$\mu m$]',fontsize=16)
        plt.text(0.725,0.95,'mean: %.2f' %np.mean(diameters),fontsize=14,transform=ax1.transAxes)
        plt.text(0.725,0.90,'median: %.2f' %np.median(diameters),fontsize=14,transform=ax1.transAxes)
        plt.text(0.725,0.85,'std: %.2f' %np.std(diameters),fontsize=14,transform=ax1.transAxes)
        plt.text(0.725,0.80,'min: %.2f' %np.min(diameters),fontsize=14,transform=ax1.transAxes)
        plt.text(0.725,0.75,'max: %.2f' %np.max(diameters),fontsize=14,transform=ax1.transAxes)
        plt.subplots_adjust(left=0.15,right=0.95,bottom=0.125,top=0.95,wspace=0.0,hspace=0.45)
        #fig1.savefig('GraphINFO/diameter_histogram_capsBinFittingBefore.eps',format='eps',dpi=200)
        
        diameters=np.array(G.es[G.es(nkind_eq=4,diameterNew_lt=cReal).indices]['diameterNew'])
        diameters2=np.array(G.es[G.es(nkind_eq=4,diameterNew_lt=cReal).indices]['diameter'])
        print('length')
        print(len(diameters2))
        fig1=plt.figure(figsize=(7.5,7.5))
        ax1=fig1.add_subplot(111)
        heights,bins,patches=ax1.hist(diameters2,weights=np.zeros_like(diameters2)+1./diameters2.size,bins=binsD,facecolor=[0.5,0.5,0.5],edgecolor=[0.5,0.5,0.5])
        heights,bins,patches=ax1.hist(diameters,weights=np.zeros_like(diameters)+1./diameters.size,bins=binsD,facecolor=[0,0,0],edgecolor=[0,0,0])
        plt.plot(plotBetaX2,plotBetaY2,'r',linewidth=2)
        ax1.xaxis.set_ticks([2,4,6.,8.,10.])
        ax1.xaxis.set_ticklabels(('2','4','6','8','10'),fontsize=30)
        ax1.yaxis.set_ticks([0.,0.01,0.02,0.03,0.04,0.05])
        ax1.yaxis.set_ticklabels(('0.0','1.0','2.0','3.0','4.0','5.0'),fontsize=30)
        plt.xlim([2,10])
        plt.ylim([0,0.05])
        plt.ylabel('Frequency [$\%$]',fontsize=30)
        plt.xlabel('Diameter [\si{\um}]',fontsize=30)
        plt.text(0.495,0.92,'Original' ,fontsize=30,transform=ax1.transAxes,fontweight='bold',color=[0.5,0.5,0.5])
        plt.text(0.495,0.85,'mean: %.2f' %np.mean(diameters2),fontsize=30,transform=ax1.transAxes,color=[0.5,0.5,0.5])
        plt.text(0.495,0.78,'median: %.2f' %np.median(diameters2),fontsize=30,transform=ax1.transAxes,color=[0.5,0.5,0.5])
        plt.text(0.495,0.71,'std: %.2f' %np.std(diameters2),fontsize=30,transform=ax1.transAxes,color=[0.5,0.5,0.5])
        plt.text(0.495,0.64,'min: %.2f' %np.min(diameters2),fontsize=30,transform=ax1.transAxes,color=[0.5,0.5,0.5])
        plt.text(0.495,0.57,'max: %.2f' %np.max(diameters2),fontsize=30,transform=ax1.transAxes,color=[0.5,0.5,0.5])
        plt.text(0.495,0.47,'After upscaling' ,fontsize=30,transform=ax1.transAxes,fontweight='bold')
        plt.text(0.495,0.40,'mean: %.2f' %np.mean(diameters),fontsize=30,transform=ax1.transAxes)
        plt.text(0.495,0.33,'median: %.2f' %np.median(diameters),fontsize=30,transform=ax1.transAxes)
        plt.text(0.495,0.26,'std: %.2f' %np.std(diameters),fontsize=30,transform=ax1.transAxes)
        plt.text(0.495,0.19,'min: %.2f' %np.min(diameters),fontsize=30,transform=ax1.transAxes)
        plt.text(0.495,0.12,'max: %.2f' %np.max(diameters),fontsize=30,transform=ax1.transAxes)
        ax1.set_position([0.18,0.16,0.775,0.775])
        ax1.tick_params(axis='both', which='major', pad=10)
        #fig1.savefig('GraphINFO/diameter_histogram_capsBinFittingCombinded.eps',format='eps',dpi=300)
        fig1.savefig('GraphINFO/diameter_histogram_capsBinFittingCombinded_'+str(mu)+'.tiff',format='tiff',dpi=300)

        #G.es['diameter']=deepcopy(G.es['diameterNew'])
        #del(G.es['diameterNew'])
        return G

#------------------------------------------------------------------------------
def improve_capillary_labeling(G,bReal=9):
        """ Histogram base upscaling approach. 
        Furhtermore, the new diameter distributions are plotted in GraphINFO 
        INPUT: G: Vascular graph in iGraph format.
        OUTPUT: 
        """
        #adjust edge labeling
        #There might be some edges which are labeled as capillaries, even if their diameter > 9.
        #These are assigned as capillaires because it was not possible to either classify them as arterioles or venules
        #Many of them enter the domain from the side
        changeLabel=G.es(nkind_eq=4,diameter_gt=bReal).indices
        changeLabel2=[]
        print('Improve Labeling')
        while len(changeLabel) > 0:
            print(len(changeLabel))
            for e in G.es[changeLabel]:
                boolLabel=0
                for j in e.tuple:
                    if G.vs[j]['nkind'] == 3 or G.vs[j]['nkind'] == 1:
                        e['nkind']=3
                        boolLabel = 1
                    elif G.vs[j]['nkind'] == 2 or G.vs[j]['nkind'] == 0:
                        e['nkind']=2
                        boolLabel = 1
                if boolLabel:
                    for j in e.tuple:
                        G.vs[j]['nkind']=e['nkind']
                else:
                    changeLabel2.append(e.index)
            if changeLabel == changeLabel2:
                print('Warning No more labeling improvement possible')
                break
            changeLabel = deepcopy(changeLabel2)
            changeLabel2=[]

        return G
        
#------------------------------------------------------------------------------
def create_histograms_for_different_vesselTypes(G,nbins=21,boolAfter=0):
        """ Creates histogram of vessel diameters for the different vessel types
        into the folder 'Diameter vessel types'
        INPUT: G: Vascular graph in iGraph format.
            nbins: number of bins for the histogram,
            boolAfter: boolean if the string 'after' should be added to the filename
        OUTPUT: diameter histograms in the folder DimaterVesselTypes
        """

        dirname='DiameterVesselTypes'
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        
        os.chdir(dirname)

        for i in np.unique(G.es['nkind']):
            diameters=np.array(G.es[G.es(nkind_eq=i).indices]['diameter'])
            fig1=plt.figure(figsize=(6,6))
            ax1=fig1.add_subplot(111)
            bins=np.linspace(np.min(diameters),np.max(diameters),nbins)
            heights,bins,patches=ax1.hist(diameters,weights=np.zeros_like(diameters)+1./diameters.size,bins=bins,color=[0.,0.6,0.6])
            plt.ylabel('Frequency [$\%$]',fontsize=16)
            plt.xlabel('Diameter [$\mu m$]',fontsize=16)
            for k,j in enumerate(heights):
                if j == np.max(heights):
                    break
            if k > nbins/2.:
                #Extra information should be close to the right border of the figure
                posX=np.min(diameters)+0.1*(np.max(diameters)-np.min(diameters))
            else:
                #Extra information should be close to the left border of the figure
                posX=np.max(diameters)-0.30*(np.max(diameters)-np.min(diameters))
            plt.text(posX,0.96*np.max(heights),'mean: %.2f' %np.mean(diameters),fontsize=14)
            plt.text(posX,0.91*np.max(heights),'median: %.2f' %np.median(diameters),fontsize=14)
            plt.text(posX,0.86*np.max(heights),'std: %.2f' %np.std(diameters),fontsize=14)
            plt.text(posX,0.81*np.max(heights),'min: %.2f' %np.min(diameters),fontsize=14)
            plt.text(posX,0.76*np.max(heights),'max: %.2f' %np.max(diameters),fontsize=14)
            if boolAfter:
                fig1.savefig('diameters_nkind_'+str(i)+'_New.eps',format='eps',dpi=200)
            else:
                fig1.savefig('diameters_nkind_'+str(i)+'.eps',format='eps',dpi=200)

        os.chdir('../')

#------------------------------------------------------------------------------
def introduce_minimum_and_maximum_diameter_for_vessel_types(G,dLimPen=6,dLimCaps=9):
        """ A minimum diameter for arterioles and venules is given. Every vessel which 
        has a diameter below is considered as capillary. For capillaries a maximum 
        diameter is given. All others are changed to nkind=5. such that they are not considered in any analysis
        INPUT: G: Vascular graph in iGraph format.
            nbins: number of bins for the histogram,
            boolAfter: boolean if the string 'after' should be added to the filename
        OUTPUT: diameter histograms in the folder DimaterVesselTypes
        """
        #Pial arterioles: nkind=0 --> OK
        probs=G.es(nkind_eq=0,diameter_lt=dLimPen).indices
        print('Number of pial arterioles with a too small diameter')
        print(len(probs))
        print(probs)
        print(G.es[probs]['diameter'])
        G.es[probs]['nkind'] = [4]*len(probs)
        
        #Pial venules: nkind = 1 --> OK
        probs=G.es(nkind_eq=1,diameter_lt=dLimPen).indices
        print('Number of pial venules with a too small diameter')
        print(len(probs))
        print(probs)
        print(G.es[probs]['diameter'])
        G.es[probs]['nkind'] = [4]*len(probs)
        
        #penetrating arterioles: nkind =2
        probs=G.es(nkind_eq=2,diameter_lt=dLimPen).indices
        print('Number of penetrating arterioles with a too small diameter')
        print(len(probs))
        print(probs)
        print(G.es[probs]['diameter'])
        G.es[probs]['nkind'] = [4]*len(probs)
        
        #ascending venules: nkind =3
        probs=G.es(nkind_eq=3,diameter_lt=dLimPen).indices
        print('Number of ascending venules with a too small diameter')
        print(len(probs))
        print(probs)
        print(G.es[probs]['diameter'])
        G.es[probs]['nkind'] = [4]*len(probs)

        #The remaining ones are mainly penetrating vessels entering from the side. 
        #Here they are classified as nkind = 5 , this means they are excluded for the data analysis studies
        changeLabel=G.es(nkind_eq=4,diameter_gt=dLimCaps).indices
        print('Number of vessels which are labeled as nkind 5')
        print(len(changeLabel))
        G.es[changeLabel]['nkind']=[5]*len(changeLabel)
        stdout.flush()

        return G
        
#------------------------------------------------------------------------------
def adjust_vertexLabels_to_edgeLabels_for_nkinds(G, valueLabel=1):
        """ Adjusts the label of the tuples of the edges to the value of the edge.
      
        INPUT: G: Vascular graph in iGraph format.
               valueLabel: the values which are adjusted
        OUTPUT: updatedProperties labelXX
        """
        edges=G.es(nkind_eq=valueLabel).indices
        for e in G.es[edges]:
            G.vs[e.tuple]['nkind']=[valueLabel,valueLabel]

        return G
#------------------------------------------------------------------------------
#TODO adjust descriptions from here onwards
def adjust_length_of_vessel_to_fit_oneRBC(G):
        """ Adjusts the label of the tuples of the edges to the value of the edge.
      
        INPUT: G: Vascular graph in iGraph format.
               valueLabel: the values which are adjusted
        OUTPUT: updatedProperties labelXX
        """
        vesselVol=np.array(G.es['diameter'])**2*0.25*np.pi*np.array(G.es['length'])
        G.es['volume']=vesselVol
        vrbcMouse=49.0
        volTooSmall=G.es(volume_lt=vrbcMouse).indices
    
        for i in volTooSmall:
            l=1.1*vrbcMouse/(G.es[i]['diameter']**2*0.25*np.pi)
            G.es[i]['length']=l

        return G
#------------------------------------------------------------------------------
def cut_off_sides_of_MVN(G,percentageToCutOff=15,axis=0):
  #axis = 0,cutting at xMin and xMax site
  #axis = 0,cutting at yMin and yMax site

        eps=finfo(float).eps*1e4

        #cut sides to create dead ends to connect with artificial capillarz bed
        x=[]
        y=[]
        
        for i in G.vs['r']:
              x.append(i[0])
              y.append(i[1])
        
        G.vs['x']=x
        G.vs['y']=y
        if 'center' not in G.attributes():
            center=[np.mean(x),np.mean(y)]
            print('Center is calculated')
        else:
            center=G['center']

        if axis == 0:
            varMinCut=center[0]-(1-0.01*percentageToCutOff)*(center[0]-min(x))
            varMaxCut=center[0]+(1-0.01*percentageToCutOff)*(max(x)-center[0])
        elif axis == 1:
            varMinCut=center[0]-(1-0.01*percentageToCutOff)*(center[0]-min(y))
            varMaxCut=center[0]+(1-0.01*percentageToCutOff)*(max(y)-center[0])
        
        if axis == 0:
            G['xMinCut']=varMinCut
            G['xMaxCut']=varMaxCut
        elif axis == 1:
            G['yMinCut']=varMinCut
            G['yMaxCut']=varMaxCut
        
        G['center']=center

        if axis == 0:
            varMinDelete=G.vs(x_lt=varMinCut).indices
            varMaxDelete=G.vs(x_gt=varMaxCut).indices
        elif axis == 1:
            varMinDelete=G.vs(y_lt=varMinCut).indices
            varMaxDelete=G.vs(y_gt=varMaxCut).indices

        varDeletes=[varMinDelete,varMaxDelete]
        deleteVertices=[]
        deleteEdges=[]
        G.vs['borderVerts']=[0]*G.vcount()
        if axis == 0:
            key='x'
        elif axis == 1:
            key='y'
        for varDel in varDeletes:
            for i in varDel:
                if G.vs['nkind'][i] == 4:
                    inVerts=[]
                    inEdges=[]
                    outVerts=[]
                    for j,k in zip(G.neighbors(i),G.incident(i)):
                        if G.vs[j][key] > varMinCut and G.vs[j][key] < varMaxCut:
                            inVerts.append(j)
                            inEdges.append(k)
                        else:
                            outVerts.append(j)
                    #This checks needs to be de done because we want to create edges of degree = 1, and hence edges crossing the border 
                    #should be kept
                    if len(inVerts) != 0:
                        if len(inVerts) == 1:
                            if G.vs[inVerts[0]]['degree'] == 2:
                                deleteVertices.append(i)
                            else:
                                deleteEdges.append(inEdges[0])
                                deleteVertices.append(i)
                                #The same vertex and edge are reintroduced, such that a degree 1 boundary condition can be assigned
                                #The reintroduced vertex has the key borderVerts
                                G.add_vertices(1)
                                for attr in G.vs.attribute_names():
                                    G.vs[G.vcount()-1][attr]=G.vs[i][attr]
                                G.vs[G.vcount()-1]['borderVerts']=1
                                G.add_edges([(inVerts[0],G.vcount()-1)])
                                for attr in G.es.attribute_names():
                                    if attr in ['points','lengths','lengths2','diameters2','diameters']:
                                        if i > inVerts[0]:
                                            G.es[G.ecount()-1][attr]=G.es[inEdges[0]][attr]
                                        else:
                                            G.es[G.ecount()-1][attr]=G.es[inEdges[0]][attr][::-1]
                                    else:
                                        G.es[G.ecount()-1][attr]=G.es[inEdges[0]][attr]
                        else:
                            for j,k in zip(inVerts,inEdges):
                                deleteEdges.append(k)
                                G.add_vertices(1)
                                for attr in G.vs.attribute_names():
                                    G.vs[G.vcount()-1][attr]=G.vs[i][attr]
                                G.vs[G.vcount()-1]['borderVerts']=1
                                G.add_edges([(j,G.vcount()-1)])
                                for attr in G.es.attribute_names():
                                    if attr in ['points','lengths','lengths2','diameters2','diameters']:
                                        if i > j:
                                            G.es[G.ecount()-1][attr]=G.es[k][attr]
                                        else:
                                            G.es[G.ecount()-1][attr]=G.es[k][attr][::-1]
                                    else:
                                        G.es[G.ecount()-1][attr]=G.es[k][attr]
                            deleteVertices.append(i)
                    else: #whole edge lies outside, all vertices can be deleted
                        deleteVertices.append(i)

        G.delete_edges(np.unique(deleteEdges).tolist())
        G.delete_vertices(np.unique(deleteVertices).tolist())
        G.vs['degree']=G.degree()

        return G

#------------------------------------------------------------------------------
def cut_off_bottom_MVN(G,depthToCutOff=1200):

        z=[]
        for i in G.vs['r']:
            z.append(i[2])

        G.vs['z']=z
        zMaxCut=depthToCutOff
        G['zMaxCut']=zMaxCut
        zMaxDelete=G.vs(z_gt=zMaxCut).indices
        deleteVertices=[]
        deleteEdges=[]
        for i in zMaxDelete:
            if G.vs['nkind'][i] == 4:
                inVerts=[]
                inEdges=[]
                outVerts=[]
                for j,k in zip(G.neighbors(i),G.incident(i)):
                    if G.vs[j]['z'] < zMaxCut:
                        inVerts.append(j)
                        inEdges.append(k)
                    else:
                        outVerts.append(j)
                if len(inVerts) != 0:
                    if len(inVerts) == 1:
                        if G.vs[inVerts[0]]['degree'] == 2:
                            deleteVertices.append(i)
                        else:
                            deleteEdges.append(inEdges[0])
                            deleteVertices.append(i)
                            G.add_vertices(1)
                            for attr in G.vs.attribute_names():
                                G.vs[G.vcount()-1][attr]=G.vs[i][attr]
                            G.vs[G.vcount()-1]['borderVerts']=1
                            G.add_edges([(inVerts[0],G.vcount()-1)])
                            for attr in G.es.attribute_names():
                                if attr in ['points','lengths','lengths2','diameters2','diameters']:
                                    if i > inVerts[0]:
                                        G.es[G.ecount()-1][attr]=G.es[inEdges[0]][attr]
                                    else:
                                        G.es[G.ecount()-1][attr]=G.es[inEdges[0]][attr][::-1]
                                else:
                                    G.es[G.ecount()-1][attr]=G.es[inEdges[0]][attr]
                    else:
                        for j,k in zip(inVerts,inEdges):
                            deleteEdges.append(k)
                            G.add_vertices(1)
                            for attr in G.vs.attribute_names():
                                G.vs[G.vcount()-1][attr]=G.vs[i][attr]
                            G.vs[G.vcount()-1]['borderVerts']=1
                            G.add_edges([(j,G.vcount()-1)])
                            for attr in G.es.attribute_names():
                                if attr in ['points','lengths','lengths2','diameters2','diameters']:
                                    if i > j:
                                        G.es[G.ecount()-1][attr]=G.es[k][attr]
                                    else:
                                        G.es[G.ecount()-1][attr]=G.es[k][attr][::-1]
                                else:
                                    G.es[G.ecount()-1][attr]=G.es[k][attr]
                        deleteVertices.append(i)
                else:
                    deleteVertices.append(i)

        G.delete_edges(np.unique(deleteEdges).tolist())
        G.delete_vertices(np.unique(deleteVertices).tolist())
        G.vs['degree']=G.degree()

        return G


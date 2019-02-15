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
import matplotlib.pyplot as plt
import gc
from scipy import finfo, ones, zeros
from scipy.interpolate import griddata

__all__ = ['prepare_for_concatenating_the_compound_NW']
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def prepare_for_concatenating_the_compound_NW(Ga,Gd):
        """ 1. The edge attribute 'eBlock' labels all non-capillary vessels. Hence, all
        vessels with eBlock=1 should be either arterioles (medianLabelAV=3) or venules 
        (medianLabelAV=4). The medianLabelAV label is adjusted based on neighboring edges.
        2. There might also be As and Vs which do not have an eBlock Label. 
        Their medianLabelAV will be changed to 1 (=capillaries)

        INPUT: G: Vascular graph in iGraph format.
        OUTPUT: updatedProperties eBlock and medianLabelAV
        """

        #1.Shift the realistic network, such that the centers of the NW overlap
        #2.Delete all capillaries of the artificial network which are found in the implantation region
        #3. 
        xMinCut=Gd['xMinCut']
        xMaxCut=Gd['xMaxCut']
        yMinCut=Gd['yMinCut']
        yMaxCut=Gd['yMaxCut']
        zMaxCut=Gd['zMaxCut']
        centerGd=Gd['center']
        print('shift realistic network by')
        origin = np.mean(Ga.vs['r'], axis=0)[:2]
        shift = (np.append(origin,0.)-np.append(centerGd,Gd['zMeanPial'])).tolist()
        print(shift)
        Gd.vs['r']=[v['r']+np.array(shift) for v in Gd.vs]
        #vgm.shift(Gd, shift)
        Ga.vs['z'] = [r[2] for r in Ga.vs['r']]
        Ga.vs['x'] = [r[0] for r in Ga.vs['r']]
        Ga.vs['y'] = [r[1] for r in Ga.vs['r']]
        xMinCutGa=xMinCut + shift[0]
        xMaxCutGa=xMaxCut + shift[0]
        yMinCutGa=yMinCut + shift[1]
        yMaxCutGa=yMaxCut + shift[1]
        zMaxCutGa=zMaxCut + shift[2]
        centerGa=[centerGd[0]+shift[0],centerGd[1]+shift[1]]
        Ga['xMinCut']=xMinCutGa
        Ga['xMaxCut']=xMaxCutGa
        Ga['yMinCut']=yMinCutGa
        Ga['yMaxCut']=yMaxCutGa
        Ga['zMaxCut']=zMaxCutGa
        Ga['center']=centerGa

        #First all capillary vertices lying in the center of ther artificial network ar deleted
        delVerts=Ga.vs(x_gt=xMinCutGa,x_lt=xMaxCutGa,y_gt=yMinCutGa,y_lt=yMaxCutGa,z_lt=zMaxCutGa,nkind_eq=4).indices
        Ga.delete_vertices(delVerts)

        #Now all artificial penetrating trees and artificial venules are deleted if their root point is lying
        #in the area of the implantation including points of the penetrting trees 
        #lying outside the implantation regio
        #Arteriol roots
        delVertsPA=Ga.vs(x_gt=xMinCutGa,x_lt=xMaxCutGa,y_gt=yMinCutGa,y_lt=yMaxCutGa,z_lt=zMaxCutGa,av_eq=1).indices
        delVertsWholeTree=[]
        doneVerts=[]
        for i in delVertsPA:
            treeVertices2=[i]
            while treeVertices2 != []: 
                treeVertices=[]
                for j in treeVertices2:
                    neighbors=Ga.neighbors(j)
                    for k in neighbors:
                        if Ga.vs[k]['nkind'] == 2 and k not in doneVerts:
                            treeVertices.append(k)
                            delVertsWholeTree.append(k)
                    doneVerts.append(j)
                treeVertices2=deepcopy(treeVertices)
            delVertsWholeTree.append(i)

        #Venule roots
        delVertsPA=Ga.vs(x_gt=xMinCutGa,x_lt=xMaxCutGa,y_gt=yMinCutGa,y_lt=yMaxCutGa,z_lt=zMaxCutGa,vv_eq=1).indices
        delVertsWholeTree=[]
        doneVerts=[]
        for i in delVertsPA:
            treeVertices2=[i]
            while treeVertices2 != []:
                treeVertices=[]
                for j in treeVertices2:
                    neighbors=Ga.neighbors(j)
                    for k in neighbors:
                        if Ga.vs[k]['nkind'] == 3 and k not in doneVerts:
                            treeVertices.append(k)
                            delVertsWholeTree.append(k)
                    doneVerts.append(j)
                treeVertices2=deepcopy(treeVertices)
            delVertsWholeTree.append(i)

        #Eliminate remaining a and v vessels at implant location
        delVertsPA=Ga.vs(x_gt=xMinCutGa,x_lt=xMaxCutGa,y_gt=yMinCutGa,y_lt=yMaxCutGa,z_lt=zMaxCutGa,nkind_eq=2).indices
        Ga.delete_vertices(delVertsPA)

        delVertsPV=Ga.vs(x_gt=xMinCutGa,x_lt=xMaxCutGa,y_gt=yMinCutGa,y_lt=yMaxCutGa,z_lt=zMaxCutGa,nkind_eq=3).indices
        Ga.delete_vertices(delVertsPV)

        #delete unconnected components
        while (len(Ga.components())) > 1:
            delVerts=Ga.components()[len(Ga.components())-1]
            Ga.delete_vertices(delVerts)

        return Ga,Gd

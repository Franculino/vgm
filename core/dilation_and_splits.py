from __future__ import division, print_function
from sys import stdout
import copy
import numpy as np
import vgm
import matplotlib.pyplot as plt
import pdb

def split_edges_prior_to_central_dilation(G='G_standard.pkl', 
                                          edges=[3],
                                          savename='G_standard_split.pkl',cf=0.8):
    """Splits specific edges in two end and one central parts, in order to
    prepare them for central dilation / constriction. 
    WARNING: For newly added edges the source is always the smaller
             vertex index and the target the higher vertex index. This needs to
	     be considered in post-processing (e.g. flow direction, position of RBCs)
    INPUT: G: VascularGraph
           edges: Edges to be split
           savename: Name of VascularGraph with split-edges to be saved to
                     disk.
           cf: Center fraction. E.g. cf=2/3 would split the edge into 1/6,
             4/6, 1/6. HOWEVER, the resulting splitting position depends on the number
             of available points and the spacing between the points.
    OUTPUT: None, VascularGraph written to disk.
    """

    savename2=savename
    edges2=edges
    G = vgm.read_pkl(G)
    G.es(edges2)['split'] = [True for e in edges]
    G.add_points(1.)
    while len(G.es(split_ne=None)) > 0:
        eindex = G.es(split_ne=None).indices[0]
        dfactor = 1.
        vi, ei, dilated_ei = G.central_dilation(eindex, dfactor, cf)
        G.es[ei]['split'] = [None for e in ei]
    del G.es['split']
    stdout.write("\rDilation Step \n")
    vgm.write_pkl(G, savename)

#-------------------------------------------------------------------------------------------

def split_and_evolve_to_steady_state(G='G_standard',edges=[3],fdilation=[1.0],
			   time=4.0e3, ht0=0.4, Greference=None,
                           plotPrms=[0., 4.0e3, 5e1, True],
                           samplePrms=[0., 4.0e3, 2, True],cf=0.8,**kwargs):
    """Test effect of central dilation at a bifurcation of equal-flow daughter
    vessels.
    WARNING: For newly added edges the source is always the smaller
             vertex index and the target the higher vertex index. This needs to
	     be considered in post-processing (e.g. flow direction, position of RBCs)
    INPUT: G: Input Graph as pkl-file (name without the ending .pkl)
  	   edges: list of edges which are dilated
	   fdilation: list of factors by which the edge diameter is changed
	   time: time periode which is evolved
           ht0: initial tube hematocrit value throughout the VascularGraph
                (this is ignored if Greference is defined)
           Greference: VascularGraph from which the initial RBC distribution is
                       to be taken. (If set to None, RBCs will be distributed
                       randomly, respecting the ht0 value supplied) The indices od edges
		       and vertices as well as the direction must exactly the same. Otherwise
		       there will be differences in the positions of RBCs
           plotPrms: plot parameters of RBC position plots (start, stop, step,
                     overwrite)
           samplePrms: sample parameters (start, stop, step, overwrite)
           cf: Center fraction. E.g. cf=2/3 would split the edge into 1/6,
             4/6, 1/6. HOWEVER, the resulting splitting position depends on the number
             of available points and the spacing between the points.
           **kwargs:
               httBC: tube hematocrit boundary condition at inflow
               SampleDetailed:Boolean whether every step should be samplede(True) or
                              if the sampling is done by the given samplePrms(False)
    OUTPUT: None, results written to disk.
    """

    SampleDetailed=False
    if 'SampleDetailed' in kwargs.keys():
        SampleDetailed=kwargs['SampleDetailed']

    filename=G+'.pkl'
    savename='G_split.pkl'
    G = vgm.read_pkl(filename)
    if kwargs.has_key('httBC'):
        for vi in G['av']:
            for ei in G.adjacent(vi):
                    G.es[ei]['httBC']=kwargs['httBC']
    G.add_points(1.)

    G.es['dfactor'] = [None for e in G.es]
    G.es[edges]['dfactor'] = fdilation

    while len(G.es(dfactor_ne=None)) > 0:
        eindex = G.es(dfactor_ne=None).indices[0]
        dfactor = G.es[eindex]['dfactor']
        vi, ei, dilated_ei = G.central_dilation(eindex, dfactor, cf)
        G.es[ei]['dfactor'] = [None for e in ei]
    vgm.write_pkl(G, savename)

    if Greference is not None:
        Gr = vgm.read_pkl(Greference)
        G.es['rRBC'] = copy.deepcopy(Gr.es['rRBC'])
        LSd = vgm.LinearSystemHtd(G, dThreshold=10.0, ht0='current')
    else:    
        LSd = vgm.LinearSystemHtd(G, dThreshold=10.0, ht0=ht0)

    if SampleDetailed:
        LSd.evolve(time=time, method='direct', plotPrms=plotPrms,
               samplePrms=samplePrms,SampleDetailed=True)
    else:
        LSd.evolve(time=time, method='direct', plotPrms=plotPrms,
               samplePrms=samplePrms) 

#-------------------------------------------------------------------------------------------

def split_and_steady_state_noRBCs(G='G_standard',edges=[3],fdilation=[1.0],cf=0.8):
    """Test effect of central dilation at a bifurcation of equal-flow daughter
    vessels.
    WARNING: For newly added edges the source is always the smaller
             vertex index and the target the higher vertex index. This needs to
	     be considered in post-processing (e.g. flow direction, position of RBCs)
    INPUT: G: Input Graph as pkl-file (name without the ending .pkl)
           edges: list of edges which are dilated
           fdilation: list of factors by which the edge diameter is changed
           cf: Center fraction. E.g. cf=2/3 would split the edge into 1/6,
             4/6, 1/6. HOWEVER, the resulting splitting position depends on the number
             of available points and the spacing between the points.
    OUTPUT: None, results written to disk.
    """
    filename=G+'.pkl'
    savename='G_split.pkl'
    G = vgm.read_pkl(filename)
    G.add_points(1.)

    G.es['dfactor'] = [None for e in G.es]
    G.es[edges]['dfactor'] = fdilation

    while len(G.es(dfactor_ne=None)) > 0:
        eindex = G.es(dfactor_ne=None).indices[0]
        dfactor = G.es[eindex]['dfactor']
        vi, ei, dilated_ei = G.central_dilation(eindex, dfactor, cf)
        G.es[ei]['dfactor'] = [None for e in ei]
    vgm.write_pkl(G, savename)

    LSd = vgm.LinearSystem(G)

    LSd.solve(method='direct')

# -------------------------------------------------------------------------------------------

def transient_dilation(G='G_standard',edges=[240, 243, 246, 249], 
                  fdilation=[1.1, 1.1, 0.9, 0.9], ttotal=900,ht0=0.4,
                  ttransient=[400, 100, 10], plotstep=2, samplestep=2,**kwargs):
    """Load VascularGraph from disk. Run RBC-transport simulation, where a
    vessel is dilated at some point during the simulation. The dilation occurs
    in several steps. Note that there are some hard-coded variables which need to be
    adjusted if use-case changes!
    INPUT: G: Input Graph as pkl-file (name without the ending .pkl) 
	   edges: List of edges to dilate
           fdilation: List of dilation-factors
           ttotal: total time
           ht0: initial tube hematocrit value throughout the VascularGraph
           ttransient: [tinitial, tduration, steps]= [start of the dilation, 
			time period till dilation is finished, number of steps
			for dilation]
           plotstep: timestep for plotting
	   samplestep: timestep for sampling
           **kwargs:
               httBC: tube hematocrit boundary condition at inflow
	       bigger: 0 = 2 Vessel network, 1= 2in2 Vessel network 2=Honeycomb
	       wholeBr: Boolean whether the whole Branch is dilated (TRUE) or only
		center of the Branch is dilated (FALSE)
    OUTPUT: None, pre- and post-dilation VascularGraph, sample-dictionary, and
            RBC-plots are written to disk.
    """

    filename=G+'.pkl'
    G = vgm.read_pkl(filename)
    if kwargs.has_key('httBC'):
        for vi in G['av']:
            for ei in G.adjacent(vi):
                    G.es[ei]['httBC']=kwargs['httBC']
        
    if kwargs.has_key('bigger'):
        NW=kwargs['bigger']
    else:
        NW=0
    

    if kwargs.has_key('wholeBr'):
        wholeBr=kwargs['wholeBr']
    else:
	wholeBr=False


    G.add_points(1.)
    dilatedList=[]
    
    G.es['dfactor'] = [None for e in G.es]
    G.es[edges]['dfactor'] = 1.0
    
    if wholeBr:
        dilatedList=edges
    else:
        while len(G.es(dfactor_ne=None)) > 0:
            eindex = G.es(dfactor_ne=None).indices[0]
            dfactor = G.es[eindex]['dfactor']
            vi, ei, dilated_ei = G.central_dilation(eindex, dfactor, 4/5.)
            G.es[ei]['dfactor'] = [None for e in ei]
   	    dilatedList.append(dilated_ei)
        for i in range(len(dilatedList)):
            dilatedList[i]=dilatedList[i]-(len(dilatedList)-i-1)
    
    Gd = copy.deepcopy(G)
    LSd = vgm.LinearSystemHtdWCyth(Gd, dThreshold=10.0, ht0=ht0)
    
    #Run simulation without dilation till tinitial is reached
    LSd.evolve(time=ttransient[0], method='direct', 
           plotPrms=[0, ttransient[0], plotstep],
           samplePrms=[0, ttransient[0], samplestep])

    
    #LSd._plot_sample_average('G_pre_dilation.vtp')
    #vgm.write_pkl(Gd, 'G_pre_dilation.pkl')

    tstep = ttransient[1] / ttransient[2]
    Gd.es['dfactor'] = [None for e in G.es]
    Gd.es[dilatedList]['dfactor'] = fdilation
    fsteps = [Gd.es[edge]['diameter'] * (fdil - 1) / ttransient[2]
              for edge, fdil in zip(dilatedList, fdilation)]
    for step in xrange(ttransient[2]):
        for edge, fstep in zip(dilatedList, fsteps):
            Gd.es[edge]['diameter'] += fstep
            stdout.write("\rEdge = %g \n" %edge)
            stdout.write("\rDiameter = %f \n" %Gd.es[edge]['diameter'])
	#START: Consider Change of minDist during dilation of vessel
        LSd._update_minDist_and_nMax(esequence=dilatedList)
	LSd._update_tube_hematocrit(esequence=dilatedList)
	#END: Consider Change of minDist during dilation of vessel
        LSd._update_nominal_and_specific_resistance(esequence=dilatedList)
        LSd._update_eff_resistance_and_LS()
        LSd.evolve(time=tstep, method='direct',
               plotPrms=(0.0, tstep, plotstep),
               samplePrms=(0.0, tstep, samplestep),init=False)
        filename2='G_transient_dilation_'+str(step)+'.vtp'
        LSd._plot_sample_average(filename2)
        vgm.write_pkl(Gd, 'G_transient_dilation_'+str(step)+'.pkl')
        stdout.write("\rDilation Step = %g \n" %step)
        #stdout.write("\rDiameter = %f \n" %Gd.es[342]['diameter'])


    
    tstep = ttotal - ttransient[0] - ttransient[1]
    LSd.evolve(time=tstep, method='direct',
               plotPrms=(0.0, tstep, plotstep),
               samplePrms=(0.0, tstep, samplestep),init=False)

    vgm.write_pkl(Gd, 'G_post_dilation.pkl')
    vgm.write_vtp(Gd, 'G_post_dilation.vtp', False)



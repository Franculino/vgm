from __future__ import division

import numpy as np

__all__ = ['Exchange']


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class Exchange(object):
    """This class implements the exchange of a passive tracer between two 
    VascularGraphs. Each vertex of one VascularGraph can connect to an 
    arbitrary number of vertices in the other. These connections do not show up
    as edges in the VascularGraphs, but are stored as 'soft links' in the 
    Connector of the Exchange instance. 
    """
    
    def __init__(self, G1, G2, Connector, substance, strategy):
        """Initializes an Exchange instance.
        INPUT: G1: First VascularGraph.
               G2: Second VascularGraph.
               Connector: Connector between G1 and G2.
               substance: Name of the substance that is exchanged.
               strategy: Name of the strategy that should be used for the
                         exchange. Possible options include:
                         'simple'
        OUTPUT: None       
        """
        self._G1 = G1
        self._G2 = G2
        self._substance = substance
        self._strategy = strategy
        if strategy == 'simple':
            self.exchange = self.exchange_simple
            self.update_timestep = self.update_timestep_simple
        else:
            raise KeyError('Unknown exchange strategy!')
        self.update_connector(Connector)
        self.update_volume() 
        self.update_concentration()
        self.update_timestep()                                       
                                   
                               
    def update_volume(self):
        """Produces a local copy of the current vertex volumes of G1 and G2.
        INPUT: None
        OUTPUT: None 
        """
        self._volumeG1 = self._G1.vertex_volume()
        self._volumeG2 = self._G2.vertex_volume()
        
    
    def update_connector(self, Connector):
        """Updates the Connector used to 'soft link' G1 and G2.
        INPUT: Connector: Instance of the Connector class. Depending on the 
                          exchange strategy used, this Connector needs to 
                          fulfill certain requirements (e.g. have a property 
                          called exchangeCoefficient).
        OUTPUT: None                    
        """
        self._Connector = Connector
    
        
    def update_concentration(self):    
        """Produces a local copy of the current substance concentrations of G1 
        and G2.
        INPUT: None
        OUTPUT: None 
        """
        self._concentrationG1 = self._G1.get_concentration(self._substance)
        self._concentrationG2 = self._G2.get_concentration(self._substance)
    

    def exchange_simple(self, **kwargs):
        """Exchanges the substance between the VascularGraphs G1 and G2 based
        on the assumption that the amount exchanged depends linearly on the 
        surface area, concentration difference, and exchange coefficient (see
        Reichold et al., JCBFM, 2009).
        INPUT: **kwargs
               steps: Number of time-steps to advect (using the current 
                      time-step self.dt). 
               time: Advection duration (self.dt is adjusted such that the CFL-
                     criterion is fulfilled and self.dt is a proper divisor
                     of time).   
        OUTPUT: None, G1 and G2 are change in-place.
        """        
        
        # compute step size and number of steps:
        if kwargs.has_key('steps'):
            nSteps = kwargs['steps']
            dt = self.dt
        elif kwargs.has_key('time'):
            time = kwargs['time']
            dt = time / np.ceil(time / self.dt)
            nSteps = int(time / dt)
        else:
            raise KeyError
        
        

        G1 = self._G1
        G2 = self._G2        
        for step in range(nSteps):
            oldCG1 = self._concentrationG1
            oldCG2 = self._concentrationG2
            for c in self._Connector.connections:
                concentrationDifference = oldCG1[c.i1] - oldCG2[c.i2]
                from1To2 = c.exchangeFactor * concentrationDifference
    
                # update concentrations in the VascularGraphs:
                G1.vs[c.i1]['substance'][self._substance] -= dt * \
                                                from1To2 / self._volumeG1[c.i1]
                G2.vs[c.i2]['substance'][self._substance] += dt * \
                                                from1To2 / self._volumeG2[c.i2]
            self.update_concentration()                                                                                                          

                
    def update_timestep_simple(self):
        """Updates timestep value to ensure numerical stability.
        The computation of the time step is based on the fact that all 
        coefficients in the discretized exchange equation need to have the same
        sign (positive in this case). If volume, surface area and exchange 
        coefficient do not change during the simulation, it only needs to be 
        executed once at the beginning.
        WARNING: This will only ensure numerical stability. To be physically 
                 sound, the timestep needs to be small enough that only so much
                 substance flows between vertices, which at most equates their 
                 respective concentrations!
        INPUT: None
        OUTPUT: None
        """
        Connector = self._Connector
        volumeG1 = self._volumeG1
        volumeG2 = self._volumeG2

        tMin = 1e200
        # Treat G1:
        for c in Connector.connections:
            dt = volumeG1[c.i1] / c.exchangeFactor
            if dt < tMin:
                tMin = dt
        # Treat G2:
        for vertex in Connector.vertexG2ToVerticesG1.keys():
            exchangeFactor = 0.0
            for neighbor in Connector.vertexG2ToVerticesG1[vertex]:
                exchangeFactor = exchangeFactor + \
                                 Connector.connections[neighbor].exchangeFactor  
            dt = volumeG2[vertex] / exchangeFactor
            if dt < tMin:
                tMin = dt
        
        self.dt = tMin                       
            

            
        
        

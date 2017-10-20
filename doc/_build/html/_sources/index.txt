.. Vascular Graph Model documentation master file, created by
   sphinx-quickstart on Sat Feb 25 15:56:26 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Vascular Graph Model documentation
==================================

Introduction
------------

The Python package *Vascular Graph Model* (VGM) is a computational modeling
framework for the simulation of cerebral blood flow in realistic vascular
networks. The cortical vasculature (and, optionally, the brain
tissue) is internally represented by an unstructured computational grid. Every 
edge in this VascularGraph corresponds to a blood vessel, and every node
stands for a bifurcation or end-point.

Many physiological (*in vivo*) effects such as the dynamic distribution of red
blood cells, and the existence of an endothelial surface layer are implemented.

Moreover, the package contains tools to post-process high resolution
angiography data obtained by *synchrotron radiation X-ray tomographic
microscopy* (SRXTM).

For an in-depth explanation of the fluid-dynamical foundations of the model, as
well as some applications, please consult the following published works:

    | Reichold, J.; Stampanoni, M.; Keller, A. L.; Buck, A.; Jenny, P. & Weber, B. 
    | **Vascular graph model to simulate the cerebral blood flow in realistic vascular networks**
    | *J Cereb Blood Flow Metab*, 2009, 29, 1429-43

    | Obrist, D.; Weber, B.; Buck, A. & Jenny, P. 
    | **Red blood cell distribution in simplified capillary networks**
    | *Philos Transact A Math Phys Eng Sci*, 2010, 368, 2897-918


Contents
--------

.. toctree::
    :maxdepth: 2

    installation
    tutorial
    todo
    vgm



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


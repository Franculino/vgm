Installation
============

VGM is a Python module that was written for a version of Python < 3.0, but can
probably be ported to Python 3.x (using `2to3
<http://docs.python.org/library/2to3.html>`_) with little effort, once all
dependent packages have been ported.  


Required C-libraries
--------------------

* iGraph

`iGraph <http://www.igraph.sourceforge.net/>`_ is the backbone of the
VGM-package. It implements graph objects and a multitude of methods with which
to manipulate and measure graph traits. The software is written in the
programming language C resulting in fast execution times. The corresponding
python-package used in the VGM is a front-end to the C-library and therefore
benefits from the speed of the compiled language. 
Installation is relatively straight forward. Please follow the instructions
given on the `iGraph webpage <http://www.igraph.sourceforge.net/>`_.


Required python packages
------------------------

* `Numpy <http://pypi.python.org/pypi/numpy/>`_ (*)
* `Scipy <http://pypi.python.org/pypi/scipy/>`_ (*)
* `Matplotlib <http://pypi.python.org/pypi/matplotlib/>`_ (*)
* `Cython <http://pypi.python.org/pypi/Cython/>`_ (*)
* `PyAMG <http://pypi.python.org/pypi/pyamg/>`_
* `Quantities <http://pypi.python.org/pypi/quantities/>`_
* `Python-igraph <http://pypi.python.org/pypi/python-igraph/>`_
* `xlrd <http://pypi.python.org/pypi/xlrd/>`_
* `xlwt <http://pypi.python.org/pypi/xlwt/>`_
* `xlutils <http://pypi.python.org/pypi/xlutils/>`_
* `nose <http://pypi.python.org/pypi/nose/>`_

In principle, VGM should work with any Python distribution if all required
packages are installed. However, it is highly recommended to use a *scientific*
Python distribution such as the one contained in the open source mathematics
software `SAGE <http://sagemath.org/>`_. 
A proprietary alternative to SAGE is the `Enthought Python
Distribution <http://www.enthought.com/products/getepd.php>`_. Here, however,
only the SAGE distribution will be described.

SAGE comes with a wealth of python packages already installed, including all
packages above that are marked by (*). Moreover, SAGE includes `ATLAS
<http://math-atlas.sourceforge.net/>`_, a `BLAS <http://www.netlib.org/blas/>`_
installation automatically tuned to the host architecture - highly beneficial
for the performance of the packages *numpy* and *scipy*, among others.

In order to install SAGE, please follow the `installation guide
<http://www.sagemath.org/doc/installation/>`_. In order to install the
non-included python packages, download and unzip the packages, ``cd`` to the
respective directory and issue::
    
    sage -python setup.py install

Note that this requires the executable ``sage`` to be in your ``PATH``.
Finally, place the *VGM* folder in
``$SAGEROOT/local/lib/python/site-packages/`` (where ``$SAGEROOT`` is your SAGE
root directory) and compile the cython files (those, that have an .spyx
extension) by ``cd``-ing to the appropriate directory, evoking SAGE and
entering::

    cython_create_local_so('name_of_module.spyx')

Where "name_of_module" obviously needs to be replaced by the actual module names.
Before this can be done, the small SAGE fix described below needs to be
applied.

Finally, if all steps have been performed, you can test your installation by::

    sage -ipython
    import vgm

Which should not produce any errors.
``vgm.`` followed by ``tab`` will give you a list of available modules.
``vgm.modulename.functionname?`` will print out the docstring of a function,
which should provide helpful information on the objective, input- and output
parameters of the respective function.

SAGE fix
--------

The cython version included in SAGE has difficulties with the directive::

    from __future__ import xyz

This is addressed in the patch provided here. Please replace the file
``$SAGEROOT/devel/sage-main/build/sage/misc/cython.py`` by `this customized
version <../../patch/cython.py>`_.


Visualization software
----------------------

There are a number of open source, multi-platform software solutions that are
suitable for the visualisation of VascularGraph geometry, as well as vertex and
edge based properties. Three prominent candidates are

* `Paraview <http://www.paraview.org>`_
* `VisIt <https://wci.llnl.gov/codes/visit/home.html>`_
* `Mayavi <http://code.enthought.com/projects/mayavi/>`_

All of the above build on the `VTK visualisation toolkit
<http://www.vtk.org/>`_ and are similar in functionality.
:mod:`g_output` contains various methods to write VascularGraphs to disk. Among
them are :mod:`g_output.write_vtp`, :mod:`g_output.write_pvd_time_series`,
which produce Paraview/VisIt/Mayavi-readable output.


.. toctree::
    :maxdepth: 2

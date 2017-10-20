# cython: profile=True
#from __future__ import division, with_statement

from copy import deepcopy
import cPickle
import matplotlib.pyplot as plt
import numpy as np
import cython
cimport numpy as np
cimport libc.stdio as stdio
from pylab import is_string_like

__all__ = ['write_mv3d', 'write_vtp', 'write_pvd_time_series', 'write_graphml',
           'write_pkl', 'write_amira_mesh_ascii', 'write_landmarks']

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def write_mv3d(Graph,filename):
    """Writes a vascular graph to disk in mv3d format, which is used by 
    John Kaufhold's threshold relaxation routine
    INPUT: Graph:  Vascular graph in iGraph format.
           filename: Name of the mv3d to be written.
    OUTPUT: Mv3d-file written to disk. Note that the mv3d format contains 
            diameter information, rather than radii as is the case in the AMIRA
            Mesh format.      
    """
    
    G = Graph

    numOfPoints = 0
    for e in G.es:
        numOfPoints += len(e['points'])
    f = open(filename,'w')
    f.write('# MicroVisu3D file\n')
    f.write('# Number of lines   '+str(G.ecount())+'\n')
    f.write('# Number of points  '+str(numOfPoints)+'\n')
    f.write('# Number of inter.  '+str(G.vcount())+'\n')
    f.write('#\n')
    f.write('# No\t\tx\t\ty\t\tz\t\td\n')
    f.write('#\n')
    for i,e in enumerate(G.es):        
        for j,p in enumerate(e['points']):
            f.write('%d\t%.6f\t%.6f\t%.6f\t%.6f\n' % \
                    (i,p[0],p[1],p[2],e['diamteters'][j]))
        f.write('\n')    
    f.close()
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

cdef void write_array(stdio.FILE* f, array, name, int zeros=0, verbose=False):
    """Print arrays with different number of components, setting very low
    values (which paraview cannot handle) and NaNs to 'substitute'.
    Optionally, a given number of zero-entries can be prepended to an
    array. This is required when the graph contains unconnected vertices.
    TODO: Make printing integer arrays work!
    """    
    tab = "  "
    cdef float substituteD = -1000.
    cdef int substituteI = -1000
    cdef float zeroD = 0.
    cdef int zeroI = 0
    cdef long Nai
    cdef np.ndarray[long, ndim=1] Naj
    try:
        noc = np.shape(array)[1]
        firstel = array[0][0]
        Nai = len(array)
        Naj = np.array(map(len, array), dtype='int')
    except:
        noc = 1    
        firstel = array[0]
        Nai = len(array)
        Naj = np.array([0], dtype='int')

      
    if type(firstel) == str:
        if verbose:
            print "WARNING: array '%s' contains data of type 'string'!" % name    
        return    # Cannot have string-representations in paraview.
    if "<type 'NoneType'>" in map(str, np.unique(np.array(map(type, array)))):
        if verbose:
            print "WARNING: array '%s' contains data of type 'None'!" % name
        return        
    if any([type(firstel) == x for x in 
             [float, np.float32, np.float64, np.float128]]):
        atype = "Float32"
        format = "%f"
    elif any([type(firstel) == x for x in 
             [int, np.int8, np.int32, np.int64]]):
        #atype = "Float32"
        #format = "%f"     
        atype = "Int32"
        format = "%i"
    else: 
        if verbose:
            print "WARNING: array '%s' contains data of unknown type!" % name  
        return
                  
    stdio.fprintf(f, '%s<DataArray type="%s" Name="%s" ',  cs(4*tab), cs(atype), cs(name))
    stdio.fprintf(f, 'NumberOfComponents="%i" format="ascii">\n', ci(noc))
    
    fstringAp = "%s"+format+"\n"        
    fstringBp = format + " "
    spacep = 5*tab
    

    
    cdef:
        char* fstringA = fstringAp
        char* fstringB = fstringBp
        char* space = spacep
        np.ndarray[double, ndim=1] aoD
        np.ndarray[double, ndim=2] atD
        np.ndarray[long, ndim=1] aoI
        np.ndarray[long, ndim=2] atI        
        int i, j
        #Py_ssize_t i, j
            
    if noc == 1:
        if atype == "Float32": 
            for i in xrange(zeros):
                stdio.fprintf(f, fstringA, space, zeroD)        
            aoD = np.array(array, dtype='double')            
            for i in xrange(Nai):
                if not np.isfinite(aoD[i]):
                    stdio.fprintf(f, fstringA, space, substituteD)
                else:
                    stdio.fprintf(f, fstringA, space, aoD[i])            
        elif atype == "Int32":
            for i in xrange(zeros):
                stdio.fprintf(f, fstringA, space, zeroI)        
            aoI = np.array(array, dtype=np.int64)
            for i in xrange(Nai):
                if not np.isfinite(aoI[i]):
                    stdio.fprintf(f, fstringA, space, substituteI)
                else:
                    stdio.fprintf(f, fstringA, space, aoI[i])

    else:
        if atype == "Float32": 
            atD = np.array(array, dtype='double')
            for i in range(zeros):
                stdio.fprintf(f, '%s', space)
                for j in range(Naj[0]):
                    stdio.fprintf(f, fstringB, zeroD)
                stdio.fprintf(f, '\n')            
            for i in range(Nai):
                stdio.fprintf(f, '%s', space)
                for j in range(Naj[i]):
                    if not np.isfinite(atD[i, j]):
                        stdio.fprintf(f, fstringB, substituteD)
                    else:
                        stdio.fprintf(f, fstringB, atD[i, j])
                stdio.fprintf(f, '\n')            
        elif atype == "Int32":
            # Note that array needs to be of type int32, otherwise the printed
            # numbers will be wrong (wrapping?!)
            atI = np.array(array, dtype=np.int32)
            for i in range(zeros):
                stdio.fprintf(f, '%s', space)
                for j in range(Naj[0]):
                    stdio.fprintf(f, fstringB, zeroI)
                stdio.fprintf(f, '\n')            
            for i in range(Nai):
                stdio.fprintf(f, '%s', space)
                for j in range(Naj[i]):
                    if not np.isfinite(atI[i, j]):
                        stdio.fprintf(f, fstringB, substituteI)
                    else:
                        stdio.fprintf(f, fstringB, atI[i, j])
                stdio.fprintf(f, '\n')            
    stdio.fprintf(f, '%s</DataArray>\n', cs(4*tab))
    
cdef char* cs(object s):
    cdef char* cs = s
    return cs
    
cdef float cf(object f):
    cdef float cf = f
    return cf
    
cdef long ci(object i):
    cdef long ci = i
    return ci        
#------------------------------------------------------------------------------------
def write_vtp(graph, filename, tortuous=True, verbose=False):
    """Writes a graph in iGraph format to a vtp-file (e.g. for plotting with 
    Paraview). Adds an index to both edges and vertices to make comparisons
    with the iGraph format easier.
    INPUT: graph: Graph in iGraph format
           filename: Name of the vtp-file to be written. Note that no filename-
                     ending is appended automatically.
           tortuous: Determines whether or not the physiological, i.e. tortuous 
                     geometry of the vascular graph is plotted. If set to
                     False, the straight cylinder representation is written
                     instead. Note that in the tortuous version, vertices at
                     bifurcations appear multiple times - once in every edge
                     incident to the bifurcation.
           verbose: Whether or not to print to the screen if writing an array 
                    fails.          
    OUTPUT: vtp-file written to disk.
    """
    
    # Make a copy of the graph so that modifications are possible, whithout 
    # changing the original. Add indices that can be used for comparison with
    # the original, even after some edges / vertices in the copy have been 
    # deleted:
    G = deepcopy(graph)
    G.vs['index'] = xrange(G.vcount())
    if G.ecount() > 0:
        G.es['index'] = xrange(G.ecount())
    
    # Delete selfloops as they cannot be viewed as straight cylinders and their
    # 'angle' property is 'nan':
    G.delete_edges(np.nonzero(G.is_loop())[0].tolist())
    
    # Convert the substance dictionary to arrays:
    if 'substance' in G.vs.attribute_names():
        substances = G.vs[0]['substance'].keys()
        for substance in substances:    
            c = []
            for v in G.vs:
                c.append(v['substance'][substance])
            G.vs[substance] = c
        del G.vs['substance']       
            
    cdef:
        long i, j, v1, v2, nPoints_, counter
        double step, dv1, dv2
        np.ndarray[double, ndim=1] aIn, aOut    
        char* tab = "  "
        char* fname = filename                   
        stdio.FILE* f = stdio.fopen(fname, 'w')
        
    # Find unconnected vertices:
    unconnected = np.nonzero([x == 0 for x in G.strength(weights=
                  [1 for i in xrange(G.ecount())])])[0].tolist()

    # Header
    stdio.fprintf(f, '<?xml version="1.0"?>\n')
    stdio.fprintf(f, '<VTKFile type="PolyData" version="0.1" ')
    stdio.fprintf(f, 'byte_order="LittleEndian">\n')
    stdio.fprintf(f, '%s<PolyData>\n', cs(1*tab))
    if tortuous:
        stdio.fprintf(f, '%s<Piece NumberOfPoints="%i" ', \
                cs(2*tab), ci(len(np.vstack(G.es['points']))+len(unconnected)))
    else:
        stdio.fprintf(f, '%s<Piece NumberOfPoints="%i" ', \
                cs(2*tab), ci(G.vcount()))
    stdio.fprintf(f, 'NumberOfVerts="%i" ', ci(len(unconnected)))
    stdio.fprintf(f, 'NumberOfLines="%i" ', ci(G.ecount()))
    stdio.fprintf(f, 'NumberOfStrips="0" NumberOfPolys="0">\n')

    # Vertex data
    keys = G.vs.attribute_names()
    keysToRemove = ['r','pBC','rBC','kind','sBC','inflowE','outflowE','adjacent','mLocation','lDir','diameter']
    for key in keysToRemove:
        if key in keys:
            keys.remove(key)
    stdio.fprintf(f, '%s<PointData Scalars="Scalars_p">\n', cs(3*tab))
    #write_array(f, xrange(G.vcount()),'index',verbose)
    if tortuous:
        write_array(f, np.hstack(G.es['diameters']),
                    'diameter',len(unconnected),verbose)
        nPoints = map(len,G.es['points'])
        nEdges = G.ecount()
        eVertices = G.get_edgelist()
        aOut = np.zeros(np.sum(nPoints))
        for key in keys:
            aIn = np.array(G.vs[key], dtype='double')
            
            counter = 0
            for i in xrange(nEdges):
                v1 = eVertices[i][0]
                v2 = eVertices[i][1]
                dv1 = aIn[v1]
                dv2 = aIn[v2]
                nPoints_ = nPoints[i]
                step = (dv2-dv1) / (nPoints_ - 1.0)
                for j in xrange(nPoints_):
                    aOut[counter] = dv1 + j*step
                    counter += 1
                
            write_array(f, aOut,key,len(unconnected),verbose)
    else:
        for key in keys:
            write_array(f, G.vs[key],key,verbose)
    stdio.fprintf(f, '%s</PointData>\n', cs(3*tab))    

    # Edge data
    keys = G.es.attribute_names()
    keysToRemove = ['diameters','lengths','points','rRBC','tRBC']
    for key in keysToRemove:
        if key in keys:
            keys.remove(key)        
    stdio.fprintf(f, '%s<CellData Scalars="diameter">\n', cs(3*tab))
    #write_array(f, xrange(G.ecount()),'index',len(unconnected),verbose)
    for key in keys:
        write_array(f, G.es[key],key,len(unconnected),verbose)
    stdio.fprintf(f, '%s</CellData>\n', cs(3*tab))    
    
    # Vertices
    stdio.fprintf(f, '%s<Points>\n', cs(3*tab))
    if tortuous:
        if len(unconnected) > 0:
            write_array(f, np.vstack([np.vstack(G.vs(unconnected)['r']),
                                   np.vstack(G.es['points'])]),'r',verbose)
        else:
            write_array(f, np.vstack(G.es['points']),'r',verbose)
    else:
        write_array(f, np.vstack(G.vs['r']),'r',verbose)
    stdio.fprintf(f, '%s</Points>\n', cs(3*tab))
    
    
    # Unconnected vertices
    cdef long vertex
    if unconnected != []:
        stdio.fprintf(f, '%s<Verts>\n', cs(3*tab))
        stdio.fprintf(f, '%s<DataArray type="Int32" ', cs(4*tab))
        stdio.fprintf(f, 'Name="connectivity" format="ascii">\n')
        if tortuous:
            for i in xrange(len(unconnected)):
                stdio.fprintf(f, '%s%i\n', cs(5*tab), i)            
        else:
            for vertex in unconnected:
                stdio.fprintf(f, '%s%i\n', cs(5*tab), vertex)
        stdio.fprintf(f, '%s</DataArray>\n', cs(4*tab))            
        stdio.fprintf(f, '%s<DataArray type="Int32" ', cs(4*tab))
        stdio.fprintf(f, 'Name="offsets" format="ascii">\n')
        for i in xrange(len(unconnected)):
            stdio.fprintf(f, '%s%i\n', cs(5*tab), 1+i)
        stdio.fprintf(f, '%s</DataArray>\n', cs(4*tab))                    
        stdio.fprintf(f, '%s</Verts>\n', cs(3*tab))            
    
    # Edges
    stdio.fprintf(f, '%s<Lines>\n', cs(3*tab))
    stdio.fprintf(f, '%s<DataArray type="Int32" ', cs(4*tab))
    stdio.fprintf(f, 'Name="connectivity" format="ascii">\n')
    cdef long ecount, point
    if tortuous:
        ecount = len(unconnected)
        pcount = []
        for edge in G.es:
            pcount.append(len(edge['points']))
            for point in xrange(pcount[-1]):
                stdio.fprintf(f, '%s%i %i\n', cs(5*tab), ecount, ecount+1)
                ecount += 2
    else:
        for edge in G.get_edgelist():
            stdio.fprintf(f, '%s%i %i\n', cs(5*tab), ci(edge[0]), ci(edge[1]))
    stdio.fprintf(f, '%s</DataArray>\n', cs(4*tab))        
    stdio.fprintf(f, '%s<DataArray type="Int32" ', cs(4*tab))
    stdio.fprintf(f, 'Name="offsets" format="ascii">\n')
    cdef:
        char* space
        np.ndarray[long, ndim=1] pcountcs    
    if tortuous:
        pcountcs = np.cumsum(pcount)
        pspace = 5*tab
        space = pspace
        for i in xrange(len(pcountcs)):
            stdio.fprintf(f, '%s%i\n', space, pcountcs[i])           
    else:
        for i in xrange(G.ecount()):
            stdio.fprintf(f, '%s%i\n', cs(5*tab), 2+i*2)        
    stdio.fprintf(f, '%s</DataArray>\n', cs(4*tab))                    
    stdio.fprintf(f, '%s</Lines>\n', cs(3*tab))    

    # Footer
    stdio.fprintf(f, '%s</Piece>\n', cs(2*tab))
    stdio.fprintf(f, '%s</PolyData>\n', cs(1*tab))
    stdio.fprintf(f, '</VTKFile>\n')              

    stdio.fclose(f)

#------------------------------------------------------------------------------
#def write_vtp_from_pkl(loadName, saveName):
#    """Writes a graph in iGraph format to a vtp-file (e.g. for plotting with
#    Paraview). Adds an index to both edges and vertices to make comparisons
#    with the iGraph format easier.
#    INPUT: loadName: Name of the pkl-file to be loaded. Note that no filename-
#                     ending is appended automatically.
#           saveName: Name of the saved pkl-file
#    OUTPUT: vtp-file written to disk.
#    """
#    # Header 
#    stdio.fprintf(f, '<?xml version="1.0"?>\n')
#    stdio.fprintf(f, '<VTKFile type="PolyData" version="0.1" ')
#    stdio.fprintf(f, 'byte_order="LittleEndian">\n')
#    stdio.fprintf(f, '%s<PolyData>\n', cs(1*tab))
#    if tortuous:
#        stdio.fprintf(f, '%s<Piece NumberOfPoints="%i" ', \
#                cs(2*tab), ci(len(np.vstack(G.es['points']))+len(unconnected)))
#    else:
#        stdio.fprintf(f, '%s<Piece NumberOfPoints="%i" ', \
#                cs(2*tab), ci(G.vcount()))
#    stdio.fprintf(f, 'NumberOfVerts="%i" ', ci(len(unconnected)))
#    stdio.fprintf(f, 'NumberOfLines="%i" ', ci(G.ecount()))
#    stdio.fprintf(f, 'NumberOfStrips="0" NumberOfPolys="0">\n')
#    #Vertex Data
#    stdio.fprintf(f, '%s<PointData Scalars="Scalars_p">\n', cs(3*tab))
#    write_array(f, G.vs[key],key,verbose)
#
#
#------------------------------------------------------------------------------


def write_pvd_time_series(outputFilename, filenameList, timeList=None):
    if timeList == None:
        timeList = range(len(filenameList))
    tab = '    '
    f = open(outputFilename,'w')
    f.write('<?xml version=\"1.0\"?>')
    f.write('<VTKFile type=\"Collection\" version=\"0.1\" \
             byte_order=\"LittleEndian\">')
    f.write(tab + '<Collection>')
    for i, filename in enumerate(filenameList):
        s = "%s%s%f%s%s%s%s" % (tab*2, '<DataSet timestep=\"', timeList[i],
                                '\" group=\"\" part=\"0\"', ' file=\"',
                                filenameList[i], '\"/>')
        f.write(s)
    f.write(tab + '</Collection>')
    f.write('</VTKFile>')


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def write_graphml(GR, outfile):
    """Saves a vascular graph to a GraphML file.
    INPUT: G:        Vascular graph in iGraph format.
           filename: Name (and path) of the output file.
    OUTPUT: File written to disk.
    """
    G = deepcopy(GR)
    # Convert attributes stored as numpy arrays to conventional Python lists. 
    # This is necessary because the igraph graphml routine does not play well 
    # with numpy:    
    for key in G.vs.attribute_names():
        if type(G.vs[0][key]) == np.ndarray:
            G.vs[key] = [x.tolist() for x in G.vs[key]]
    for key in G.es.attribute_names():
        if type(G.es[0][key]) == np.ndarray:
            G.es[key] = [x.tolist() for x in G.es[key]]
    G.write_graphml(outfile)   


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def write_pkl(G,filename):
    """Saves a vascular graph to a Python pickle file.
    INPUT: G:        Vascular graph in iGraph format.
           filename: Name (and path) of the output file.
    OUTPUT: File written to disk.
    """

    with open(filename,'wb') as f:
        cPickle.dump(G,f,protocol=2)
    

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class LabelField(object):
    """Implements the LabelFields used in the AmiraMesh file format.
    """
    def __init__(self, name, groupNames, evType, idList, colors=None):
        """Initializes a LabelField instance.
        INPUT: name: The name of the LabelField as string.
               groupNames: The names of the groups that the LabelField 
                           comprises as a list of strings.
               evType: The type of LabelField, which can be either 'EDGE' or
                       'VERTEX' (note the capitals).
               idList: A list of IDs, which defines the group that each edge
                       (or vertex) belongs to.
               colors: A list of two rgb 3-tuples that define the coloring of 
                       the groups. If not provided, a blue to red coloring will 
                       be used as the default.
        OUTPUT: None                
        """                       
        self.name = name
        self.groupNames = groupNames
        self.evType = evType
        self.idList = idList
        if colors is None:
            colors = ((0., 0., 1.), (1., 0., 0.))
        self.update_colors(colors[0], colors[1])
                
        
    def update_colors(self, rgb0, rgb1):
        """Updates the colors of the LabelField groups by linear interpolation
        between two red-green-blue values.
        INPUT: rgb0: The rgb value of the group with the lowest index as a 
                     3-tuple.
               rgb1: The rgb value of the group with the highest index as a
                     3-tuple.
        OUTPUT: None, the attribute self.colors of the LabelField is modified
                in-place.
        """
        N = len(self.groupNames) + 1  # Accounting for the unnamed Id0 group
        red = np.linspace(rgb0[0], rgb1[0], N)
        green = np.linspace(rgb0[1], rgb1[1], N)
        blue = np.linspace(rgb0[2], rgb1[2], N)
        self.colors = zip(red, green, blue)

    @staticmethod
    def from_graph_data(G, attribute, evType, lfName, intervals, **kwargs):
        """Constructs LabelField instance from graph attribute.
        INPUT: G: VascularGraph from which the LabelField is to be constructed.
               attribute: Name of the relevant graph attribute.
               evType: The type of LabelField, which can be either 'EDGE' or
                       'VERTEX' (note the capitals).               
               lfName: The name of the LabelField as string.
               intervals: The intervals that divide the graph attribute values 
                          in groups. These groups will become the groups of the 
                          LabelField. String format with semicolons dividing
                          groups, colons dividing maximum and minimum within
                          a group and brackets signifying boundaries. I.e.: 
                          round brackets indicate exclusiveness, angle brackets
                          indicate inclusiveness.
                          Note that all values that do not fall into any of the
                          defined intervals will be put into the Id0 group.
                          E.g. '[3]; (3,5]' will create a group of values that
                          are exactly equal to 3, a group of values that are 
                          larger than 3 and smaller or equal to 5, as well as a
                          group of values that do not fall into any of the two
                          aforementioned groups.
               **kwargs:
               groupNames: The names of the groups that the LabelField 
                           comprises as a list of strings.           
               colors: A list of two rgb 3-tuples that define the coloring of 
                       the groups. If not provided, a blue to red coloring will
                       be used as the default.
        OUTPUT: LabelField instance.
        """
        if evType == 'VERTEX':
            dataSequence = G.vs
        elif evType == 'EDGE':
            dataSequence = G.es
        else:
            raise KeyError
    
        indexList = []
        for s in intervals.split(';'):
            cs = s.split(',')

            if len(cs) == 1:
                cs = cs[0].strip()
                searchString = attribute + '_eq=' + cs[1:-2]
                eval('indexList.append(dataSequence(' + searchString + ').indices)')
            else:
                cs[0] = cs[0].strip()
                cs[1] = cs[1].strip()
                if cs[0][0] == '(':
                    leftBracket = '_gt='
                elif cs[0][0] == '[':
                    leftBracket = '_ge='
                else:
                    raise ValueError
                
                if cs[1][-1] == ')':
                    rightBracket = '_lt='
                elif cs[1][-1] == ']':
                    rightBracket = '_le='
                else:
                    raise ValueError
    
                searchString = attribute + leftBracket + cs[0][1:-1] + ',' + \
                               attribute + rightBracket + cs[1][0:-2]
                eval('indexList.append(dataSequence(' + searchString + ').indices)')

        idList = [0 for i in xrange(len(dataSequence))]
        for i, indices in enumerate(indexList):
            for j in indices:
                idList[j] = i+1
        
        if kwargs.has_key('groupNames'):
            groupNames = kwargs['groupNames']
        else:
            groupNames = []
            for i in xrange(len(indexList)):
                groupNames.append('Group_' + str(i))
                
        if kwargs.has_key('colors'):
            colors = kwargs['colors']
        else:
            colors = None            
                
        return LabelField(lfName, groupNames, evType, idList, colors)                                                           
    
    
def write_amira_mesh_ascii(G,filename,edgeData=None,vertexData=None,
                           LabelFields=None):
    """Saves a vascular graph to AmiraMesh ASCII format.
    INPUT: G: Vascular graph in iGraph format.
           filename: Name (and path) of the output file.
           edgeData: Name of edge data to be included (as list), optional.
                     Currently, the data is assumed to be of float[1] type.
           vertexData: Name of verex data to be included (as list), optional.
                       Currently, the data is assumed to be of float[1] type.
           LabelFields: List of LabelFields to be included in the AmiraMesh.
           OUTPUT: File saved to disk
    """
    
    f = open(filename,'w')
    f.write('# AmiraMesh 3D ASCII 2.0\n\n\n')
    f.write('define VERTEX %i\n' % G.vcount())
    f.write('define EDGE %i\n' % G.ecount())
    f.write('define POINT %i\n\n' % sum([len(e['points']) for e in G.es]))    
    f.write('Parameters {\n')
    if LabelFields is not None:
        for LF in LabelFields:
            f.write('    %s {\n' % LF.name)
            for i, gn in enumerate(LF.groupNames):
                f.write('        %s {\n' % gn)
                f.write('            Color %f %f %f,\n' % (LF.colors[i+1][0], 
                                                           LF.colors[i+1][1], 
                                                           LF.colors[i+1][2]))
                f.write('            Id %i\n' % (i+1))
                f.write('        }\n')
            f.write('        Id 0,\n')
            f.write('        Color %f %f %f\n' % (LF.colors[0][0], 
                                                  LF.colors[0][1], 
                                                  LF.colors[0][2]))
            f.write('    }\n')                            
    f.write('    ContentType "HxSpatialGraph"\n}\n\n')
    f.write('VERTEX { float[3] VertexCoordinates } @1\n')
    f.write('EDGE { int[2] EdgeConnectivity } @2\n')
    f.write('EDGE { int NumEdgePoints } @3\n')
    f.write('POINT { float[3] EdgePointCoordinates } @4\n')
    f.write('POINT { float thickness } @5\n')
    offsetE  = 6
    offsetV  = 6
    offsetLF = 6
    if edgeData is not None:        
        for i, data in enumerate(edgeData):
            f.write('EDGE { float %s } @%i\n' % (data, offsetE+i))
        offsetV = offsetE + len(edgeData)
    else:
        offsetV = offsetE
        
    if vertexData is not None:
        for i, data in enumerate(vertexData):
            f.write('VERTEX { float %s } @%i\n' % (data, offsetV+i))
        offsetLF = offsetV + len(vertexData)
    else:
        offsetLF = offsetV
        
    if LabelFields is not None:
        for i, LF in enumerate(LabelFields):
            f.write('%s { int %s } @%i\n' % (LF.evType, LF.name, offsetLF+i))
    f.write('\n# Data section follows\n')

    f.write('@1\n')
    for v in G.vs:
        f.write('%f %f %f\n' % (v['r'][0], v['r'][1], v['r'][2]))
    f.write('\n@2\n')
    for e in G.get_edgelist():
        f.write('%i %i\n' % (e[0], e[1]))
    f.write('\n@3\n')
    for e in G.es:
        f.write('%i\n' % len(e['points']))
    f.write('\n@4\n')
    for e in G.es:
        for point in e['points']:
            f.write('%f %f %f\n' % (point[0], point[1], point[2]))
    f.write('\n@5\n')
    for e in G.es:
        for radius in e['diameters']/2.0:
            f.write('%f\n' % radius)
    if edgeData is not None:
        for i, data in enumerate(edgeData):        
            f.write('\n@%i\n' % (offsetE+i))
            for e in G.es:
                f.write('%f\n' % e[data])
    if vertexData is not None:
        for i, data in enumerate(vertexData):        
            f.write('\n@%i\n' % (offsetV+i))
            for v in G.vs:
                f.write('%f\n' % v[data])
    if LabelFields is not None:
        for i, LF in enumerate(LabelFields):
            f.write('\n@%i\n' % (offsetLF+i))            
            for gid in LF.idList:
                f.write('%i\n' % gid)                    
              
    f.close()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------    


def write_landmarks(coordinates, filename):
    """Writes an Amira Mesh Landmark file to disk.
    INPUT: coordinates: List of 3D coordinates (list of lists).
           filename: Absolute or relative filename path.
    OUTPUT: None (landmark file written to disk).           
    """
    f = open(filename, 'w')
    f.write('# AmiraMesh 3D ASCII 2.0\n')
    f.write('\n')
    f.write('\n')
    f.write('define Markers %i\n' % len(coordinates))
    f.write('\n')
    f.write('Parameters {\n')
    f.write('    NumSets 1,\n')
    f.write('    ContentType "LandmarkSet"\n')
    f.write('}\n')
    f.write('\n')
    f.write('Markers { float[3] Coordinates } @1\n')
    f.write('\n')
    f.write('# Data section follows\n')
    f.write('@1\n')
    for point in coordinates:
        f.write('%f %f %f\n' % (point[0], point[1], point[2]))
    f.close()
        

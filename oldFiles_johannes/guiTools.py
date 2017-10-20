from __future__ import division

import tkFileDialog
import Tkinter

__all__ = ['uigetdir', 'uigetfile', 'uigetfiles']

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def uigetdir(title='Please select a directory'):
    """Brings up a GUI dialog to select a directory. The directory name is 
    returned.
    INPUT: title: The title of the dialog (optional).
    OUTPUT: Full path-name of the directory that was chosen.
    """

    root = Tkinter.Tk()
    root.withdraw()
    dirname = tkFileDialog.askdirectory(parent=root, initialdir="/", 
                                        title=title)
                                       
    return dirname


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def uigetfile(title='Please select a file'):
    """Brings up a GUI dialog to select a file. The filename is returned. 
    INPUT: title: The title of the dialog (optional).
    OUTPUT: Full path-name of the file that was chosen.
    """

    root = Tkinter.Tk()
    root.withdraw()
    filename = tkFileDialog.askopenfilename(parent=root, title=title)

    return filename


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def uigetfiles(title='Please select files'):
    """Brings up a GUI dialog to select multiple files. The filenames are 
    returned as a list. 
    INPUT: title: The title of the dialog (optional).
    OUTPUT: Full path-name of the files that were chosen as list.
    """

    root = Tkinter.Tk()
    root.withdraw()
    filenames = tkFileDialog.askopenfilenames(parent=root, title=title)

    return filenames


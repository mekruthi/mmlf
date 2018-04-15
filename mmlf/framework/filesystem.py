# Maja Machine Learning Framework
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

# 2007-08-09, Mark Edgington

import os
import time
import tempfile
import warnings

import mmlf

class RefNameExistsError(Exception):
    "Error which is raised when a path is created using an already existing path-reference name"
    pass

class BaseUserDirectory(object):
    """ Represents a user's base-directory (typically $HOME/.mmlf)
    
    In this directory, the user can define worlds, YAML configuration files, 
    and the interaction server logs information while running a world.
    """    
    def __init__(self, basePath=mmlf.getRWPath()):
        self.basePath = basePath
        
        # dictionary of strings used to refer to a path (as the keys), and the
        # strings containing the absolute path for these paths (as the values).
        self.pathDict = {} 
        
        self.timeString = None
        self.setBasePath(basePath) # set path to default location
        
        # Some root directories are added to the base user directory            
        self.createPath(["config"], refName="rwconfig", force=True)
        
        # Set ro_area
        head = os.path.split(os.path.abspath(__file__))[0]
        head = os.path.split(head)[0]
        self.addAbsolutePath(head, "ro_area")
        head = os.path.split(head)[0]
        self.addAbsolutePath(head + os.sep + "config", "roconfig")
        
    
    def setBasePath(self, absolutePath=None):
        """
        Set (and create if necessary) the base user-directory path.  If a path is specified
        (as an absolute path string), then it is used.  Otherwise, the default path is
        used.
        
        If for some reason this fails, then a writeable temporary directory is chosen.
        """
        # if absolute path is specified, then use it.
        if absolutePath != None:
            try:
                self.createPath([], refName='basepath') # try creating the directory specified by self.basePath
                self.basePath = absolutePath
                return # if successful, then exit the method
            except Exception, e:
                warnings.warn("Could not create user directory for given absolute path %s. \n Reason: %s\n" % (absolutePath, e))

        self.basePath = os.path.abspath(mmlf.getRWPath())
        
        try:
            self.createPath([], refName='basepath') # try creating the directory specified by self.basePath
        except Exception, e:
            # if creating the self.basePath directory fails, then we need 
            # to choose a directory in the $TEMP directory which is writeable.
            self.basePath = tempfile.mkdtemp()
            self.pathDict['basepath'] = self.basePath # add to path dictionary
            
            warnings.warn("Could not create default user directory. Using %s instead.\n Reason: %s\n" % (self.basePath, e))
    
    def fileExists(self, pathRef, fileName):
        """
        Check to see if the file (located in the path referred to by pathRef) exists. 
        
        """
        basePath = self.pathDict[pathRef]
        fullPath = os.path.join(basePath, fileName)
        return os.path.exists(fullPath)
    
    def getFileObj(self, pathRef, fileName, fileOpenMode='rb'):
        """
        Get a file handle object for the fileName specified, assuming that the file is located in
        the directory referred to by pathRef (in self.pathDict).  It is the programmer's 
        responsibility to close this file object later on after it has been created..
        
        """
        fullPath = self.getAbsolutePath(pathRef, fileName)
        fileObj = open(fullPath, fileOpenMode)
        return fileObj
        
    def getFileAsText(self, pathRef, fileName):
        """
        Get the contents of the file fileName located in the directory referred to by pathRef, and
        return it as a string.
        
        """
        f = self.getFileObj(pathRef, fileName)
        fileText = f.read()
        f.close()
        
        return fileText
    
    def getAbsolutePath(self, pathRef, fileName=None):
        """
        Return the absolute path to file fileName located in the directory referred to by pathRef.
        If no filename is specified, then only the path corresponding to pathRef is returned.
        
        """
        basePath = self.pathDict[pathRef]

        if fileName != None:
            absPath = os.path.join(basePath, fileName)
        else:
            absPath = basePath
        
        return absPath
    
    def addAbsolutePath(self, absDirPath, pathRef, force=False):
        """
        Add an absolute path to self.pathDict.  This is so that other commands can use this path
        via the pathRef shortcut.
        
        The force argument allows an existing pathRef to be overridden.  Normally this is
        not recommended, unless the pathRef is known not to be used globally.
        
        """
        # verify that an absolute file path has been provided
        assert(absDirPath == os.path.abspath(absDirPath))
        
        # add path to self.pathDict
        if self.pathDict.has_key(pathRef) and (force == False):
            raise RefNameExistsError("The pathRef '%s' already exists." % pathRef)
        else:
            self.pathDict[pathRef] = absDirPath
    
    def appendStringToPath(self, pathRef, stringToAppend=None):
        """
        Modify the absolute path stored under pathRef, appending a "_<timestamp>" string to the end
        of the path (where <timestamp> will be something like 20070929_12_59_59, which represents
        12:59:59 on 2007-09-29).  If _<timestamp> has already been appended to the end of the path,
        it is replaced with a current timestamp.
        
        If stringToAppend is provided, then a timestamp is *NOT* added, and instead, only the
        specified string is appended.
        
        """
        timeString = time.strftime("%Y%m%d_%H_%M_%S") # the current time -- a string of length==15
        
        # create dict if it doesnt exist.  This dict is used for tracking if a pathRef has been
        #  previously modified -- if so, then its original value is stored in the dictionary
        try:
            self.modifiedPathRefs
        except:
            self.modifiedPathRefs = {}
        
        if stringToAppend != None:
            appendString = stringToAppend
        else:
            appendString = "_" + timeString
        
        # reset the 'currentlogdir' baseRef
        if pathRef in self.modifiedPathRefs:
            newDir = self.modifiedPathRefs[pathRef] + appendString
        else:
            currentDir = self.getAbsolutePath(pathRef=pathRef)
            self.modifiedPathRefs[pathRef] = currentDir
            newDir = currentDir + appendString
        
        self.addAbsolutePath(absDirPath=newDir, pathRef=pathRef, force=True)
    
    def setTime(self, timeStr = None):
        """
        Sets the time string of baseUserDir (which is used to distinguish the logs of different runs).
        
        If no timeString is given, it defaults to the current time
        """
        if timeStr != None:
            self.timeString = timeStr
        else:
            ms = str(time.time()).split('.')[1]
            self.timeString = time.strftime("%Y%m%d_%H_%M_%S") + "_" + ms 
    
    def createPath(self, pathList, refName=None, baseRef=None, force=False):
        """
        Attept to create the path specified in pathList.  pathList is a list which contains a series
        of directory names.  For example, it might be ['abc','def','ghi'], which would cause this
        method to create all the necessary directories such that the following path exists:
            $HOME/.mmlf/abc/def/ghi
        
        refName is a quick reference name which can be used to refer to the path.  If specified,
        the full path will be added to a dictionary, in which the key is the value of refName
        (a string, for example).
        
        If the baseRef argument is provided, it will be looked up and used as the base directory to
        which the path defined in pathList will be appended.  For example, if there already exists
        a 'logdir' refName in self.pathDict, then writing createPath(['a','b','c'], baseRef='logdir')
        might create (depending on how 'logdir' is defined) the directories needed for the path
        "$HOME/.mmlf/logs/a/b/c/".
        
        The force argument allows an existing refName to be overridden.  Normally this is
        not recommended, unless the refName is known not to be used globally.
        
        """
        # check to see if the pathDict has the specified baseRef as a key.  If so, use the value as
        # the base path. Otherwise, use self.basePath as the base path
        if (baseRef != None) and self.pathDict.has_key(baseRef):
            basePath = self.pathDict[baseRef]
        else:
            basePath = self.basePath
        
        pathExtension = os.sep.join(pathList)
        targetAbsolutePath = os.path.join(basePath, pathExtension)
        
        # add path to self.pathDict if a reference name has been specified
        if refName != None:
            if self.pathDict.has_key(refName) and (force == False):
                raise RefNameExistsError("The refName '%s' already exists." % refName)
            else:
                self.pathDict[refName] = targetAbsolutePath
        
        # try to create the directory path, and if it already exists, just quietly exit the method
        try:
            os.makedirs(targetAbsolutePath)
        except OSError, (errno, strerror):
            # check to see what the nature of the error was...
            if os.name == 'nt':
                if errno == 183: # if the file/directory exists already
                    pass # ignore the error
                else:
                    raise # raise the error
            else:                
                if errno == 17: # if the file/directory exists already
                    pass # ignore the error
                else:
                    raise # raise the error


class LogFile(object):
    """
    Represents a logfile, to which data can be written.
    
    """
    def __init__(self, baseUserDir, logFileName, baseRef=None):
        """
        Create the LogFile instance.  If baseRef is specified, the path in baseUserDir.pathDict
        with this key will be used as the directory in which the logfile will be created.
        Otherwise, baseUserDir.pathDict['logbase'] is used as the directory where the logfile will
        be created.
        
        """
        self.userDirObj = baseUserDir
        
        if baseRef == None:
            baseRef = 'logbase'
        
        self.baseRef = baseRef
        self.logFileName = logFileName
        
        # add ourself to baseUseDir.logFileDict - TODO: does this create a circular reference?
        # self.userDirObj.logFileDict[logFileName] = self
        
    def addText(self, text):
        """
        add the specified text to the logfile.  Note, no newline is added to the logfile, so this
        must be included in the text string if it is desired.
        
        """
        f = self.userDirObj.getFileObj(self.baseRef, self.logFileName, fileOpenMode='ab') #append,binary
        f.write(text)
        f.close()
        
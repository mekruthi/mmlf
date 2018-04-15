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

import os
import glob

# The root of the search (should be the world directory)
root = os.sep.join(__file__.split(os.sep)[:-1])

# The global dict of MMLF environments and worlds
MMLF_ENVIRONMENTS = {}
MMLF_WORLDS = {}

docstring = \
""" Package that contains all available MMLF world environments.

A list of all environments:
"""

# search all modules in the directory
for file_name in glob.glob(root + "/*/environments/*.py"):
    # Compute the package path for the current directory
    dir_path = file_name.split(os.sep)
    # Find the last occurrence of "MMLF" in the path
    mmlf_index = None
    for index, token in enumerate(dir_path):
        if token == "mmlf": mmlf_index = index
    dir_path = dir_path[mmlf_index:]
    if dir_path[-1].endswith('.py'):
        dir_path[-1] = dir_path[-1][:-3] 
    module_path = ".".join(dir_path)
    package_name, module_name = module_path.split('.')[-3], module_path.split('.')[-1]
    if module_name == "__init__" or not "EnvironmentClass" in open(file_name).read():
        continue
    
    # Import the module
    try:
        module = __import__(module_path, {}, {}, ["dummy"])
    except Exception, e:
        import traceback
        import logging
        logging.warn("Exception during import of %s: %s\n%s" 
                         % (module_name, e, traceback.format_exc()))
        continue
        
    # If this module exports MDP nodes
    if hasattr(module, "EnvironmentClass"):
        MMLF_ENVIRONMENTS[module.EnvironmentName] = {"module_name": module_name,
                                                     "env_class": module.EnvironmentClass} 
        MMLF_WORLDS[module.EnvironmentName] = package_name
        
        docstring += "\t* Environment '*%s*' from module %s\n\t  implemented in class %s.\n" \
                             % (module.EnvironmentName, module_name, module.EnvironmentClass.__name__)
        if module.EnvironmentClass.__doc__:
            for line in module.EnvironmentClass.__doc__.split("\n")[2:]:
                if line.strip() == "": break
                docstring += "\t\t%s\n" % line

__doc__ = docstring

# Clean up...
del(os, root, dir_path, mmlf_index, index, token, module_path, package_name,
    module_name, module, file_name, line, docstring)

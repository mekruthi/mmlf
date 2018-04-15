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

import setuptools
from setuptools import setup
import os
import sys
import shutil

VERSION = '1.0'

# Check that python version is at least 2.6 and not 3
req_version = (2,6)
cur_version = sys.version_info
if (cur_version[0] < req_version[0] or\
    (cur_version[0] == req_version[0] and cur_version[1] < req_version[1])):
    raise Exception("MMLF requires python 2.6 or higher.")
elif cur_version[0] == 3:
    raise Exception("MMLF has not been ported to python 3 yet.")

# Check that all necessary packages are present
required_packages = {"numpy" : "numpy", 
                     "scipy" : "scipy",
                     "pyyaml" : "yaml",
                     "matplotlib" : "matplotlib"}

for package_name, package in required_packages.items():
    try:
        __import__(package)
    except ImportError:
        raise Exception("MMLF requires the %s package to work properly" % package_name)

# Add all config files to be installed to /etc/mmlf
data_files = []
for dirpath, dirnames, filenames in os.walk('config'):
    filenames = map(lambda filename: "%s%s%s" % (dirpath, os.sep, filename), filenames)
    if sys.platform == 'win32':
        data_files.append(("%s\\mmlf\\%s" % (os.environ["APPDATA"],
                                             dirpath), filenames))
    elif sys.platform in ['darwin', 'linux2']:
        data_files.append(("/etc/mmlf/%s" % dirpath, filenames))
    else:
        raise Exception("ERROR: Unknown platform %s" % sys.platform)

# Actually install package
setup(name='mmlf',
      version=VERSION,
      author='Jan Hendrik Metzen, Mark Edgington',
      author_email='jhm@informatik.uni-bremen.de',
      description='The Maja Machine Learning Framework',
      long_description=
"""Maja Machine Learning Framework

The Maja Machine Learning Framework (MMLF) is a general framework for 
problems in the domain of Reinforcement Learning (RL). It provides a 
set of RL related algorithms and a set of benchmark domains. 
Furthermore it is easily extensible and allows to automate 
benchmarking of different agents. 

Among the RL algorithms are TD(lambda), DYNA-TD, CMA-ES,
Fitted R-Max, and Monte-Carlo learning. MMLF contains different 
variants of the maze-world and pole-balancing problem class as 
well as the mountain-car testbed.

Further documentation is available under http://mmlf.sourceforge.net/

Contact the mailing list MMLF-support@lists.sourceforge.net
if you have any questions.""",
      license='GPL',
      url='http://mmlf.sourceforge.net/',
      download_url='http://downloads.sourceforge.net/project/mmlf/mmlf/%s/mmlf_%s.tar.gz' % (VERSION,VERSION),
      platforms='any',
      packages = setuptools.find_packages(),
      scripts = ["run_mmlf", "mmlf_gui"],
      data_files=data_files,
      package_data={'': ['*.cfg']},
      classifiers = [
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: X11 Applications :: Qt",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
)

if sys.argv[1] == 'install' and sys.platform != 'win32': # For LINUX + MacOSX etc.
    # Add starter to GNOME application menu
    if os.path.exists("/usr/share/applications"):
        while True:
            addStarter = raw_input("Do you want to add a starter to GNOME "
                                   "application menu? [yn]")
            if addStarter in ['y', 'Y']:
                shutil.copy("mmlf.desktop", "/usr/share/applications")
                if os.path.exists("doc/logo/MMLF_white.png"):
                    shutil.copy("doc/logo/MMLF_white.png", "/usr/share/icons/MMLF_white.png")
                elif os.path.exists("doc/_build/html/_static/MMLF_white.png"):
                    shutil.copy("doc/_build/html/_static/MMLF_white.png",
                                "/usr/share/icons/MMLF_white.png")
                break
            elif addStarter in ['n', 'N']:
                break
            else:
                print "Invalid input, please type 'y' or 'n'."

    # Install documentation to /usr/share/doc
    if os.path.exists("/usr/share/doc"):
        if os.path.exists("/usr/share/doc/mmlf"):
            shutil.rmtree("/usr/share/doc/mmlf")  
        shutil.copytree("doc/_build/html", "/usr/share/doc/mmlf")
        
elif sys.argv[1] == 'install': # For Windows
    # we write the documentation into the MMLF RW area for a lack of a better place
    docPath = os.environ["USERPROFILE"] + os.sep + ".mmlf" + os.sep + "doc"
    if os.path.exists(docPath):
        shutil.rmtree(docPath)
    shutil.copytree("doc/_build/html", docPath)
    

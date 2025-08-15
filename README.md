# agilent-ir-formats

## Agilent File Format Handling for Infrared Spectroscopy
Author: Alex Henderson <[alex.henderson@manchester.ac.uk](alex.henderson@manchester.ac.uk)>              
Version: 1.0.0  
Copyright: (c) 2018-2025 Alex Henderson   

## About ##
Python package to read hyperspectral image files produced by infrared spectroscopy instrumentation from Agilent Technologies, Inc.
  
Currently, the code reads single or multi-tile images (*.seq files or *.dmt files) 

## Help information
``` python
Class to open, read and export the contents of an Agilent Fourier Transform Infrared (FTIR) microscopy file.

FTIR instruments from Agilent Technologies Inc., that use a focal plane array detector, can store hyperspectral
images in single 'tile' or multi-tile 'mosaic' file formats. This class can read both single and multi-tile images.
Files with a filename extension of *.seq or *.dmt are compatible.

The class has properties and methods allowing the user to explore the numeric values in the file. In addition, some
metadata values are also accessible.

Properties:
    wavenumbers     x-axis values of the spectral dimension.
    data            spectral intensities of the hyperspectral data as a 3D object (height, width, datapoints).
    total_spectrum  sum of intensity in all pixels, as a function of wavenumber.
    total_image     sum of intensity in all pixels as a function of position (height, width).
    metadata        simple metadata relating to these data.
    hdf5_metadata   metadata arranged into a hierarchy for use in HDF5 export of these data.

Methods:
    read()          open and parse a file.
    export_hdf5()   create a representation on disc of the file in the HDF5 file format.

Static methods:
    filetype()      string identifying the type of files this class reads.
    filefilter()    string identifying the Windows file extensions for files this class can read.
    find_files()    list of all readable files in a directory structure.
    isreadable()    whether this class is capable of reading a given file.
    version()       the version number of this code.
```

## Usage ##
### Example 1 ###
Open a file and display simple metadata. 

``` python
from pprint import pprint   # only for this example

from agilent_ir_formats.agilent_ir_file import AgilentIRFile

filename = r"C:\mydata\myfile\myfile.dmt"

reader = AgilentIRFile()
reader.read(filename)

wavenumbers = reader.wavenumbers
intensities = reader.intensities
metadata = reader.metadata

print(wavenumbers.shape)
print(intensities.shape)
pprint(metadata)

# output...

(728,)
(128, 256, 728)
{'acqdatetime': '2023-05-11T14:37:02',
 'filename': WindowsPath('C:/mydata/myfile/myfile.dmt'),
 'firstwavenumber': 898.6699159145355,
 'fpasize': 128,
 'lastwavenumber': 3702.674331665039,
 'numpts': 728,
 'xlabel': 'wavenumbers (cm-1)',
 'xpixels': 256,
 'xtiles': 2,
 'ylabel': 'absorbance',
 'ypixels': 128,
 'ytiles': 1}
```    
### Example 2 ###
Convert a file to HDF5 format in the same location.

``` python
from agilent_ir_formats.agilent_ir_file import AgilentIRFile

filename = r"C:\mydata\myfile\myfile.dmt"

AgilentIRFile(filename).export_hdf5()
```

## Requirements ##
* python >= 3.10  
* h5py
* numpy

## Licence conditions ##
Copyright (c) 2018-2025 Alex Henderson (alex.henderson@manchester.ac.uk)   
Licensed under the MIT License. See https://opensource.org/licenses/MIT      
SPDX-License-Identifier: MIT   
Visit https://github.com/AlexHenderson/agilent-ir-formats/ for the most recent version  

---
### See also:  
* MATLAB code available here: [https://bitbucket.org/AlexHenderson/agilent-file-formats/](https://bitbucket.org/AlexHenderson/agilent-file-formats/)

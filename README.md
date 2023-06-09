# agilent-ir-formats

## Agilent File Format Handling for Infrared Spectroscopy
Author: Alex Henderson <[alex.henderson@manchester.ac.uk](alex.henderson@manchester.ac.uk)>              
Version: 0.1.0  
Copyright: (c) 2018-2023 Alex Henderson   

## About ##
Python package to read hyperspectral image files produced by infrared spectroscopy instrumentation from Agilent Technologies, Inc.
  
Currently, the code reads single or multi-tile images (*.seq files or *.dmt files) 

## Usage ##
``` python
from pprint import pprint   # only for this example

from agilentirformats import AgilentIRFile

filename = r"C:\mydata\myfile\myfile.dmt"

reader = AgilentIRFile()
reader.read(filename)

xvalues = reader.xvalues
intensities = reader.intensities
metadata = reader.metadata

print(xvalues.shape)
print(intensities.shape)
pprint(metadata)

# output...

(728,)
(728, 128, 256)
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

## Requirements ##
* python >= 3.10  
* numpy

## Licence conditions ##
Copyright (c) 2018-2023 Alex Henderson (alex.henderson@manchester.ac.uk)   
Licensed under the MIT License. See https://opensource.org/licenses/MIT      
SPDX-License-Identifier: MIT   
Visit https://github.com/AlexHenderson/agilent-ir-formats/ for the most recent version  

---
### See also:  
* MATLAB code available here: [https://bitbucket.org/AlexHenderson/agilent-file-formats/](https://bitbucket.org/AlexHenderson/agilent-file-formats/)


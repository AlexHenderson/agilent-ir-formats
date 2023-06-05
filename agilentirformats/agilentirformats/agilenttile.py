"""
    Agilent File Format Handling for Infrared Spectroscopy
    Copyright (C) 2018-2023  Alex Henderson <alex.henderson@manchester.ac.uk>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    See the GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import math
from pathlib import Path
import struct

import numpy as np


class AgilentTile:

    def __init__(self):
        super().__init__()
        self.data = None
        self.datatype = np.dtype('<f')  # little-endian single-precision float

    def filetype(self):
        return "Agilent FTIR Single Tile Files"

    def filefilter(self):
        return "*.seq"

    @staticmethod
    def isreadable(filename=None):
        # Check filename is provided.
        if filename is not None:
            # Check file extension.
            filename = Path(filename)
            if filename.suffix.lower() != ".seq":
                return False

            # Additional tests here

            # Passed all available tests so we can read this file
            return True
        else:
            return False

    def _readwinint32(self, binary_file):
        return int.from_bytes(binary_file.read(4), byteorder='little')

    def _readwindouble(self, binary_file):
        return struct.unpack('<d', binary_file.read(8))[0]

    def _readtile(self, filename):
        tile = np.memmap(filename, dtype=self.datatype, mode='r')
        # skip the (unknown) header
        tile = tile[255:]
        tile = np.reshape(tile, (self.numberofpoints, self.fpasize, self.fpasize))
        return tile

    def _getwavenumbersanddate(self):
        bspfilename = self._filename.with_suffix(".bsp")
        with open(bspfilename, "rb") as binary_file:
            binary_file.seek(2228, 0)
            self.startwavenumber = self._readwinint32(binary_file)
            binary_file.seek(2236, 0)
            self.numberofpoints = self._readwinint32(binary_file)
            binary_file.seek(2216, 0)
            self.wavenumberstep = self._readwindouble(binary_file)

            stopwavenumber = self.startwavenumber + (self.wavenumberstep * self.numberofpoints)

            self.wavenumbers = np.arange(1, self.numberofpoints + self.startwavenumber)
            self.wavenumbers = self.wavenumbers * self.wavenumberstep
            self.wavenumbers = np.delete(self.wavenumbers, range(0, self.startwavenumber - 1))

            # # read in the whole file (it's small) and regex it for the acquisition date/time
            # binary_file.seek(0, os.SEEK_SET)
            # contents = binary_file.read()
            # regex = re.compile(b"Time Stamp.{44}\w+, (\w+) (\d\d), (\d\d\d\d) (\d\d):(\d\d):(\d\d)")
            # matches = re.match(regex, contents)
            # matches2 = re.match(b'(T)', contents)

    def _getfpasize(self):
        tilefilename = self._filename.with_suffix(".dat")
        tilesize = tilefilename.stat().st_size
        data = tilesize - (255 * 4)  # remove header
        data = data / self.numberofpoints
        data = data / 4  # sizeof float
        self.fpasize = int(math.sqrt(data))  # fpa size likely to be 64 or 128 pixels square

    def read(self, filename=None):
        """ToDo: If filename is None, open a dialog box"""
        if AgilentTile.isreadable(filename):
            self._filename = Path(filename)

            # Read the .dmt file to get the wavenumbers and date of acquisition
            # Generate the .dmt filename
            self._getwavenumbersanddate()
            self._getfpasize()

            self.xtiles = 1
            self.ytiles = 1

            self.numxpixels = self.fpasize * self.xtiles
            self.numypixels = self.fpasize * self.ytiles

            alldata = np.empty((self.numberofpoints, self.numypixels, self.numxpixels), dtype=self.datatype)

            ystart = 0
            for y in reversed(range(self.ytiles)):
                ystop = ystart + self.fpasize

                xstart = 0
                for x in (range(self.xtiles)):
                    xstop = xstart + self.fpasize

                    tilefilename = self._filename.with_suffix(".dat")
                    tile = self._readtile(tilefilename)

                    alldata[:, ystart:ystop, xstart:xstop] = tile

                    xstart = xstop
                ystart = ystop

            alldata = np.fliplr(alldata)

            info = {'filename': filename,
                    'xpixels': self.numxpixels,
                    'ypixels': self.numypixels,
                    'xtiles': self.xtiles,
                    'ytiles': self.ytiles,
                    'numpts': self.numberofpoints,
                    'fpasize': self.fpasize
                    }
            # kwargs consists of: xlabel, ylabel, xdata, ydata, info.
            return dict(ydata=alldata, ylabel='absorbance',
                        xdata=self.wavenumbers, xlabel='wavenumbers (cm-1)',
                        info=info)

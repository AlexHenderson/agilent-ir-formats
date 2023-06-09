"""
    Agilent File Format Handling for Infrared Spectroscopy
    Copyright (c) 2018-2023  Alex Henderson <alex.henderson@manchester.ac.uk>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the MIT licence.

    For the latest version visit https://github.com/AlexHenderson/agilent-ir-formats/

"""

from datetime import datetime
from enum import Enum
import math
from pathlib import Path
import re
import struct
from typing import Optional

import numpy as np


class AgilentIRFile:

    class TileOrMosaic(Enum):
        """Embedded Enumeration class to hold flags for the types of data the outer class can cope with.
        """
        TILE = 1    # : Single tile images from a focal plane array experiment.
        MOSAIC = 2  # : Multiple single tile images arranged into a mosaic.

    @staticmethod
    def filetype() -> str:
        """Returns a string identifying the type of files this class reads.
        The text is suitable for a Windows OpenFile dialog box.

        :return: String describing the file types this class can read.
        :rtype: str
        """
        return "Agilent FTIR Files (*.dmt, *.seq)"

    @staticmethod
    def filefilter() -> str:
        """Returns a string identifying the Windows file extensions for files this class can read.
        The text is suitable for a Windows OpenFile dialog box.

        :return: String containing the Windows file extensions for files this class can read.
        :rtype: str
        """
        return "*.dmt;*.seq"

    @staticmethod
    def isreadable(filename: str | Path = None) -> bool:
        """Determines whether this class is capable of reading a given file.

        :param filename: A filename to check.
        :type filename: str or :class:`pathlib.Path`, optional
        :return: `True` if the file can be read, `False` otherwise.
        :rtype: bool
        """
        if not filename:
            return False

        filename = Path(filename)
        if filename.suffix.lower() not in [".dmt", ".seq"]:
            return False

        # Passed all available tests so suggest we can read this file
        return True

    def __init__(self, filename: str | Path = None):
        """Constructor method

        :param filename: The location of a file to read.
        :type filename: str or :class:`pathlib.Path`, optional
        """
        self._datatype = np.dtype('<f')  # little-endian single-precision float
        self._tile_or_mosaic: Optional[AgilentIRFile.TileOrMosaic] = None
        self._num_datapoints: Optional[int] = None
        self._first_wavenumber: Optional[float] = None
        self._wavenumber_step: Optional[float] = None
        self._last_wavenumber: Optional[float] = None
        self._wavenumbers: Optional[np.ndarray] = None
        self._num_xtiles: Optional[int] = None
        self._num_ytiles: Optional[int] = None
        self._fpa_size: Optional[int] = None
        self._num_xpixels: Optional[int] = None
        self._num_ypixels: Optional[int] = None
        self._data: Optional[np.ndarray] = None
        self._acquisition_datetime: Optional[datetime] = None
        self._file_has_been_read: bool = False

        if filename:
            self._filename = Path(filename)
            match self._filename.suffix.lower():
                case ".dmt":
                    self._tile_or_mosaic = AgilentIRFile.TileOrMosaic.MOSAIC
                case ".seq":
                    self._tile_or_mosaic = AgilentIRFile.TileOrMosaic.TILE
                case _:
                    raise RuntimeError("Cannot read this file type.")

    def _num_tiles(self) -> None:
        """Determines how many focal plane array tiles are in the x and y directions of the image.
        For single tile images, both x and y will be 1.

        """
        match self._tile_or_mosaic:
            case AgilentIRFile.TileOrMosaic.TILE:
                self._num_xtiles = 1
                self._num_ytiles = 1
            case AgilentIRFile.TileOrMosaic.MOSAIC:

                finished = False
                self._num_xtiles = 0
                while not finished:
                    tile_filename = self._filename.parent / f"{self._filename.stem}_{self._num_xtiles:04d}_0000.dmd"
                    if not tile_filename.is_file():
                        finished = True
                    else:
                        self._num_xtiles += 1

                finished = False
                self._num_ytiles = 0
                while not finished:
                    tile_filename = self._filename.parent / f"{self._filename.stem}_0000_{self._num_ytiles:04d}.dmd"
                    if not tile_filename.is_file():
                        finished = True
                    else:
                        self._num_ytiles += 1

            case _:
                raise RuntimeError("Cannot read this file type.")

    def _get_fpa_size(self) -> None:
        """Determines the number of pixels across the focal plane array detector. These detectors are square.

        """
        match self._tile_or_mosaic:
            case AgilentIRFile.TileOrMosaic.MOSAIC:
                tile_filename = self._filename.parent / (self._filename.stem + "_0000_0000.dmd")
            case AgilentIRFile.TileOrMosaic.TILE:
                tile_filename = self._filename.with_suffix(".dat")
            case _:
                raise RuntimeError("Cannot read this file type.")

        tile_size = tile_filename.stat().st_size
        data = tile_size - (255 * 4)  # remove header
        data = data / self._num_datapoints
        data = data / 4  # sizeof float
        self._fpa_size = int(math.sqrt(data))  # fpa size likely to be 64 or 128 pixels square

    def _read_win_int32(self, binary_file) -> int:
        """Reads 4 bytes from the file, treats then as a 32-bit little-endian integer, and returns that integer.

        :param binary_file: An open binary file stream.
        :return: An integer.
        :rtype: int
        """
        return int.from_bytes(binary_file.read(4), byteorder='little')

    def _read_win_double(self, binary_file) -> float:
        """Reads 8 bytes from the file and converts these to a double precision float.

        :param binary_file: An open binary file stream.
        :return: A double precision float.
        :rtype: float
        """
        return struct.unpack('<d', binary_file.read(8))[0]

    def _get_wavenumbers(self) -> None:
        """Extracts a vector of wavenumber values from the file.

        """
        match self._tile_or_mosaic:
            case AgilentIRFile.TileOrMosaic.MOSAIC:
                filename = self._filename.with_suffix(".dmt")
            case AgilentIRFile.TileOrMosaic.TILE:
                filename = self._filename.with_suffix(".bsp")
            case _:
                raise RuntimeError("Cannot read this file type.")

        with open(filename, "rb") as binary_file:
            binary_file.seek(2228, 0)
            first_wavenumber_index = self._read_win_int32(binary_file)

            binary_file.seek(2236, 0)
            self._num_datapoints = self._read_win_int32(binary_file)

            binary_file.seek(2216, 0)
            self._wavenumber_step = self._read_win_double(binary_file)

            self._wavenumbers = np.arange(1, (first_wavenumber_index + self._num_datapoints))
            self._wavenumbers = self._wavenumbers * self._wavenumber_step
            self._wavenumbers = np.delete(self._wavenumbers, range(0, first_wavenumber_index - 1))

            self._first_wavenumber = self._wavenumbers[0]
            self._last_wavenumber = self._wavenumbers[-1]

    def _get_acquisition_date(self) -> None:
        """Extracts the date and time of spectral acquisition from the file.

        """
        match self._tile_or_mosaic:
            case AgilentIRFile.TileOrMosaic.MOSAIC:
                filename = self._filename.with_suffix(".dmt")
            case AgilentIRFile.TileOrMosaic.TILE:
                filename = self._filename.with_suffix(".bsp")
            case _:
                raise RuntimeError("Cannot read this file type.")

        # read in the whole file (it's small) and regex it for the acquisition date/time
        file_contents = filename.read_bytes()
        regex = re.compile(b"Time Stamp.{44}\w+, (\w+) (\d\d), (\d\d\d\d) (\d\d):(\d\d):(\d\d)")
        matches = re.search(regex, file_contents)

        if matches:
            month = matches.group(1).decode()
            day = matches.group(2).decode()
            year = matches.group(3).decode()
            hour = matches.group(4).decode()
            mins = matches.group(5).decode()
            secs = matches.group(6).decode()

            match month:
                case "January":
                    month = 1
                case "February":
                    month = 2
                case "March":
                    month = 3
                case "April":
                    month = 4
                case "May":
                    month = 5
                case "June":
                    month = 6
                case "July":
                    month = 7
                case "August":
                    month = 8
                case "September":
                    month = 9
                case "October":
                    month = 10
                case "November":
                    month = 11
                case "December":
                    month = 12
                case _:
                    raise RuntimeError("Unrecognised month of acquisition.")

            dateobj = datetime.strptime(f"{year} {month} {day} {hour} {mins} {secs}", "%Y %m %d %H %M %S")
            self._acquisition_datetime = dateobj

    def _read_tile(self, filename: Path) -> np.ndarray:
        """Extracts the spectral intensity values from a given tile, defined by the tile's filename.
        Note that the returned :class:`numpy.ndarray` is memory mapped, rather than being read into RAM.

        :param filename: The filename of a spectral tile file.
        :type filename: :class:`pathlib.Path`
        :return: A memory mapped :class:`numpy.ndarray` containing the spectral intensities as a 3D object.
        :rtype: :class:`numpy.ndarray`
        """
        tile = np.memmap(str(filename), dtype=self._datatype, mode='r')
        # skip the (unknown) header
        tile = tile[255:]
        tile = np.reshape(tile, (self._num_datapoints, self._fpa_size, self._fpa_size))
        return tile

    def read(self, filename: str | Path = None) -> None:
        """Opens the file at location `filename`.
        If a filename was not provided when the object was created, it must be provided here.
        The same object can be reused by providing a new filename. In this case the object is reinitialised with this
        new file.

        :param filename: The location of a file to read.
        :type filename: str or :class:`pathlib.Path`, optional
        """
        if not AgilentIRFile.isreadable(filename):
            raise RuntimeError("Cannot read this file.")

        # Reset this object
        self.__init__(filename)

        # Read the .dmt/.seq file to get the wavenumbers and date of acquisition
        self._get_wavenumbers()
        self._get_acquisition_date()

        # Determine how many tiles there are in x- and y- dimensions
        self._num_tiles()
        self._get_fpa_size()
        self._num_xpixels = int(self._fpa_size * self._num_xtiles)
        self._num_ypixels = int(self._fpa_size * self._num_ytiles)

        # Make space for the data, but note the tiles are np.memmap'ed
        self._data = np.empty((self._num_datapoints, self._num_ypixels, self._num_xpixels), dtype=self._datatype)

        match self._tile_or_mosaic:
            case AgilentIRFile.TileOrMosaic.TILE:
                tilefilename = self._filename.with_suffix(".dat")
                self._data = self._read_tile(tilefilename)
            case AgilentIRFile.TileOrMosaic.MOSAIC:

                ystart = 0
                for y in reversed(range(self._num_ytiles)):
                    ystop = ystart + self._fpa_size

                    xstart = 0
                    for x in (range(self._num_xtiles)):
                        xstop = xstart + self._fpa_size

                        tilefilename = self._filename.parent / f"{self._filename.stem}_{x:04d}_{y:04d}.dmd"
                        tile = self._read_tile(tilefilename)
                        self._data[:, ystart:ystop, xstart:xstop] = tile

                        xstart = xstop
                    ystart = ystop

            case _:
                raise RuntimeError("Cannot read this file type.")

        self._data = np.fliplr(self._data)
        self._file_has_been_read = True

    @property
    def metadata(self) -> dict:
        """Returns a `dict` containing metadata relating to the file.
        The keys of the dict are:

            'filename': absolute path to the file that was processed.
            'xpixels': number of pixels in the x direction (image width).
            'ypixels': number of pixels in the y direction (image height).
            'xtiles': number of focal plane array tiles in the x direction.
            'ytiles': number of focal plane array tiles in the y direction.
            'numpts': number of spectral data points.
            'fpasize': width (and height) of the focal plane array detector in pixels.
            'firstwavenumber': lowest wavenumber recorded.
            'lastwavenumber': highest wavenumber recorded.
            'xlabel': a label that can be used for the x-axis of a spectral plot ('wavenumbers (cm-1)').
            'ylabel': a label that can be used for the y-axis of a spectral plot ('absorbance').
            'acqdatetime': date and time of data acqusition in ISO 8601 format (YYYY-MM-DDTHH:mm:ss)

        :return: A dict of parameters extracted from the file
        :rtype: dict
        """
        if not self._file_has_been_read:
            self.read(self._filename)

        meta = {'filename': self._filename,
                'xpixels': self._num_xpixels,
                'ypixels': self._num_ypixels,
                'xtiles': self._num_xtiles,
                'ytiles': self._num_ytiles,
                'numpts': self._num_datapoints,
                'fpasize': self._fpa_size,
                'firstwavenumber': self._first_wavenumber,
                'lastwavenumber': self._last_wavenumber,
                'xlabel': 'wavenumbers (cm-1)',
                'ylabel': 'absorbance',
                'acqdatetime': self._acquisition_datetime.isoformat()
                }
        return meta

    @property
    def intensities(self) -> np.ndarray:
        """Returns a :class:`numpy.ndarray` of `float` containing the spectral intensities of the image.
        The shape of the ndarray is (number_of_data_points, height_of_image_in_pixels, width_of_image_in_pixels)

        :return: A :class:`numpy.ndarray` of spectral intensities.
        :rtype: :class:`numpy.ndarray`
        """
        if not self._file_has_been_read:
            self.read(self._filename)
        return self._data

    @property
    def xvalues(self) -> np.ndarray:
        """Returns a :class:`numpy.ndarray` of `float` containing the wavenumber values.

        :return: A :class:`numpy.ndarray` of wavenumber values.
        :rtype: :class:`numpy.ndarray`
        """
        if not self._file_has_been_read:
            self.read(self._filename)
        return self._wavenumbers

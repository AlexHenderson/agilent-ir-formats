"""
    Agilent File Format Handling for Infrared Spectroscopy
    Copyright (c) 2018-2024  Alex Henderson <alex.henderson@manchester.ac.uk>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the MIT licence.

    For the latest version visit https://github.com/AlexHenderson/agilent-ir-formats/

"""

__version__ = "0.4.0"

from datetime import datetime
from enum import Enum
import math
from pathlib import Path
import re
import struct
from typing import Optional

import h5py
import numpy as np


class AgilentIRFile:
    """Class to open, read and export the contents of an Agilent Fourier Transform Infrared (FTIR) microscopy file.

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

    """

    class TileOrMosaic(Enum):
        """Nested Enumeration class to hold flags for the types of data the outer class can cope with.

        """
        TILE = 1    # Single tile images from a focal plane array experiment.
        MOSAIC = 2  # Multiple single tile images arranged into a mosaic.

    @staticmethod
    def filetype() -> str:
        """Return a string identifying the type of files this class reads.

        The text is suitable for a Windows OpenFile dialog box.

        :return: String describing the file types this class can read.
        :rtype: str
        """
        return "Agilent FTIR Files (*.dmt, *.seq)"

    @staticmethod
    def filefilter() -> str:
        """Return a string identifying the Windows file extensions for files this class can read.

        The text is suitable for a Windows OpenFile dialog box.

        :return: String containing the Windows file extensions for files this class can read.
        :rtype: str
        """
        return "*.dmt;*.seq"

    @staticmethod
    def find_files(search_location: str | Path = ".", recursive: bool = True) -> list[str]:
        """Return a list of Agilent IR image files in a search location.

        `search_location` is searched for *.dmt and *.seq files.

        If `recursive is `True` (default) all paths below the `search_location` will be searched. Otherwise, only the
        `search_location` directory will be searched.

        Discovered files are checked to see if they are readable, and discarded if not.

        :param search_location: Directory to act as starting point for tree search.
        :type search_location: str or :class:`pathlib.Path`, optional
        :param recursive: Whether a recursive search is required.
        :type recursive: bool, optional
        :return: list of discovered files.
        :rtype: list[str]
        :raises RuntimeError: Raised if the `search_location` is not a directory.
        """

        search_location = Path(search_location)
        if not search_location.is_dir():
            raise RuntimeError("search_location should be a directory.")

        if recursive:
            mosaicdmtfiles = list(search_location.rglob("*.dmt"))
        else:
            mosaicdmtfiles = list(search_location.glob("*.dmt"))
        mosaicdmtfiles = list(map(Path.resolve, mosaicdmtfiles))
        mosaicdmtfiles = list(map(Path.as_posix, mosaicdmtfiles))

        if recursive:
            singletileseqfiles = list(search_location.rglob("*.seq"))
        else:
            singletileseqfiles = list(search_location.glob("*.seq"))
        singletileseqfiles = list(map(Path.resolve, singletileseqfiles))
        singletileseqfiles = list(map(Path.as_posix, singletileseqfiles))

        foundfiles = list()
        for file in mosaicdmtfiles:
            if AgilentIRFile.isreadable(file):
                foundfiles.append(file)
        for file in singletileseqfiles:
            if AgilentIRFile.isreadable(file):
                foundfiles.append(file)

        return foundfiles

    @staticmethod
    def isreadable(filename: str | Path = None) -> bool:
        """Determine whether this class is capable of reading a given file.

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

        # Look inside the file
        # If we have a mosaic, the .dmt file will have the words "Mosaic Tiles X" inside
        if filename.suffix.lower() == ".dmt":
            file_contents = filename.read_bytes()
            regex = re.compile(b"Mosaic Tiles X")
            matches = re.search(regex, file_contents)
            if not matches:
                return False

        # See if there is a .bsp file with the same name as the .seq file.
        # If so, does it contain the words "Phase Apodization"
        if filename.suffix.lower() == ".seq":
            bspfilename = filename.with_suffix(".bsp")
            if not bspfilename.is_file():
                return False
            else:
                file_contents = bspfilename.read_bytes()
                regex = re.compile(b"Phase Apodization")
                matches = re.search(regex, file_contents)
                if not matches:
                    return False

        # Passed all available tests so suggest we can read this file
        return True

    @staticmethod
    def version() -> str:
        """Return the version number of this code.

        See https://semver.org/ for more information.

        :return: Version number of this code.
        :rtype: str
        """
        return __version__

    def __init__(self, filename: str | Path = None):
        """Constructor method.

        A filename is required either here, or as a parameter to the `read` method.

        :param filename: The location of a file to read.
        :type filename: str or :class:`pathlib.Path`, optional
        :raises RuntimeError: Raised if the file is of a format that cannot be read.
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
        self._totalimage: Optional[np.ndarray] = None
        self._totalspectrum: Optional[np.ndarray] = None
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

    def __repr__(self) -> str:
        """Generate a string representation of this object.

        :return: String representation of this object.
        :rtype: str
        """
        return f"Agilent FTIR image. Filename: {self._filename}."

    def __str__(self) -> str:
        """Generate a string representation of this object.

        :return: String representation of this object.
        :rtype: str
        """
        return f"Agilent FTIR image. Filename: {self._filename}."

    def _num_tiles(self) -> None:
        """Determine how many focal plane array tiles are in the x and y directions of the image.

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
        """Determine the number of pixels across the focal plane array detector.

        These detectors are square, so this value is both the height and width of the focal plane array detector.
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
        """Read 4 bytes from the file, treat them as a 32-bit little-endian integer, and return that integer.

        :param binary_file: An open binary file stream.
        :return: An integer.
        :rtype: int
        """
        return int.from_bytes(binary_file.read(4), byteorder='little')

    def _read_win_double(self, binary_file) -> float:
        """Read 8 bytes from the file, convert these to a double precision float and return the value.

        :param binary_file: An open binary file stream.
        :return: A double precision float.
        :rtype: float
        """
        return struct.unpack('<d', binary_file.read(8))[0]

    def _get_wavenumbers(self) -> None:
        """Extract a vector of wavenumber values from the file.

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
        """Extract the date and time of spectral acquisition from the file.

        """
        match self._tile_or_mosaic:
            case AgilentIRFile.TileOrMosaic.MOSAIC:
                filename = self._filename.with_suffix(".dmt")
            case AgilentIRFile.TileOrMosaic.TILE:
                filename = self._filename.with_suffix(".bsp")
            case _:
                raise RuntimeError("Cannot read this file type.")

        # Read in the whole file (it's small) and regex it for the acquisition date/time.
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
        """Extract the spectral intensity values from a given tile, defined by the tile's filename.

        Note the returned :class:`numpy.ndarray` is memory mapped, rather than being read into memory.

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
        """Open the file at location `filename`.

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
        # Convert data to row-major orientation. Slowest changing dimension first, fastest changing dim last.
        # Data is in this orientation: (y, x, channels)
        self._data = np.transpose(self._data, (1, 2, 0))

        # Calculate some summaries for ease of access
        if (self._tile_or_mosaic == AgilentIRFile.TileOrMosaic.MOSAIC) or \
           (self._tile_or_mosaic == AgilentIRFile.TileOrMosaic.TILE):
            self._totalimage = np.sum(self._data, axis=2)

        if (self._tile_or_mosaic == AgilentIRFile.TileOrMosaic.MOSAIC) or \
           (self._tile_or_mosaic == AgilentIRFile.TileOrMosaic.TILE):
            self._totalspectrum = np.sum(np.sum(self._data, axis=0), axis=0)
        else:
            self._totalspectrum = self._data

        self._file_has_been_read = True

    def _generate_hdf5_metadata(self) -> dict[str, any]:
        """Return a `dict` containing metadata parameters that can be used for the HDF5 file format.

        These metadata terms are designed to suitable for multiple analytical techniques.
        The keys of the dict are:
            '/metadata/generator/name': name of this class.
            '/metadata/generator/url': online location of this class.
            '/metadata/generator/version': version number of this code. See https://semver.org/ for more information.

            '/metadata/source/filename': name of the source file.

            '/metadata/data/datapoints': number of spectral data points.
            '/metadata/data/height': height of the image in pixels.
            '/metadata/data/width': width of the image in pixels.
            '/metadata/data/image_origin': location of the image origin. For example, 'upper left'.
            '/metadata/data/shape': shape of the data object.
            '/metadata/data/shape_interpretation': interpretation of the axes of the data object.

            '/metadata/plotting/xlabelname': physical quantity of the spectral x-axis dimension.
            '/metadata/plotting/xlabelunit': unit of the spectral x-axis dimension.
            '/metadata/plotting/xlabel': label suitable for the x-axis of a spectral plot.
            '/metadata/plotting/ylabelname': physical quantity of the spectral y-axis dimension.
            '/metadata/plotting/ylabelunit': unit of the spectral y-axis dimension.
            '/metadata/plotting/ylabel': label suitable for the y-axis of a spectral plot.
            '/metadata/plotting/plot_high2low': whether it is appropriate to plot the x-axis from high to low.

            '/metadata/experiment/first_xvalue': lowest wavenumber.
            '/metadata/experiment/last_xvalue': highest wavenumber.

            '/metadata/semantics/technique/accuracy_of_term': how accurate are the semantics of this section.
            '/metadata/semantics/technique/term': ontological name of this analysis technique.
            '/metadata/semantics/technique/description': ontological description of this analysis technique.
            '/metadata/semantics/technique/uri': ontological uri of this analysis technique.

        :return: Metadata parameters suitable for HDF5.
        :rtype: dict
        """
        if not self._file_has_been_read:
            self.read(self._filename)

        # /metadata/generator
        h5metadata = dict()
        h5metadata['/metadata/generator/name'] = 'agilent-ir-formats'
        h5metadata['/metadata/generator/url'] = 'https://github.com/AlexHenderson/agilent-ir-formats'
        h5metadata['/metadata/generator/version'] = __version__

        # /metadata/source
        h5metadata['/metadata/source/filename'] = str(self._filename.name)

        # /metadata/data
        h5metadata['/metadata/data/datapoints'] = self._num_datapoints
        if (self._tile_or_mosaic == AgilentIRFile.TileOrMosaic.MOSAIC) or \
           (self._tile_or_mosaic == AgilentIRFile.TileOrMosaic.TILE):
            h5metadata['/metadata/data/height'] = self._num_ypixels
            h5metadata['/metadata/data/width'] = self._num_xpixels
            h5metadata['/metadata/data/image_origin'] = 'upper left'
            h5metadata['/metadata/data/shape'] = self._data.shape
            h5metadata['/metadata/data/shape_interpretation'] = ["height", "width", "datapoints"]

        # /metadata/plotting
        h5metadata['/metadata/plotting/xlabelname'] = "wavenumber"
        h5metadata['/metadata/plotting/xlabelunit'] = "cm^-1"
        h5metadata['/metadata/plotting/xlabel'] = \
            f"{h5metadata['/metadata/plotting/xlabelname']} ({h5metadata['/metadata/plotting/xlabelunit']})"
        h5metadata['/metadata/plotting/ylabelname'] = "absorbance"
        h5metadata['/metadata/plotting/ylabelunit'] = ""
        h5metadata['/metadata/plotting/ylabel'] = h5metadata['/metadata/plotting/ylabelname']
        h5metadata['/metadata/plotting/plot_high2low'] = True

        # /metadata/experiment
        h5metadata['/metadata/experiment/first_xvalue'] = self._first_wavenumber,
        h5metadata['/metadata/experiment/last_xvalue'] = self._last_wavenumber,

        # /metadata/semantics
        h5metadata['/metadata/semantics/technique/accuracy_of_term'] = 'http://www.w3.org/2004/02/skos/core#exactMatch'
        h5metadata['/metadata/semantics/technique/term'] = 'Fourier transform infrared microscopy'
        h5metadata['/metadata/semantics/technique/description'] = \
            'The collection of spatially resolved infrared spectra of a sample during optical microscopy. ' \
            'The infrared spectra are obtained by single pulse of infrared radiation, ' \
            'and are subject to a Fourier transform.'
        h5metadata['/metadata/semantics/technique/uri'] = 'http://purl.obolibrary.org/obo/CHMO_0000051'

        return h5metadata

    def export_hdf5(self, filename: str | Path = None):
        """Write a version of the file to disc in HDF5 format.

        If `filename` is `None`, the source file's name is used, swapping the .dmt/.seq extension with .h5.

        The data is both chunked and compressed. The total intensity spectrum and total intensity image
        (where appropriate) are also exported. A range of associated metadata is also included.

        A free viewer for HDF5 files is available at https://www.hdfgroup.org/downloads/hdfview/.

        :param filename: The name of the target HDF5 file.
        :type filename: str or :class:`pathlib.Path`, optional
        """
        if not self._file_has_been_read:
            self.read(self._filename)

        if filename is None:
            filename = self._filename.with_suffix(".h5")

        # Default parameters for HDF5 export
        h5parameters = dict()
        h5parameters['extn'] = '.h5'
        h5parameters['chunkwidth'] = 64
        h5parameters['chunkheight'] = 64
        h5parameters['chunkxaxis'] = 100
        h5parameters['deflate'] = 4
        h5parameters['shuffle'] = True

        chunkwidth = min(h5parameters['chunkwidth'], self._num_xpixels)
        chunkheight = min(h5parameters['chunkheight'], self._num_ypixels)
        chunkxaxis = min(h5parameters['chunkxaxis'], self._num_datapoints)

        with h5py.File(filename, "w") as h5file:
            # Write the intensity values
            ds = h5file.create_dataset("/data/intensities", self._data.shape, self._data.dtype,
                                       chunks=(chunkheight, chunkwidth, chunkxaxis),
                                       compression="gzip", compression_opts=h5parameters['deflate'],
                                       shuffle=h5parameters['shuffle'])
            ds[:] = self._data

            # Write the wavenumber values
            ds = h5file.create_dataset("/data/xvalues", self._wavenumbers.shape, self._wavenumbers.dtype,
                                       chunks=(chunkxaxis,),
                                       compression="gzip", compression_opts=h5parameters['deflate'],
                                       shuffle=h5parameters['shuffle'])
            ds[:] = self._wavenumbers

            # Write the total signal spectrum intensity values
            ds = h5file.create_dataset("/data/totalspectrum", self._totalspectrum.shape, self._totalspectrum.dtype,
                                       chunks=(chunkxaxis,),
                                       compression="gzip", compression_opts=h5parameters['deflate'],
                                       shuffle=h5parameters['shuffle'])
            ds[:] = self._totalspectrum

            if self._totalimage is not None:
                # Write the total image intensity values
                # No point chunking, but required for compression
                ds = h5file.create_dataset("/data/totalimage", self._totalimage.shape, self._totalimage.dtype,
                                           chunks=(chunkheight, chunkwidth),
                                           compression="gzip", compression_opts=h5parameters['deflate'],
                                           shuffle=h5parameters['shuffle'])
                ds[:] = self._totalimage

            meta = self._generate_hdf5_metadata()
            for k in meta:
                h5file[k] = meta[k]

    @property
    def intensities(self) -> np.ndarray:
        """Return a 3D :class:`numpy.ndarray` of `float` containing the spectral intensities of the image.

        The shape of the ndarray is (height_of_image_in_pixels, width_of_image_in_pixels, number_of_data_points).

        :return: A :class:`numpy.ndarray` of spectral intensities.
        :rtype: :class:`numpy.ndarray`
        """
        if not self._file_has_been_read:
            self.read(self._filename)
        return self._data

    @property
    def wavenumbers(self) -> np.ndarray:
        """Return a 1D :class:`numpy.ndarray` of `float` containing the wavenumber values.

        :return: A :class:`numpy.ndarray` of wavenumber values.
        :rtype: :class:`numpy.ndarray`
        """
        if not self._file_has_been_read:
            self.read(self._filename)
        return self._wavenumbers

    @property
    def total_image(self) -> np.ndarray:
        """Return a 2D :class:`numpy.ndarray` of `float` containing the sum of spectral intensities at each pixel.

        The shape of the ndarray is (height_of_image_in_pixels, width_of_image_in_pixels).

        :return: A :class:`numpy.ndarray` of total spectral intensities.
        :rtype: :class:`numpy.ndarray`
        """
        if not self._file_has_been_read:
            self.read(self._filename)
        return self._totalimage

    @property
    def total_spectrum(self) -> np.ndarray:
        """Return a 1D :class:`numpy.ndarray` of `float` containing the sum of spectral intensities of all pixels.

        :return: A :class:`numpy.ndarray` of total spectral intensities.
        :rtype: :class:`numpy.ndarray`
        """
        if not self._file_has_been_read:
            self.read(self._filename)
        return self._totalspectrum

    @property
    def metadata(self) -> dict:
        """Return a `dict` containing metadata relating to the file.

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
            'acqdatetime': date and time of data acquisition in ISO 8601 format (YYYY-MM-DDTHH:mm:ss)

        :return: A dict of parameters extracted from the file.
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
                'xlabel': 'wavenumber (cm^-1)',
                'ylabel': 'absorbance',
                'acqdatetime': self._acquisition_datetime.isoformat(),
                'datashape': "(y, x, wavenumbers)"
                }
        return meta

    @property
    def hdf5_metadata(self):
        """Return a `dict` containing HDF5-specific metadata relating to the file.

        These metadata are used when exporting to HDF5.
        A free viewer for HDF5 files is available at https://www.hdfgroup.org/downloads/hdfview/.

        :return: A dict of parameters extracted from the file.
        :rtype: dict
        """
        return self._generate_hdf5_metadata()

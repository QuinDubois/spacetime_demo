from osgeo import gdal
from osgeo import osr
import numpy as np
import netCDF4 as nc


class cube_meta(object):

    def __init__(self, dataList):

        # Initialize an emty storage matrix
        objMat = [None] * 4

        # main data set opened in gdal
        objMat[0] = dataList

        # get projection data
        objMat[1] = osr.SpatialReference(wkt=objMat[0].GetProjection())

        # extract geo transform from gdal object
        objMat[2]=objMat[0].GetGeoTransform()

        # set time vector
        max = objMat[0].RasterCount
        min = 0

        objMat[3] = np.arange(min, max, 1)

        # save as barracuda object
        self.cubeObj = objMat


    # returns a list of gdal or netcdf4 objects
    def extract_original_data(self):

        return self.cubeObj[0]

    def spatial_reference(self):
        out = self.cubeObj[0].GetProjection()
        return out


    # returns a list of SRS codes for each raster
    def extract_epsg_code(self):

        code = self.cubeObj[1].GetAttrValue('AUTHORITY',1)
        epsg = ("EPSG:" + str(code))

        return epsg


    def extract_units(self):


        unit = self.cubeObj[1].GetAttrValue('UNIT',0)

        return unit



    def upper_left_corner(self):


        v = self.cubeObj[2]
        corner = [v[i] for i in [3,0]]

        return corner



    def pixel_size(self):

        v = self.cubeObj[2]
        size = v[1]

        return size


    def number_of_bands(self):


        bands = self.cubeObj[0].RasterCount

        return bands


    def extract_time(self):

        return self.cubeObj[3]

    def get_raster_dimensions(self):

        xDim = self.cubeObj[0].RasterXSize
        yDim = self.cubeObj[0].RasterYSize

        dims = tuple([xDim, yDim])

        return dims


    def get_data_range(self):

        band = (self.cubeObj[0].GetRasterBand(1))
        max = band.GetMaximum()
        min = band.GetMinimum()

        range = tuple([min, max])

        return range

    def get_nodata_value(self):

        band = (self.cubeObj[0].GetRasterBand(1))
        noDat = band.GetNoDataValue()

        return noDat


    def get_latitude(self):

        # pixel size
        ysize = -self.pixel_size()

        # dimensions
        height = self.get_raster_dimensions()[1]

        # upper left corner
        y = self.upper_left_corner()[0]

        # dimensions from 0 to max dims of dataset
        my=np.arange(start=0, stop=height)

        # get lats
        latVec = np.multiply(my, ysize) + y # latitude vector

        return latVec


    def get_longitude(self):

        # pixel size
        xsize = self.pixel_size()

        # dimensions
        width = self.get_raster_dimensions()[0]

        # upper left corner
        x = self.upper_left_corner()[1]

        # dimensions from 0 to max dims of dataset
        mx=np.arange(start=0, stop=width)

        # get lats
        longVec = np.multiply(mx, xsize) + x # latitude vector

        return longVec





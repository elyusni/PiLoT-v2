import numpy as np
from osgeo import gdal
from pathlib import Path
import time
import rasterio

def read_DSM_config(ref_dsm, ref_dom, npy_save_path):

    with rasterio.open(ref_dsm) as dsm_ds:
        dsm_data = dsm_ds.read(1)
        dsm_transform = dsm_ds.transform

    with rasterio.open(ref_dom) as dom_ds:
        dom_data = dom_ds.read([1, 2, 3])

    start = time.time()
    dataset = gdal.Open(ref_dsm)
    band = dataset.GetRasterBand(1)
    geotransform = dataset.GetGeoTransform()

    if Path(npy_save_path).exists():
        area = np.load(npy_save_path)
    else:
        area = band.ReadAsArray()
        np.save(npy_save_path, area)

    valid_mask = (area > 0) & (area < 10000)

    area_minZ = np.median(area[valid_mask])

    print("area_minZ (valid minimum elevation):", area_minZ)

    return geotransform, area, area_minZ, dsm_data, dsm_transform,dom_data

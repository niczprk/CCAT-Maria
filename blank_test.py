import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import os

size_deg = 10.0
pix_arcmin = 1.0
ra0_deg = 200.0
dec0_deg = -30.0

outdir_pers= "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/blank_ccat_test_outputs"

outdir_pro="/Users/zaparniukn/Documents/maria/blank_ccat_test_outputs"

outdir = outdir_pro

# Create a blank FITS file

naxis1 = int(size_deg * 60 / pix_arcmin)  # Number of pixels along RA
naxis2 = int(size_deg * 60 / pix_arcmin)  # Number of pixels along Dec

wcs = WCS(naxis=2)
wcs.wcs.crpix = [naxis1 / 2, naxis2 / 2]
wcs.wcs.cdelt = np.array([-pix_arcmin / 60, pix_arcmin / 60])  # degrees per pixel
wcs.wcs.crval = [ra0_deg, dec0_deg]  # Reference coordinates (RA, Dec)
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

fits.PrimaryHDU(data=np.zeros((naxis2, naxis1)), header=wcs.to_header()).writeto(
    os.path.join(outdir, "blank_map_TAN.fits"), overwrite=True
)
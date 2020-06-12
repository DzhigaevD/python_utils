# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


import numpy as np
from scipy.fftpack import ifftshift, fftshift, fft2, fftn
import matplotlib.pyplot as plt

# NB: It is possible to use the PYNX_PU environment variable to choose the GPU or language,
# e.g. using PYNX_PU=opencl or PYNX_PU=cuda or PYNX_PU=cpu or PYNX_PU=Titan, etc..

from pynx.utils.pattern import siemens_star
from pynx.cdi import *
from PIL import Image

npzfile = np.load('/media/dzhigd/OS/WORK_DIRECTORY/dzhigd/CurrentProject/FIB_STO_BCDI/APS/data/Sample3_gold_NP/Sample3__165/data_61_256_256.npz')
data = npzfile['amp']

support = np.absolute(ifftshift(fftn(ifftshift(data))))
support = support/np.max(np.max(np.max(support)))
support[support<0.1] = 0;
support[support>0] = 1;

#plt.imshow(support[30,:,:])
#
#plt.show()

#print(data.shape)
#print(data.dtype)

cdi = CDI(fftshift(data), obj=None, support=fftshift(support), mask=None, wavelength=0.113e-9,
          pixel_size_detector=55e-10)

cdi.init_free_pixels()

# Initial scaling of the object [ only useful if there are masked pixels !]
#cdi = ScaleObj(method='F') * cdi

show = 40

# Do 200 cycles of HIO, displaying object every N cycle and log-likelihood every 20 cycle
cdi = HIO(calc_llk=20, show_cdi=show, fig_num=2) ** 200 * cdi

# Support update operator
sup = SupportUpdate(threshold_relative=0.25, smooth_width=(5, 0.5, 800), force_shrink=False)

if True:
    # Do 40 cycles of HIO, then 5 of ER, update support, repeat
    cdi = (sup * ER(calc_llk=20, show_cdi=show, fig_num=2) ** 5
           * HIO(calc_llk=20, show_cdi=show, fig_num=2) ** 40) ** 20 * cdi
else:
    # Do 40 cycles of HIO, update support, repeat
    cdi = (sup * HIO(calc_llk=20, show_cdi=show, fig_num=2) ** 40) ** 20 * cdi

# Finish with ML or ER
cdi = ML(reg_fac=1e-2, calc_llk=20, show_cdi=show, fig_num=1) ** 100 * cdi
#cdi = ER(calc_llk=20, show_cdi=show, fig_num=2) ** 100 * cdi

# Obtain real space object
plt.figure(4)
obj_real = CDI.get_obj(cdi)
plt.imshow(abs(obj_real[30,:,:]))


# Saving procedures

#CDI.save_obj_cxi(cdi,filename = 'test.cxi' )

#cdi = ShowCDI(fig_num=1) * cdi


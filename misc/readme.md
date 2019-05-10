To show Us and Ground truth images from h5 files.
Let's show 100th slice.

```
import numpy as np
import h5py
from PIL import Image

h5f = h5py.File("us_gt_vol.h5", "r")
us = h5f["us_vol"][:]
us_img = Image.fromarray(us[:,100,:])
us_img.show()

gt = h5f["gt_vol"][:]
gt_img = Image.fromarray(np.uint8(np.transpopse(gt, (1,0,2))[:,100,:]*255))
gt_img.show()
```

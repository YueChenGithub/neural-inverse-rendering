import torch
from pathlib import Path
from PIL import Image
from tools.convert_img_channel import rgb_change_background
import glob

# dir = 'home/yue/Desktop/ma2/tools/png2jpg'
# subfolders = sorted(glob.glob(dir + "/*/"))

# for subfolder in subfolders:
#     print(subfolder)

subfolder = '/home/yue/Desktop/ma2/tools/png2jpg/'

# for name in ['DiffCol_0001', 'DiffDirCol_0001', 'DiffIndCol_0001', 'GlossDirCol_0001', 'GlossIndCol_0001', 'Normal_0001']:
for name in ['GlossDirCol_0001', 'GlossIndCol_0001']:
    rgb = Image.open(Path(subfolder, name + '.png')).convert("RGB")
    rgba_ref = Image.open(Path(subfolder,  'rgba.png'))
    rgb_change_background(rgb, rgba_ref, background_origin=0, background_new=1).save(Path(subfolder, name[:-5] + '.png'))

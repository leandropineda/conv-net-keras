from scipy import misc
from skimage import filtes

def binarizeImage(imfile,dest_path):
    destfile=dest_path+imfile[imfile.find("1000")+5:]
    im=misc.imread(f)
    imbin = filters.threshold_otsu(im)
    misc.imsave(imbin,destfile)
    error(0)

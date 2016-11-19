from scipy import misc
from skimage import filters
import matplotlib.pyplot as plt

def binarizeImage(imfile, dest_path):
    destfile = imfile.replace('data_dest_dir', 'dataset_preproc')
    im=misc.imread(imfile,flatten=True)
    #plt.show()
    try:
        th=filters.threshold_otsu(im)
    except:
        th=0
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            if im[x][y]<th:
                im[x][y]=0
            else:
                im[x][y]=255
    misc.imsave(destfile,im)


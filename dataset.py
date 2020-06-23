import os
import pydicom as dicom
import random
import glob

from chainer.dataset import dataset_mixin
import numpy as np
from chainercv.transforms import random_crop,center_crop,resize

class prjData(dataset_mixin.DatasetMixin):
    def __init__(self, path, osem=1):
        self.ids = []
        self.rev = []
        self.patient_id = []
        self.slice = []
        self.osem = osem
        self.W = 800 # size of the projection image
        self.H = 528
        print("Load sinograms from: {}".format(path))
        for fn in glob.glob(os.path.join(path,"**/*.npy"), recursive=True):
            self.ids.append(fn)
            self.rev.append("_rev" in fn)
            filename = os.path.basename(fn)
            self.patient_id.append(int(filename[2:5]))
            self.slice.append(int(filename[7:10]))
        print("Loaded: {} images".format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        img = np.load(self.ids[i])
        imgs = np.stack([(img.reshape(self.W,self.H)[i::self.osem,:]).reshape(-1,1) for i in range(self.osem)])
        return imgs, self.rev[i], self.patient_id[i], self.slice[i]

## load images everytime from disk: slower but low memory usage
class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, path, baseA=-500, rangeA=700, crop=(256,256), scale_to=-1, random=0, imgtype="dcm", dtype=np.float32):
        self.path = path
        self.base = baseA
        self.range = rangeA
        self.ids = []
        self.random = random
        self.crop = crop
        self.scale_to = scale_to
        self.ch = 1
        self.dtype = dtype
        self.imgtype=imgtype
        print("Load training dataset for discriminator from: {}".format(path))
        for file in glob.glob(os.path.join(self.path,"**/*.{}".format(imgtype)), recursive=True):
            fn, ext = os.path.splitext(file)
            self.ids.append(fn)
        if len(self.ids)==0:
            self.ids.append('dummy')
        print("Loaded: {} images".format(len(self.ids)))
        
    def __len__(self):
        return len(self.ids)

    def get_img_path(self, i):
        return '{:s}.{}'.format(self.ids[i],self.imgtype)

    def img2var(self,img):
#        return(0.0755*img/1020.0 + 0.0755)
        return(2*(np.clip(img,self.base,self.base+self.range)-self.base)/self.range-1.0).astype(self.dtype)

    def var2img(self,var):
#        return(1020.0*(var-0.0755)/0.0755)
        return(0.5*(1.0+var)*self.range + self.base)

    def get_example(self, i):
        if self.imgtype == "npy":
            img = np.load(self.get_img_path(i))
            img = 2*(np.clip(img,self.base,self.base+self.range)-self.base)/self.range-1.0
            if len(img.shape) == 2:
                img = img[np.newaxis,]
        else:
            ref_dicom = dicom.read_file(self.get_img_path(i), force=True)
    #        print(ref_dicom)
    #        ref_dicom.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
            img = ref_dicom.pixel_array+ref_dicom.RescaleIntercept
            img = self.img2var(img)
            img = img[np.newaxis,:,:]
        if self.scale_to>0:
            img = resize(img,(self.scale_to,self.scale_to))
        H,W = self.crop
#        print(img.shape)
        if img.shape[1]<H+2*self.random or img.shape[2] < W+2*self.random:
            p = max(H+2*self.random-img.shape[1],W+2*self.random-img.shape[2])
            img = np.pad(img,((0,0),(p,p),(p,p)),'edge')
        if H+self.random < img.shape[1] and W+self.random < img.shape[2]:
            img = center_crop(img,(H+self.random, W+self.random))
            img = random_crop(img,self.crop)
        return img

def write_dicom(fn,new):
    # prepare template dcm for saving
    dummy_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),"dummy.dcm")
    ref_dicom = dicom.dcmread(dummy_file, force=True)
    ref_dicom.file_meta.TransferSyntaxUID = dicom.uid.ExplicitVRLittleEndian #dicom.uid.ImplicitVRLittleEndian
    ref_dicom.is_little_endian = True
    ref_dicom.is_implicit_VR = False

#    img = np.full(ref_dicom.pixel_array.shape, -1000, dtype=np.float32)
#    ch,cw = img.shape
#    h,w = new.shape
#    img[(ch-h)//2:(ch+h)//2,(cw-w)//2:(cw+w)//2] = new
    img = (new - ref_dicom.RescaleIntercept).astype(ref_dicom.pixel_array.dtype)           
    ref_dicom.PixelData = img.tostring()
    ref_dicom.Rows, ref_dicom.Columns = img.shape
    uid = ref_dicom[0x20,0x52].value.split(".")  # Frame of Reference UID
    uid[-1] = "1213"
    uidn = ".".join(uid)
    ref_dicom[0x20,0x52].value=uidn  # Frame of Reference UID
    ref_dicom.save_as(fn)

import numpy as np
from PIL import Image
from tqdm import tqdm
import datetime
import pickle
import time
import cv2 as cv
class ImagePreprocessor(object):
    """Preprocess images that contains many other areas that are not our interests"""
    def __init__(self, debug=False):
        self.debug = debug

    def setImage(self, img, dst=None):
        """ Convert other format into numpy.ndarray

        Arguments:
            img(str| np.ndarray | PIL.Image.Image | torch.Tensor)
            dst(str): directory that will be used to store processed images
        """
        self.dst = dst
        imtype = type(img)
        if imtype == str:
            self._img = np.asarray(Image.open(img))
        elif imtype == np.ndarray:
            self._img = img
        elif isinstance(img, Image.Image):
            self._img = np.asarray(img)
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3:
                img = img.permute(1, 2, 0)
            ## TODO check img max val ?
            self._img = img.mul(255).byte().cpu().numpy()
        else:
            raise ValueError(f"Type Not Supported: {imtype}")
        
        assert self._img.dtype == np.uint8

        if self.debug:
            print(self._img.shape, self._img.dtype)

    def rotate(self, angle, center=None, scale=1.0, boarderValue=(0,0,0)):
        """
        Note: some problems may not be solved currently
        """
        img = self._img
        h, w = img.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, scale)
        rotated = cv.warpAffine(img, M, (w, h), borderValue=boarderValue)
        return rotated

    def getRoiByConnected(self, thresh, areaSize):
        '''
        Arguments:
            thresh (np.ndarray): preprocessed thresh image
            areaSize(int): any components whose size is less than `areaSize` 
                will be filtered, if None, no components will be filtered
        '''
        img = self._img
        num_labels, labels, stats, centers = cv.connectedComponentsWithStats(thresh, connectivity=8, ltype=cv.CV_32S)
        image = None
        if self.debug:
            image = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            # image = np.asarray(image, dtype=np.float32)
            print(image.shape, image.ndim,image.dtype)
            # print(num_labels)
        roi_list = []
        ## TODO figure out why it starts from 1
        for t in range(1, num_labels, 1):
            x, y, w, h, area = stats[t]
            if areaSize is not None and area < areaSize:
                continue
            # save roi coordinate and width and height
            roi_list.append((x, y, w, h))
            if self.debug:
                cx, cy = centers[t]
                # 标出中心位置
                cv.circle(image, (np.int32(cx), np.int32(cy)), 2, (0, 255, 0), 2, 8, 0)
                # 画出外接矩形
                cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2, 8, 0)
                showImage(image, name="debug components", size=(w, h))
                break

        return num_labels, labels, roi_list, image

    CROP_RETAIN_WH = 0
    CROP_MIN_WH = 1
    CROP_MAX_WH = 2
    def cropByComponents(self, roi = None, \
        thresh=127, maxval=255, 
        dst_size=100000, xoff=0, yoff=0,
        crop_policy=CROP_MIN_WH, areaSize = 24000
        ):
        """
        Arguments:
            dst_size(int): final size of the image if specified
            xoff(int): offset for x
            yoff(int): offset for y

            roi (list(int)): x , y, w, h
                if roi is None: processor will try to find the first area 
                that is connected and it's size is greater than areaSize
            dst(str): image file saved location
            crop_policy(int): if 0,  width and height will be retained, 
                if greater than 0, final width and height will be the same,
                if 1, the width and height will be the smallest one
                if 2, biggest one
        """
        # im = cv.imdecode(np.fromfile(f"{src}", dtype=np.uint8), imread_type) # gray
        im = self._img
        if self._img.ndim == 3 and self._img.shape[-1] != 1:
            gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        else:
            gray = im

        ret, th1 = cv.threshold(gray, thresh, maxval, cv.THRESH_BINARY | cv.THRESH_OTSU )

        # contours, hierarchy = cv.findContours(im, 1, 2)
        # cnt = contours[0]
        # area = cv.contourArea(cnt)
        if roi is None:
            num_labels, labels, roi_list, image = self.getRoiByConnected(th1, areaSize=areaSize)
            x, y, w, h = roi_list[0]
        else:
            x, y, w, h = roi
        # rois.append(roi_list[0])
        if crop_policy == ImagePreprocessor.CROP_MIN_WH:
            mwh = min(w, h, dst_size)
            w = h = mwh
        elif crop_policy == ImagePreprocessor.CROP_MAX_WH:
            mwh = max(w, h, dst_size)
            w = h = mwh
        else:
            raise ValueError("Policy not supported!")
        x += xoff
        y += yoff
        cimage = im[ y:y+h, x:x+w ]

        if self.debug:
            print("ROI: ", x, y, " WH: ", w, h)
            showImage(cimage)

        self.crop_img = cimage
        self.crop_roi = [x, y, w, h] # roi_list[0]
        # return roi_list[0]

    def saveImage(self, img, dst, imsuffix=".jpg"):
        retval, buf = cv.imencode(imsuffix, cimage)
        buf.tofile(dst)

    def findEdge(self, im = None, thresh=127, maxval=255, \
            canny_thresh1=120, canny_thresh2=200, edge_weight= 0.6):
        ''' This is a test function and may change sometimes
        Arguments:
            im(np.ndarray): 
        '''
        # gray = cv.imdecode(np.fromfile(impath, dtype=np.uint8), 0)
        # gray = np.copy(self._img)
        if im is None:
            im = self._img

        if im.ndim == 3 and im.shape[-1] != 1:
            gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        else:
            gray = np.copy(im)
        # gray = cv.GaussianBlur(gray, (3, 3), 1)
        cimg = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT,1, 100,
                                    param1=50,param2=30,minRadius=30,maxRadius=84)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        # ret, th1 = cv.threshold(gray, thresh, maxval, cv.THRESH_BINARY | cv.THRESH_OTSU )
        th1 = cv.adaptiveThreshold(gray, maxval, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 0 )
        # ret, th1 = cv.threshold(th1, thresh, maxval, cv.THRESH_BINARY | cv.THRESH_OTSU )

        # 开闭运算去除噪点
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        th1 = cv.morphologyEx(th1, cv.MORPH_OPEN, kernel)
        th1 = cv.morphologyEx(th1, cv.MORPH_CLOSE, kernel)

        edge = cv.Canny(th1, canny_thresh1, canny_thresh2)
 
        # if self.debug:
        #     dst = cv.addWeighted(gray, (1-edge_weight), edge, edge_weight, 0)
        #     plt.figure(figsize=(10, 8 ), dpi=100)
        #     plt.imshow(dst, cmap='gray')
        #     plt.show()

        return edge, th1, cimg

    # show the colorful components
    def colorImg(self, thresh, num_labels, labels):
        '''
        thresh: 填充后的二值图
        num_labels: 连通组件的个数
        labels: 连通组件的输出标记图像，背景index=0
        '''
        # make the colors
        colors = []
        for i in range(num_labels):
            b = np.random.randint(0, 256)
            g = np.random.randint(0, 256)
            r = np.random.randint(0, 256)
            colors.append((b, g, r))
        colors[0] = (0, 0, 0)

        # draw the image
        h, w = thresh.shape
        image = np.zeros((h, w, 3), dtype=np.uint8)
        for row in range(h):
            for col in range(w):
                image[row, col] = (colors[labels[row, col]])

        return image

class CvImageReader(dict):
    """
    """
    def __init__(self, root, rootId, img_type=0, img_suffix=".png", **kwargs):
        super().__init__()
        self.root = root
        self.rootId = rootId
        self.img_suffix = img_suffix
        self.imgs = list(sorted([img for img in sorted(os.listdir(osp.join(root, rootId))) if img.endswith(self.img_suffix)]))
        self.img_type = img_type
        print("Dataset: ", len(self.imgs))

    def plt_hist(self, idxes, prange=None, density=False, stacked=False):
        """
        Arguments:
            idxes(list(int) | int)
            prange(list): plt range (two elements)
        """
        if prange is None:
            prange = [0, 256]

        if type(idxes) == list:
            imgs = []
            paths = []
            for idx in tqdm(idxes):
                img, path = self[idx]
                # img.append(img.ravel())
                # p = path[path.rfind("\\")+1:]
                # paths.append(p)
                plt.hist(img.ravel(), 256, prange, stacked=stacked, density=density)
            # print(paths, len(imgs))
            # imgs = np.asarray(imgs)
            plt.show()
        elif type(idxes) == int:
            img, path = self[idxes]
            plt.hist(img.ravel(), 256, prange, density=density)
            if not stacked:
                plt.xlabel(path)
        else:
            print("Not Supported!")

    def __getitem__(self, idx):
        if self.rootId is not None:
            path = osp.join(self.root, self.rootId, self.imgs[idx])
        else:
            path = osp.join(self.root, self.imgs[idx])
        return cv.imread(path, self.img_type), path

    def __len__(self):
        return len(self.imgs)

def saveROIOnly(src, dst = None, imread_type = 0):
    """ imread_type: IMREAD_GRAY 0 gray , IMREAD_COLOR
    """
    # src = config.DataDir / "train_raw"
    im = cv.imdecode(np.fromfile(f"{src}", dtype=np.uint8), imread_type) 
    # ret, th1 = cv.threshold(im, 127, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(im,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY, 25, 2)
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    # morphed = cv.morphologyEx(th2, cv.MORPH_CLOSE, kernel)
    # plt.imshow(morphed, 'gray')
    cnts = cv.findContours(th2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv.contourArea)[-1]
    ## (4) Crop and save it
    x,y,w,h = cv.boundingRect(cnt)
    print(x, y, w, h)
    plt.imshow(im[y:y+h, x:x+w], 'gray')
    plt.xticks([]), plt.yticks([])
    
    plt.show()

def resizeKeepRatio(img, size, ori_size = None):
    """ori_size: None, size should be tuple
        tuple, width of size will be used (height will be ignored), keep ratio
        @retrun resized image
    """
    if ori_size is not None:
        h = ori_size[1] / ori_size[0] * size[0]
        size = (size[0], int(h))
    return cv.resize(img, size)

def showImage(img, name = "default", size = (1024, 1024), ori_size = None, skip_wait=False):
    if size is not None:
        img = resizeKeepRatio(img, size, ori_size)
    cv.imshow(name, img)
    if not skip_wait:
        cv.waitKey(0)

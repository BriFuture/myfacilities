"""
TODO reformat this file and imageutil, labelme utils
"""


import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import os.path as osp
from .labelmeutils import encodeImage

import torch

class PlotHelper():
    """Helper used to plot serveral images into one figure
    """
    def __init__(self, vmin = 0, vmax = 255):
        self.vmin = vmin
        self.vmax = vmax
        
    def checkzeros(self, imglist, imgstart = 0):
        channels = len(imglist)
        zero = []
        for i in range(channels):
            imgploti = imglist[i]
            # print(imgploti)
            if np.count_nonzero(imgploti) == 0:
                zero.append(i + imgstart)

        if len(zero) > 0:
            print(f"Following (total: {len(zero)}) are all zeros: ", zero)

        self.zero = zero
        return len(zero)

    def _plot(self, imglist, rows=2, cols=2, dpi=60, imgstart=0, names=None, cmap='gray'):
        """
        imglist(list(np.ndarray)):
        """

        fig = plt.figure(figsize=(18, 16), dpi= dpi, facecolor='w', edgecolor='w', clear=True)
        imgstart = imgstart * (rows * cols)
        channels = len(imglist)
        # channels = imglist.shape[0]
        if imgstart + rows * cols > channels:
            print(f"Index Invalid: {imgstart} - {rows * cols}")
            # imgstart = 0

            # names = [f"{i}" for i in range(rows * cols)]
        for idx in range(imgstart, imgstart + rows * cols): #imglist.shape[0]):
            if idx >= channels: break

            a = fig.add_subplot(rows, cols, idx - imgstart +1)
            # Turn off tick labels
            a.set_yticklabels([])
            a.set_xticklabels([])
            if names is not None:
                a.set_xlabel(names[idx])
            # plt.axis('off')
            imgploti = imglist[idx]
            imgploti = plt.imshow(imgploti, cmap=cmap) # * 255, , vmin=0, vmax=255)


    def plotImgList(self, imglist, checkzeros=True, names=None, **kwargs):
        """
        Arguments:
            imglist (np.ndarray | list | torch.Tensor)
        """
        imgtype = type(imglist)
        if imgtype == np.ndarray:
            shape = imglist.shape
            ndim = imglist.ndim 
            if ndim == 2:
                imglist = [imglist]
            elif ndim == 3:
                imglist = [img for img in imglist]
            else:
                raise ValueError(f"Unsupported ndarray shapeP: {shape} {ndim}")
        elif imgtype == list:
            pass
        elif imgtype == torch.Tensor:
            assert len(imglist.shape) == 4  ## B C H W
            imglist = imglist[0,:,:,:].cpu()
            print("Plot on converted Tensor: ", imglist.shape)
            ims = [imglist[i].cpu().numpy() for i in range(len(imglist))]
        else:
            raise ValueError(f"Unsupported type: {imgtype}")
        # print(imglist)
        if names is not None:
            assert len(names) >= len(imglist)
        if checkzeros:
            self.checkzeros(imglist, kwargs.get("imgstart", 0))

        self._plot(imglist, names=names, **kwargs)
        
    def plotTensor(self, imtensor, cmap='gray', **kwargs):
        """Tensor
        """
        tlen = len(imtensor.shape)
        if tlen == 3:
        # assert imtensor.shape[0] == 3
            im = imtensor.permute(1, 2, 0).mul(256).byte().cpu().numpy()
            self.plotArray(im, cmap, **kwargs)
        elif tlen == 4:
            ## channel may not be 3
            # print(imtensor.shape)
            for imt in imtensor:
                merged = np.zeros((imt.shape[-2], imt.shape[-1]))
                for im in imt:
                    im = im.cpu().numpy()
                    self.plotArray(im, cmap, **kwargs)
                    # merged = np.where(im > 0, im, merged)
                # self.plotArray(merged, cmap, **kwargs)

    def plotArray(self, im, cmap='gray', figsize=(12, 8), **kwargs):
        # print(im.shape)
        fig = plt.figure(figsize=figsize, dpi= 100, facecolor='w', edgecolor='w', clear=True)
        plt.imshow(im, cmap=cmap)
        plt.show()

    def plotHeatmap(self, imarr, cmap='gray', **kwargs):
        # for batch in imarr:
        hms = imarr[0]
        for hm in hms:
            self.plotArray(hm)

    def drawPoints(self, points, size=None, img=None, bright=2.5, cmap='gray', **kwargs):
        """
            points (list(np.ndarray))
            size (tuple(height, width))
        """
        # assert len(size) == 3
        assert size is not None or img is not None

        if img is None:
            img = np.zeros(size, dtype=np.float)
        else:
            img = img.copy()
        # print(img.shape)
        for po in points:
            x, y = po[:2]
            x, y = int(x), int(y)
            img[x, y] = bright
        self.plotArray(img, cmap=cmap, **kwargs)
        

from PIL import Image, ImageDraw
import imgviz
import types
import json
import shutil as sh

class MaskContourSaver():
    """
    Used to save mask contour predicted by ai. 
    @Note it depends on certain model and datasets, and outpus only mask, 
        use PredictionViewer instead, which provides keypoint label output
    @See torchutils.PredictionViewer

    Arguments:
        labels (function , list) it maps the idx of predicted results into Readable texts

        root(str): root dir contains images, should be different from dst_dir
    """

    def __init__(self, labels=None, root = None, img_suffix=".jpg", debug=False, **kwargs):
        self.debug = debug
        self.root = root
        # self.imgs = sorted([file for file in os.listdir(root) if file.endswith(img_suffix)])
        self.img_suffix = img_suffix

        if type(labels) is list:
            self._captions = labels
        else:
            label = []
            assert isinstance(labels, types.FunctionType)
            try:
                for i in range(1000):
                    label.append(labels(i))
            except: pass
            self._captions = label
        if self.root is not None:
            print("Root: ", self.root)
        print("Get captions: ", self._captions)


    def _save(self, im, dst_dir, imname, shapes, imsize):
        """
        im(np.ndarray | PIL.Image.Image)
        imname(str): image name without suffix, such as `1.jpg` will get `1`
        shapes(list(dict)):
        """
        imfilename = f"{imname}{self.img_suffix}"
        ## @See Also encodeImage
        # imageData = LabelFile.load_image_file(imgloc)
        # imageData = base64.b64encode(imageData).decode("utf-8")
        imageData = encodeImage(im, self.img_suffix)

        data = {"version": "4.5.5",
            "flags": {}, "shapes":shapes, "imagePath": imfilename, 
            "imageHeight": imsize[1],
            "imageWidth": imsize[0],
            "imageData": imageData
        }
        if self.root is not None:
            try:
                imgloc = osp.join(self.root, imfilename)
                sh.copy(imgloc, osp.join(dst_dir, imfilename))
            except Exception as e:
                print(e)
        jsonloc = osp.join(dst_dir, imname+".json")
        with open(jsonloc, "w") as f:
            f.write(json.dumps(data, indent=True))
    
    def _getPrediction(self, idx):
        """
        @Returns 
            True prediction is valid
        """
        img, _ = dataset[idx]
        imsize = img.shape[-2], img.shape[-1]
        img = img.to(device)
        predictions = self.model([img])
        prediction = predictions[0]  ## batch size is one

        self.boxes = prediction["boxes"]
        self.labels = prediction["labels"].cpu().numpy()
        self.masks = prediction['masks'].mul(255).byte().cpu().numpy()
        
        self.maskcount = masks.shape[0] if maskperimage <= 0 else maskperimage
        
    def save(self, dst_dir, dataset, imgnames, drange=None,\
            maskperimage=-1, skipUnmask=True,\
            contourDis = 8, contourGap=4, contourArea=None, contourThresh=None, contourSaveMax=False, \
            device=torch.device('cpu'), **kwargs):
        """
        Before invoking this method, model should be in `eval` mode, `with torch.no_grad()` is 
        necessary.

        Arguments:
            dataset(torch.utils.Dataset): 
                dataset should return tuple (img, target)
                Generally property: 
                1. imgs, which provide the names of images, it should be passed as parameter `imgnames`
                2. root, root_dir indicates where the data is from, it should be passed to constructor
                3. Labels, which could be used to repreter label index

            imgnames(list):
            
            drange(range):

            maskperimage(int): if -1, then no mask limit for each image

            contourThresh(tuple): [threshold, maxval, imtype]
            contourArea(int): if none, contour area will not be accounted for. Otherwise any contour whose are is less than `contourArea` will be filtered
            contourGap(int): 
            contourSaveMax(bool): Not Implemented Yet
        """
        assert dst_dir != self.root
        # device = torch.device(device)
        # assert device == self.model.device
        os.makedirs(dst_dir, exist_ok=True)
        
        if drange == None:
            drange = range(0, len(dataset))
        
        assert drange[-1] <= len(dataset)

        if contourThresh is None:
            contourThresh = [127, 255, 0]
        
        assert len(contourThresh) == 3
        print("Start prediction on: ", drange, "...")
        self.contour_areas = []
        self.skipped_idxes = []
        for idx in drange:
            self._getPrediction(idx)
            if self.maskcount == 0:
                continue

            found_contours = []
            contour_areas = []
            for c in range(self.maskcount):
                maskdata = self.masks[c, 0]
                label_name = self._captions[self.labels[c]]

                ret, thresh = cv.threshold(maskdata, contourThresh[0], contourThresh[1], contourThresh[2])
                contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                if len(contours) == 0:
                    continue
                # reshape as a list, the first contour will be used
                contour = contours[0]
                actualarea = cv.contourArea(contour)

                if contourArea is not None and actualarea < contourArea:
                    # print("ContourArea is too small: ", actualarea)
                    continue
                
                if self.debug:
                    print("Actual contour area: ", actualarea)
                contour = contour.reshape(-1, 2)
                contour = cv.convexHull(contour)
                ## disable contourGap using hull instead
                if len(contour) > contourGap * 5:
                    contour = contour[0::contourGap]
                if len(contour) < 2:
                    continue
                found_contours.append(contour)
                contour_areas.append(actualarea)

            if len(contour_areas) > 0:
                self.contour_areas.append(contour_areas)
            shapes = []
            for contour in found_contours:
                shapes.append({"label": label_name, 
                    "points": contour.tolist(),       
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })
            if len(shapes) == 0 and skipUnmask:
                self.skipped_idxes.append(idx)
                continue
            print(f"Idx: {idx}: mask will be saved: {len(shapes)}")

            imfilename = imgnames[idx]
            imname = imfilename[:imfilename.rfind(self.img_suffix)]
            self._save(osp.join(self.root, imfilename), dst_dir, imname, shapes, imsize)
            print("==============================")
            # if self.debug:
            #     break
        
    @staticmethod
    def saveContour(prediction, dataset, idx, saveLoc, labelList = None,\
        gap=4, size=(512, 512), img_suffix=".jpg"):
        """
        Deprecated, Use Saver instead
        Save Contour By AI prediction and CV contours

        Arguments:
            dataset(torch.utils.Dataset)
            idx(int)
            saveLoc(str)
            labels(list)
        """
        os.makedirs(saveLoc, exist_ok=True)
        boxes = prediction["boxes"]
        labels = prediction["labels"]
        masks = prediction['masks']
        i = 1
        count = masks.shape[0]
        label = labels.cpu().numpy()
        shapes = []
        if labelList is None:
            labelList = dataset.Labels
        print(f"{idx} Masks: {count}")

        imname = dataset.imgs[idx]
        imfilename = imname[:imname.rfind(img_suffix)]
        for c in range(count):
            mask = masks[c, 0]
            label_name = labelList[label[c]]
            imdata = mask.mul(255).byte().cpu().numpy()
            # im = Image.fromarray(imdata)
            # display(im)
            ret, thresh = cv.threshold(imdata, 127, 255, 0)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            # reshape as a list
            contour = contours[0].reshape(-1, 2)
            if len(contour) > gap * 5:
                contour = contour[0::gap]
            if len(contour) < 2:
                continue
            shapes.append({"label": label_name, "points": contour.tolist(),       
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })
        imageData = LabelFile.load_image_file(osp.join(dataset.root, imname))
        imageData = base64.b64encode(imageData).decode("utf-8")
        data = {"version": "4.5.5",
            "flags": {}, "shapes":shapes, "imagePath": imname, 
            "imageHeight": size[1],
            "imageWidth": size[0],
            "imageData": imageData
        }
        with open(osp(saveLoc, imfilename+".json"), "w") as f:
            f.write(json.dumps(data, indent=True))

    @staticmethod
    def mergeLabelIntoImages(labeldir, imagedir):
        labels = os.listdir(labeldir)
        for label in labels:
            if not label.endswith(".json"):
                raise ValueError("Labels should be json file")
            path = osp.join(imagedir, label)
            if osp.exists(path):
                print("Removing: ", path)
                os.remove(path)
            sh.move(osp.join(labeldir, label), imagedir)

        sh.rmtree(labeldir)
import math
class PredictionViewer(MaskContourSaver):
    """
    1. Display model's prediction 
    2. Save prediction as labeled json files or visualized images

    Arugments:
    """

    def __init__(self, labels=None, img_suffix=".jpg", debug=False, kpcaptions=None, **kwargs):
        super().__init__(labels=labels, img_suffix=img_suffix, debug=debug, **kwargs)
        self.kpcaptions = kpcaptions
        if kpcaptions is not None:
            print("KP captions:", kpcaptions)

    @staticmethod
    def mask_to_box(mask):
        assert type(mask) is np.ndarray
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    def setPrediction(self, predictions, mask3channels=True, \
        score_threshold=0.5, hint=True):
        """Note, if predictions is a tensor contains multiple predictions, the first one will be used
            and others will be droped.
            score_threshold is deprecated, if you want to filter that is not clearly identified,
            use `box_thresh` for rpn.
        """

        self.score_threshold = score_threshold

        if type(predictions) == list:
            if len(predictions) == 0:
                raise ValueError("No Prediction found")
            prediction = predictions[0]
        else:
            prediction = predictions

        self.boxes =  prediction["boxes"].cpu().numpy()
        self.labels = prediction["labels"].cpu().numpy()
        self.scores = prediction["scores"].cpu().numpy()

        # proposals = []
        # for i, score in enumerate(self.scores):
        #     # print("Score", i, score)
        #     if score > self.score_threshold:
        #         proposals.append(i)
        # self.boxes = np.take(self.boxes, proposals, axis=0)
        # print(self.boxes)

        if "masks" in prediction:
            masks = prediction['masks']
            masks = masks.permute(0, 2, 3, 1)#.squeeze(0)
            ## N H W C
            # if mask3channels:
            #     if masks.shape[1] == 1:
            #         masks = masks.repeat(1, 1, 1, 3)
            #     # now masks showld be N H W C
            # else:
            #     masks = masks.squeeze(3)
                # now masks showld be N H W
            # maskimg = []
            # for mask in masks:
                # maskim.append(mask.mul(128).byte().cpu().numpy())
                # maskimg.append(Image.fromarray(mask.mul(128).byte().cpu().numpy()))
            # self.masks = maskimg
            self.masks = masks.mul(255).byte().cpu().numpy()
            # print(self.masks.shape)
        else:
            self.masks = None
        if "keypoints" in prediction:
            keypoints = prediction["keypoints"]
            self.keypoints = keypoints.cpu().numpy()
            self.keypoints_scores = prediction["keypoints_scores"].cpu().numpy()
            # N 2
        else:
            self.keypoints = None

        # print(self.masks.shape, self.keypoints.shape)

        if hint:
            m = self.masks.shape if self.masks is not None else 0
            k = self.keypoints.shape if self.keypoints is not None else 0
            print("Found masks", m, " Keypoints: ", k)

    def convertImg(self, imarr):
        """
        Argument:
            imarr (np.ndarray) # ndim == 3 (H, W, C) channels == 3
        Return:
            np.ndarray (H, W, C)
        """
        imtype = type(imarr)
        # print(imtype)
        if imtype == np.ndarray:
            pass
        elif isinstance(imarr, torch.Tensor):
            assert len(imarr.shape) == 3
            imarr = imarr.permute(1, 2, 0)
            if imarr.shape[-1] == 1:
                imarr = imarr.repeat(1, 1 ,3)
            imarr = imarr.mul(255).byte().cpu().numpy()
        elif isinstance(imarr, (Image.Image, )):
            # im = Image.fromarray(imarr, mode="RGB")
            im = imarr
            imarr=np.asarray(imarr)
            if imarr.ndim != 3:
                imarr = np.asarray([imarr])
                imarr = np.repeat(imarr, 3, axis=0)
                imarr = imarr.swapaxes(0, 1)
                imarr = imarr.swapaxes(1, 2)
        else:
            raise ValueError("Unsupported Type", imtype)
        
        assert imarr.ndim == 3 and imarr.shape[-1] == 3
        # print("Show: ", imarr.shape)

        return imarr

    def showMask(self, imarr, box_swap_xy=True,
        # mask
        mask_threshold=30, mask_alpha=0.25, box_only=False):
        """
        @return np.ndarray
        """
        imarr = self.convertImg(imarr)

        captions = self.caption_with_score

        # print(self.boxes, captions)
        colormap = imgviz.label.label_colormap()
        if self.masks is None or box_only:
            # data = imgviz.data.arc2017()
            # bboxes = self.boxes
            if box_swap_xy:
                bboxes = []
                for box in self.boxes:
                    bboxes.append((box[1], box[0], box[3], box[2]))
            else:
                bboxes = self.boxes
            viz = imgviz.instances2rgb(imarr, bboxes=bboxes, \
                labels=self.labels, captions=captions, \
                colormap=colormap, alpha=mask_alpha, \
                font_size=12, line_width=1)
        else:
            masks = self.masks > mask_threshold
            # mask didnot need all 3 channels
            masks = masks[:, :, :, 0]
                # masks = np.squeeze(masks, axis=3)
            viz = imgviz.instances2rgb(imarr, masks=masks, \
                labels=self.labels, captions=captions, \
                colormap=colormap, alpha=mask_alpha, \
                font_size=12, line_width=1)
        return viz

    def showKeypoint(self, im, radius=5, kp_fill=None, num_color=248, **kwargs):
        """
        Arugments:
            im(PIL.Image.Image)
        """
        if self.keypoints is None:
            return None

        assert isinstance(im, Image.Image)
        colormap = imgviz.label.label_colormap()
        imdraw = ImageDraw.Draw(im)
        keypoints = self.keypoints[:,:,:2]
        i = 1
        rgbmode = im.mode == 'RGB'
        for kps in keypoints:
            for k in kps:
                # print(fill, k)
                if rgbmode:
                    fill = tuple(x for x in colormap[num_color-i-1]) if kp_fill is None else kp_fill
                else:
                    fill = colormap[num_color-i-1][0] if kp_fill is None else kp_fill

                if radius > 0:
                    imdraw.rectangle([k[0] - radius, k[1] - radius, k[0]+radius, k[1]+radius], fill=fill)
                else:
                    imdraw.point([k[0], k[1]], fill=fill)
                i+=1
        return im

    @property
    def caption_with_score(self):
        return [f"{self._captions[i]}: {score:.2f}" for i, score in zip(self.labels, self.scores)] #if score > self.score_threshold

    def show(self, imarr, box_swap_xy=True,
        # mask
        mask_threshold=30, mask_alpha=0.25,
        # keypoint
        radius=5, kp_fill=(128, 200, 128), **kwargs
        ):
        """Draw masks and keypoints(if exists) on the image

        @Note setPrediciton must be called before this method invoked
        
        Arguments:
            imarr (np.ndarray) # ndim == 3 (H, W, C) channels == 3
            box_swap_xy(bool), the imgviz tool xy axis seems different from the screen axis,
                so it will be swapped if necessary.

        Return:
            im (Image.Image)
        """
        imarr = self.convertImg(imarr)
        # swap x, y
        viz = self.showMask(imarr, box_swap_xy, mask_threshold=mask_threshold, mask_alpha=mask_alpha)
        im = Image.fromarray(viz)
        im = self.showKeypoint(im, radius=radius, kp_fill=kp_fill)
        return im

    def displayWrongHint(self, predictions, idx=None, showWrongHintOnly=True, expected_box_num=2):    
        self.setPrediction(predictions, hint=False)
        box_num = len(self.boxes)
        if box_num != expected_box_num:
            print(f"{idx} Wrong predicted box: ", box_num)
        elif not showWrongHintOnly:
            print("Predicted box: ", box_num)

    Save_Success = 0
    Save_SkippedNoMask = 1
    Save_NoMaskPredicted = 2

    def _maskToContour(self, found_contours, contourGap, contourThresh, contourArea):
        if self.masks is None:
            return

        maskNum = self.masks.shape[0] # 

        for ci in range(maskNum):
            label_name = self._captions[self.labels[ci]]
            maskdata = self.masks[ci]
            ret, thresh = cv.threshold(maskdata, contourThresh[0], contourThresh[1], contourThresh[2])
            # thresh = cv.adaptiveThreshold(maskdata, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
            # if self.debug:
            # display(Image.fromarray(thresh))
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # print("fondContours", maskdata.shape, label_name, len(contours))
            if len(contours) == 0:
                continue
            # reshape as a list, the first contour will be used
            newcontours = []
            for contour in contours:
                actualarea = cv.contourArea(contour)

                if contourArea is not None and actualarea < contourArea:
                    # print(f"[{label_name}] ContourArea is too small: ", actualarea)
                    continue

                contour = contour.reshape(-1, 2) # two-d array is needed for labelme
                # print("Contour ", len(contour))
                conlen = len(contour)
                if conlen < 2:
                    continue
                ncontour = []
                i = 0
                x1 = contour[i]
                ncontour.append(x1)
                i += 1
                while i < conlen - 1: 
                    x2 = contour[i]
                    dis = (x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2
                    # print(dis)
                    dis = math.sqrt(dis)
                    if dis >= contourGap or contourGap <= 0:
                        x1 = x2
                        ncontour.append(x1)
                    i += 1
                contour = np.asarray(ncontour)

                # if len(contour) > contourGap * 4:
                #     contour = contour[0::contourGap]
                newcontours.append((contour, actualarea))
            # sort by area size
            newcontours = sorted(newcontours, key=lambda x: len(x[0]), reverse=True)
            # print("Contour: ", [ len(c) for c, a in newcontours])
            found_contours[label_name] = newcontours

    def saveMask(self, im, dst_dir, imname, \
        maskperimage=-1, contourGap=6, contourArea=None, contourThresh=None,\
        skipUnmask=True, radius = 5, withoutMask=False, \
        style="labelme", **kwargs):
        """@Note setPrediciton must be called before this method invoked
        save mask / keypoints

        Arguments:
            im(torch.Tensor): image object processed by Dataset/Model, 
                it should be the same with the original image no matter how it is processed (preprocess & postprocess).
                **Note, if the Dataset/Model didnot reveal the image, you'd better pass a original Image object**
            imname(str): image pure name withou suffix
            style(str): only support `labelme` currently
            maskperimage(int): if -1, then no mask limit for each image, otherwize each class will only get certain masks
        """
        os.makedirs(osp.join(dst_dir, "Annotation"), exist_ok=True)
        im = self.convertImg(im)
        
        if contourThresh is None:
            contourThresh = [127, 255, 0]

        # print(found_contours)
        shapes = []
        ## add mask type as polygon
        if not withoutMask:
            found_contours = { c: None for c in self._captions }
            ## TODO contour will change orders
            self._maskToContour(found_contours, contourGap, contourThresh, contourArea)
            for label_name, contours in found_contours.items():
                mc = 0
                if contours is None:
                    continue
                for contour, areasize in contours:
                    if maskperimage > 0 and mc >= maskperimage:
                        break
                    mc += 1
                    shapes.append({"label": label_name, 
                        "points": contour.tolist(),       
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    })
        if self.keypoints is not None:
            ## TODO how to map keypoints to label better?
            kpNum = self.keypoints.shape[0]
            mc = 0
            for ci in range(kpNum):
                kps = self.keypoints[ci]
                if maskperimage > 0 and mc >= maskperimage:
                    break

                label_name = self._captions[self.labels[ci]]
                for i, kp in enumerate(kps):
                    if self.kpcaptions is None:
                        ln = f"{label_name}-{i}"
                    else:
                        ln = self.kpcaptions[i]
                    # print(ln, kp)
                    shapes.append({"label": ln, 
                        "points": [ [ float(kp[0]), float(kp[1]) ] ],
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {}
                    })
        if len(shapes) == 0 and skipUnmask:
            return PredictionViewer.Save_SkippedNoMask
        # print(f"mask will be saved: {len(shapes)}")

        if imname.endswith(self.img_suffix):
            imname = imname[:imname.rfind(self.img_suffix)]
        # imname should withoud suffix
        self._save(im, osp.join(dst_dir, "Annotation"), imname, \
            shapes, im.shape)
        return PredictionViewer.Save_Success

    def saveViz(self, im, dst_dir, imname, mask_threshold = 127, radius=3, **kwargs):
        """
        Arguments:
            process_func (function):
        """
        if imname.endswith(self.img_suffix):
            imfilename = imname # full name
            imname = imname[:imname.rfind(self.img_suffix)]
        else:
            imfilename = f"{imname}{self.img_suffix}"

        ## save visualized image
        os.makedirs(osp.join(dst_dir, "viz"), exist_ok=True)
        annIm = self.show(im, mask_threshold=mask_threshold, radius=radius)
        annIm.save(osp.join(dst_dir, "viz", imfilename))


import torch.distributed as dist
from .imageutils import ImagePreprocessor, CvImageReader

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

class F1ScoreCalcor():
    """Binary Label F1-score calculator
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def compare(self, preds, targets):
        """
        Arguments:
            preds(Tensor): (n x 1)
            targets(Tensor): (n x 1)
        """
        self.TP += np.sum(preds & (targets == 1))
        self.FP += np.sum(preds & (targets == 0))
        self.FN += np.sum(~preds & (targets == 1))

    def export(self, printit=True):
        TP = self.TP
        FP = self.FP
        FN = self.FN
        precision = TP / (TP+FP)
        recall = TP / (TP + FN)
        f1score = 2 * precision * recall / (precision + recall)
        if printit:
            print(f"TP {self.TP}, FP {self.FP}, FN {self.FN}, Precision: {precision:.6f}, Recall: {recall}, F1-score: {f1score}")
        return precision, recall, f1score

class PearsonCorrelation(torch.nn.Module):


    def forward(self,tensor_1,tensor_2):
        x = tensor_1
        y = tensor_2

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost

class PearsonClassfication():
    """
    Arguments:
        base (torch.Tensor)
        converter( Callable ): which is used to convert imgs into torch.Tensor if not None
    """
    def __init__(self, base, converter=None):
        self.pc = PearsonCorrelation()
        self.converter = converter
        if not isinstance(base, torch.Tensor) and self.converter is not None:
            base = self.converter(base)
        self.base = base
        assert isinstance(self.base, torch.Tensor)

    def compare(self, imglists, thresh, names = None):
        """
        Arguments:
            imglists(list(torch.Tensor) Iterable) it could be a list which contains torch.Tensor instances,
                or it could be just torch.utils.data.Dataset which is iterable for getting images,
                if elemetns in list is not instance of torch.Tensor and converter is not None,
                then it will be converted first
            names(list(str) | None)
        """
        similar = []
        unsimilar = []
        corr = []
        self.pc.eval()
        print("Length of list to be checked: ", len(imglists))
        with torch.no_grad():
            for idx, img in enumerate(imglists):
                if type(img) is tuple:
                    img = img[0]
                if not isinstance(img, torch.Tensor) and self.converter is not None:
                    img = self.converter(img)
                assert isinstance(img, torch.Tensor)

                c = self.pc(self.base, img)
                corr.append(c)
                name = names[idx] if names is not None else ""
                if c > thresh:
                    similar.append((img, name))
                else:
                    unsimilar.append((img, name))
        return similar, unsimilar, corr


def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)
 
    # 获取模型输出的feature/score
    model.eval()
    features = model.features(img)
    output = model.classifier(features)
 
    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g
 
    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
 
    features.register_hook(extract)
    pred_class.backward() # 计算梯度
 
    grads = features_grad   # 获取梯度
 
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
 
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]
 
    # 以下部分同Keras版实现
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
 
    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()
 
    img = cv.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘


def getSGDOptimizer(model, lr=0.005, weight_decay=0.00002, step_size=6, **kwargs):
    params = [p for p in model.parameters() if p.requires_grad]
    # for name, param in model.named_parameters():
    #      if param.requires_grad:
    #         print(name)

    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=0.9, weight_decay=weight_decay)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=step_size,
                                                gamma=kwargs.get("gamma", 0.2))
    return optimizer, lr_scheduler
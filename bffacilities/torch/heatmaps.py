import numpy as np

class HeatmapGeneratorWithoutBatch():
    """
    Arguments: 
        out_size(int): image output resolution
        out_channels(int): model output Tensor channels, num_keypoints
    """
    def __init__(self, out_size, out_channels):
        self.out_size = out_size
        self.out_channels = out_channels
        sigma = self.out_size / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints, ratio = 1.0):
        """
        Arguments:
            keypoints: [Tensor(K, 3)] shape: N K 3
            ratio (float): generate target keypoints to dest size heatmap
        """
        batch_size = len(keypoints)
        hms = np.zeros(shape = (batch_size, self.out_size, self.out_size), dtype = np.float32)
        sigma = self.sigma
        # print("KP: ", keypoints)
        for batch_idx, kp in enumerate(keypoints): # batch kps
            # print("KP", kp)
            # assert len(kp) == 4
            for pt in kp:
                # kp torch.Tensor, size (3, )
                # print(idx, pt)
                if pt[2] > 0: 
                    x, y = int(pt[0] * ratio), int(pt[1] * ratio)
                    if x<0 or y<0 or x>=self.out_size or y>=self.out_size:
                        continue
                    topleft = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                    bottomright = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                    c,d = max(0, -topleft[0]), min(bottomright[0], self.out_size) - topleft[0]
                    a,b = max(0, -topleft[1]), min(bottomright[1], self.out_size) - topleft[1]

                    cc,dd = max(0, topleft[0]), min(bottomright[0], self.out_size)
                    aa,bb = max(0, topleft[1]), min(bottomright[1], self.out_size)
                    hms[batch_idx, aa:bb, cc:dd] = np.maximum(hms[batch_idx, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms
## for debug
from .torchutils import PlotHelper
ph = PlotHelper()

class HeatmapGenerator():
    """Generate Heatmap with Batch

    Arguments: 
        out_size(int): image output resolution
        out_channels(int): model output Tensor channels
        sigma(float)
    """
    def __init__(self, out_size, out_channels, sigma = None):
        self.out_size = out_size
        self.out_channels = out_channels
        if sigma is None:
            sigma = self.out_size / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints, ratio = 1.0):
        """
        keypoints: target ground truth keypoints [Tensor(K, 3)] shape: N K 3
        """
        batch_size = len(keypoints)
        hms = np.zeros(shape = (batch_size, self.out_channels, self.out_size, self.out_size), dtype = np.float32)
        sigma = self.sigma
        # print("KP: ", keypoints)
        for batch_idx, kp in enumerate(keypoints): # batch kps
            # print(kp)
            for idx, pt in enumerate(kp):
                # pt.shape (3,) it is [x, y, v]
                if pt[2] == 0: continue
                x, y = int(pt[0] * ratio), int(pt[1] * ratio)
                # x, y = pt[0], pt[1]
                # if x<0 or y<0 or x>=self.out_size or y>=self.out_size:
                    # continue

                topleft = int(x - 3*sigma - 1), int(y - 3*sigma - 1) 
                bottomright = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c, d = max(0, -topleft[0]), min(bottomright[0], self.out_size) - topleft[0]
                a, b = max(0, -topleft[1]), min(bottomright[1], self.out_size) - topleft[1]

                cc, dd = max(0, topleft[0]), min(bottomright[0], self.out_size)
                aa, bb = max(0, topleft[1]), min(bottomright[1], self.out_size)
                hms[batch_idx, idx, aa:bb,cc:dd] = np.maximum(hms[batch_idx, idx, aa:bb,cc:dd], self.g[a:b,c:d])
                # print("HM: ", aa, bb, cc, dd, "XY: ", x, y, "Ratio: ", ratio)
                # print("A, B: ", a, b)
                # ph.plotArray(hms[batch_idx, idx, :, :])
                # print("Max: ", np.max(hms[batch_idx, idx, :, :]), hms[batch_idx, idx, :, :].sum())
        # testhms = np.zeros(shape = (self.out_size, self.out_size), dtype = np.float32)
        # for hm in hms[0]:
        #     testhms += hm
        # ph.plotArray(testhms)
        return hms

import torch
import torch.nn as nn
from torch.nn import functional as F

def match_format(dic):
    loc = dic['loc_k'][0,:,0,:]
    val = dic['val_k'][0,:,:]
    ans = np.hstack((loc, val))
    ans = np.expand_dims(ans, axis = 0) 
    ret = []
    ret.append(ans)
    return ret

class HeatmapParser:
    def __init__(self):
        from torch import nn
        self.pool = nn.MaxPool2d(3, 1, 1)

    def nms(self, det):
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def calc(self, det):
        with torch.no_grad():
            det = torch.autograd.Variable(torch.Tensor(det))
            # This is a better format for future version pytorch

        det = self.nms(det)
        h = det.size()[-2]
        w = det.size()[-1]
        det = det.view(det.size()[0], det.size()[1], -1)
        val_k, ind = det.topk(1, dim=2)

        x = ind % w
        y = (ind / w).long()
        ind_k = torch.stack((x, y), dim=3)
        ans = {'loc_k': ind_k, 'val_k': val_k}
        return {key:ans[key].cpu().data.numpy() for key in ans}

    def adjust(self, ans, det):
        for batch_id, people in enumerate(ans): 
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2]>0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = det[0][joint_id]
                        if tmp[xx, min(yy+1, tmp.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
                            y+=0.25
                        else:
                            y-=0.25

                        if tmp[min(xx+1, tmp.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
                            x+=0.25
                        else:
                            x-=0.25
                        ans[0][0, joint_id, 0:2] = (y+0.5, x+0.5)
        return ans

    def parse(self, det, adjust=True):
        ans = match_format(self.calc(det))
        if adjust:
            ans = self.adjust(ans, det)
        return ans

from PIL import Image
from torchvision.models.detection.roi_heads import keypoints_to_heatmap
import torchvision
## for debug
# from .torchutils import PlotHelper
# ph = PlotHelper()

def translateKP(targets):
    """
        targets (list(dict))
            dict {
                "boxes": ,
                "keypoints",
            }
    """
    keypoints = []
    for gt in targets: # batch
        tmpkp = []
        for kps in gt["keypoints"]: # N points
            for kp in kps:
            # for this model we have to split all 4 keypoints separately
                tmpkp.append(kp.unsqueeze(0))
        
        # print("Tmpkp: ", tmpkp[0].shape)
        tmpkp = torch.cat(tmpkp, dim=0)
        # print("Tmpkp: ", tmpkp.shape)
        keypoints.append(tmpkp)
    # list()
    return keypoints

def np_translateKP(targets, N, K):
    keypoints = np.zeros((N, K, 2))
    for bid, gt in enumerate(targets): # batch
        for oid, kps in enumerate(gt["keypoints"]): # N points
            for kid, kp in enumerate(kps):
            # for this model we have to split all 4 keypoints separately
                # tmpkp.append(kp.unsqueeze(0))
                keypoints[bid, oid*K + kid, 0] = kp[0]
                keypoints[bid, oid*K + kid, 1] = kp[1]
    return keypoints

def heatmaps_to_keypoints(heatmaps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = widths.clamp(min=1)
    heights = heights.clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()
    # print("WH: ", widths, heights, widths_ceil, heights_ceil)

    num_keypoints = heatmaps.shape[1]

    if torchvision._is_tracing():
        xy_preds, end_scores = _onnx_heatmaps_to_keypoints_loop(heatmaps, rois,
                                                                widths_ceil, heights_ceil, widths, heights,
                                                                offset_x, offset_y,
                                                                torch.scalar_tensor(num_keypoints, dtype=torch.int64))
        return xy_preds.permute(0, 2, 1), end_scores

    xy_preds = torch.zeros((len(rois), 3, num_keypoints), dtype=torch.float32, device=heatmaps.device)
    end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=heatmaps.device)

    for i in range(len(rois)):
        roi_map_width = int(widths_ceil[i].item())
        roi_map_height = int(heights_ceil[i].item())
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = torch.nn.functional.interpolate(
            heatmaps[i][None], size=(roi_map_height, roi_map_width), mode='bicubic', align_corners=False)[0]
        # roi_map_probs = scores_to_probs(roi_map.copy())
        ## roi_map.shape K, H, W
        w = roi_map.shape[2] ## equals to roi_map_width
        pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

        x_int = pos % w
        y_int = (pos - x_int) // w
        # print("ROI HM to KP", x_int, y_int, w, roi_map_height)
        # print(width_correction, height_correction)

        # assert (roi_map_probs[k, y_int, x_int] ==
        #         roi_map_probs[k, :, :].max())
        x = (x_int.float() + 0.5) * width_correction
        y = (y_int.float() + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map[torch.arange(num_keypoints), y_int, x_int]

    return xy_preds.permute(0, 2, 1), end_scores


class Predict2Heatmap():
    """
     dimenstion (int): 2: 2-D heatmapt
    """
    def __init__(self, num_keypoints, dimension=2):

        self.num_keypoints = num_keypoints
        self.nums = dimension * num_keypoints

    def generate_2d_integral_preds_tensor(self, heatmaps, x_dim, y_dim):
        # assert isinstance(heatmaps, torch.Tensor)
        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_keypoints, y_dim, x_dim))
        accu_x = heatmaps.sum(dim=2)
        accu_y = heatmaps.sum(dim=3)
        # print(heatmaps.shape, accu_x.shape, accu_y.shape)
        # accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim) \
        #     .type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
        # accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim) \
        #     .type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
        device = accu_x.device
        # accu_x is prob now
        accu_x = accu_x * torch.arange(x_dim, dtype=torch.float, device=device)
        accu_y = accu_y * torch.arange(y_dim, dtype=torch.float, device=device)
        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        # convert discrete coordinates to continuous coordinate

        return accu_x, accu_y


    def softmax_integral_tensor(self, preds, hm_width, hm_height):
        # heatmaps to keypoint
        # global soft max , batch stacks
        preds = preds.reshape((preds.shape[0], self.num_keypoints, -1))
        preds = F.softmax(preds, 2)

        # integrate heatmap into joint location
        x, y = self.generate_2d_integral_preds_tensor(preds, hm_width, hm_height)
        # x, y should be the position
        # x = x / float(hm_width) - 0.5
        # y = y / float(hm_height) - 0.5
        preds = torch.cat((x, y), dim=2)
        # print(preds)
        # preds = preds.reshape((preds.shape[0], self.nums))
        # print(preds)
        return preds


class HeatmapLoss(nn.Module):
    """
    Arguments:
        nstack(int)
        heatmapGenerator(nn.Module)

    loss for detection heatmap
    
    It's calcuates L2 distance between prediction and groundtruth
    """
    def __init__(self, heatmapGenerator, dest_size=64, \
        size_average=True, reduce=True, imsize = 256, **kwargs):
        super().__init__()
        # self.nstack = nstack
        self.generateHm = heatmapGenerator
        self.dest_size = float(dest_size)

        self.size_average = size_average
        self.reduce = reduce

        self.predict2Heatmap = Predict2Heatmap(num_keypoints=heatmapGenerator.out_channels)
        self.imsize = imsize # 256


    def weighted_mse_loss(self, input, target, weights=None):
        out = (input - target) ** 2
        if weights is not None:
            out = out * weights
        if self.size_average:
            return out.sum() / len(input)
        else:
            return out.sum()

    def weighted_l1_loss(self, input, target, weights=None):
        out = torch.abs(input - target)
        if weights is not None:
            out = out * weights
        if self.size_average:
            return out.sum() / len(input)
        else:
            return out.sum()

    def jointLocationLoss(self, preds, gt_keypoints, L1=True):
        # gt_joints_vis = args[1]
        # num_joints = int(gt_joints_vis.shape[1] / 3)
        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]
        # ph.plotTensor(preds[:1,].detach().cpu())
        pred_jts = self.predict2Heatmap.softmax_integral_tensor(preds, hm_width, hm_height)
        # print("L1 Joint: ", pred_jts)
        # print("gt: ", gt_keypoints)
        if L1:
            return self.weighted_l1_loss(pred_jts, gt_keypoints)
        else:
            return self.weighted_mse_loss(pred_jts, gt_keypoints)

    def forward(self, hm_preds, targets, rois=None, keypoint_matched_idxs=None):
        """Assume that all input image size is equal in width and height

        Arguments:
            hm_preds 
                (list(torch.Tensor)) list[N C H W], len() == Stack
                or torch.Tensor  N C H W
            targets 
                (list(dict))
                or list(torch.Tensor) 
            # image_sizes (list(torch.Tensor))
            # keypoints(list(3d array))  Tensor.size(N, K, 3)
        """
        # print(hm_preds.device, keypoints.device)
        loss_dict = {}
        if type(hm_preds) is list:
            S = len(hm_preds)
            N, K, H, W = hm_preds[0].shape
            device = hm_preds[0].device
            ratio = self.dest_size / self.imsize
            keypoints = np_translateKP(targets, N, K) * ratio
            # keypoints = self.generateHm(keypoints, ratio) # B C H W
            keypoints = torch.as_tensor(keypoints, device=device)
            # keypoints /= H
            # print("--------------")
            for i in range(S-1, -1, -1):
                preds = hm_preds[i] # B C H W
                # ph.plotArray()
                # print(preds.shape, keypoints)
                hmloss = self.jointLocationLoss(preds, keypoints)
                # print("Test", i , hmloss, "\n", keypoints)
                loss_dict[f"hmloss_{i}"] = hmloss.mean()
        else:
            N, K, H, W = hm_preds.shape
            device = hm_preds.device
            assert isinstance(hm_preds, torch.Tensor)
            assert rois is not None
            for target, roi in zip(targets, rois):
                target = target[0, :, :2] # keypoint (x, y)
                target[:, 0] -= roi[0][0]
                target[:, 1] -= roi[0][1]
                target[:, 0] = target[:, 0]# * self.dest_size / (roi[0][2]-roi[0][0])
                target[:, 1] = target[:, 1]# * self.dest_size / (roi[0][3]-roi[0][1])
            keypoints = torch.stack(targets).to(device).squeeze(1)
            keypoints = keypoints[:, :, :2] * (roi[0][2]-roi[0][0]) / self.dest_size
            # print(hm_preds.shape, keypoints.shape)
            hmloss = self.jointLocationLoss(hm_preds, keypoints)
            loss_dict["hmloss_0"] = hmloss.mean()
            ###  heatmap method @Unused
            # keypoints = self.generateHm(keypoints, ratio) # B C H W
            # # ph.plotHeatmap(hms)
            # hms = torch.as_tensor(hms)
            # # ph.plotTensor(hms)
            # hms = hms.to(device) # B C H W
            # for i in range(S-1, -1, -1):
            #     preds = combined_hm_preds[i] # B C H W
            #     # hmloss = self._forward(preds, hms)
            #     loss_dict[f"hmloss_{i}"] = hmloss.mean()
        return loss_dict
class CrossEntropyHeatmapLoss(HeatmapLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _heatmapCompare(self, pred, gt):
        """L2
        """
        # l shape: B C H W
        # print("Loss: ", pred.shape, gt.shape) # B 4 64 64
        l = torch.sqrt((pred - gt)**2)
        # l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)           
        return l ## l of dim bsize
    def forward(self, ):
        S = len(hm_preds)
        N, K, H, W = hm_preds[0].shape
        device = hm_preds[0].device

        ratio = self.dest_size / self.imsize
        loss_dict = {}
        keypoints = [target["keypoints"] for target in targets]
        heatmaps = []
        roi = torch.as_tensor([[0., 0., self.dest_size, self.dest_size]]).to(device)
        for kps in keypoints:
            heatmaps_per_image, valid_per_image = keypoints_to_heatmap(kps, roi, imsize)
            heatmaps.append(heatmaps_per_image.view(-1))

        keypoint_targets = torch.cat(heatmaps, dim=0)
        # print(keypoint_targets.shape)
        for i in range(S):
            pred = hm_preds[i] # B C H W
            # print(pred.shape, N * K, H * W)
            keypoint_logits = pred.view(N * K, H * W)
            hmloss = F.cross_entropy(keypoint_logits, keypoint_targets)
            loss_dict[f"hmloss_{i}"] = hmloss
        return loss_dict

# define label
def generate_joint_location_label(config, patch_width, patch_height, joints, joints_vis):
    joints[:, 0] = joints[:, 0] / patch_width - 0.5
    joints[:, 1] = joints[:, 1] / patch_height - 0.5
    joints[:, 2] = joints[:, 2] / patch_width

    joints = joints.reshape((-1))
    joints_vis = joints_vis.reshape((-1))
    return joints, joints_vis

# define label


# define result
def get_joint_location_result(config, patch_width, patch_height, preds):
    # TODO: This cause imbalanced GPU useage, implement cpu version
    hm_width = preds.shape[-1]
    hm_height = preds.shape[-2]
    if config.output_3d:
        hm_depth = hm_width
        num_joints = preds.shape[1] // hm_depth
    else:
        hm_depth = 1
        num_joints = preds.shape[1]

    pred_jts = softmax_integral_tensor(preds, num_joints, config.output_3d, hm_width, hm_height, hm_depth)
    coords = pred_jts.detach().cpu().numpy()
    coords = coords.astype(float)
    coords = coords.reshape((coords.shape[0], int(coords.shape[1] / 3), 3))
    # project to original image size
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * patch_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * patch_height
    coords[:, :, 2] = coords[:, :, 2] * patch_width
    scores = np.ones((coords.shape[0], coords.shape[1], 1), dtype=float)

    # add score to last dimension
    coords = np.concatenate((coords, scores), axis=2)

    return coords

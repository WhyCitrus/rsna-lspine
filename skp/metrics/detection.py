import numpy as np
import torch
import torchmetrics as tm 

from torchvision.ops import box_iou


class mAP_Simple(tm.Metric):

    def __init__(self, cfg, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.cfg = cfg 

        self.add_state("p_boxes", default=[], dist_reduce_fx=None)
        self.add_state("p_scores", default=[], dist_reduce_fx=None)
        self.add_state("p_labels", default=[], dist_reduce_fx=None)
        self.add_state("t_boxes", default=[], dist_reduce_fx=None)
        self.add_state("t_labels", default=[], dist_reduce_fx=None)


    def update(self, p, t):
        self.p_boxes.extend([_["boxes"] for _ in p])
        self.p_scores.extend([_["scores"] for _ in p])
        self.p_labels.extend([_["labels"] for _ in p])
        self.t_boxes.extend([_["boxes"] for _ in t])
        self.t_labels.extend([_["labels"] for _ in t])

    def compute(self):
        # p is a list of dicts with keys "boxes", "scores", and "labels"
        # t is a list of dicts with keys "boxes", "labels"
        # each element in the list represents an image
        tp = {i: 0 for i in range(5)}
        fp = {i: 0 for i in range(5)}
        fn = {i: 0 for i in range(5)}
        for big_tuple in zip(self.p_boxes, self.p_scores, self.p_labels, self.t_boxes, self.t_labels):
            pi_box, pi_score, pi_label, ti_box, ti_label = [_.cpu() for _ in big_tuple]
            for each_level in range(5):
                if each_level not in ti_label:
                    # if foramen at that level not in ground truth
                    # then if it is in prediction, it is false positive
                    # use 0.5 as threshold
                    fp[each_level] += (pi_label[pi_score >= 0.5] == each_level).sum().item()
                else:
                    tmp_gt = ti_box[ti_label == each_level]
                    assert len(tmp_gt) == 1, f"tmp_gt is length {len(tmp_gt)}" # there should only ever be a max of 1 box for each level
                    tmp_pred_boxes = pi_box[pi_score >= 0.5]
                    tmp_pred_labels = pi_label[pi_score >= 0.5]
                    tmp_pred_boxes = tmp_pred_boxes[tmp_pred_labels == each_level]
                    if len(tmp_pred_boxes) == 0:
                        # false negatives
                        fn[each_level] += len(tmp_gt) 
                    else:
                        # calculate IoU of gt boxes and predicted boxes for that level
                        tmp_ious = box_iou(tmp_pred_boxes, tmp_gt)
                        # if len(tmp_pred_boxes) is X and len(tmp_gt) is Y
                        # then tmp_ious.shape = (X, Y)
                        # and IoU between tmp_pred_boxes[i] and tmp_gt[j] is tmp_ious[i, j]
                        # again, there should only be one gt box
                        if (tmp_ious[:, 0] >= 0.5).sum().item() > 0:
                            tp[each_level] += 1
                        # any additional preds above 0.5 are FPs
                        fp[each_level] += (tmp_ious[:, 0] >= 0.5).sum().item() - 1
                        # not meeting threshold are FPs
                        fp[each_level] += (tmp_ious < 0.5).sum().item()
        map_dict = {f"map{each_level}": tp[each_level] / (tp[each_level] + fp[each_level] + fn[each_level]) for each_level in range(5)}
        map_dict["map_mean"] = np.mean([v for v in map_dict.values()])
        return map_dict


class CenterDiffAbs(mAP_Simple):

    def compute(self):
        diff_dict = {f"{i}_x": [] for i in range(5)} 
        diff_dict.update({f"{i}_y": [] for i in range(5)})

        for big_tuple in zip(self.p_boxes, self.p_scores, self.p_labels, self.t_boxes, self.t_labels):
            pi_box, pi_score, pi_label, ti_box, ti_label = [_.cpu().numpy() for _ in big_tuple]
            for each_level in range(5):
                if each_level not in ti_label:
                    continue
                else:
                    tmp_gt = ti_box[ti_label == each_level]
                    assert len(tmp_gt) == 1, f"tmp_gt is length {len(tmp_gt)}" # there should only ever be a max of 1 box for each level
                    tmp_gt = tmp_gt[0]
                    gt_xc, gt_yc = (tmp_gt[0] + tmp_gt[2]) / 2, (tmp_gt[1] + tmp_gt[3]) / 2
                    # get predicted box with highest score at this level
                    tmp_pred = pi_box[pi_label == each_level]
                    if len(tmp_pred) == 0 or each_level not in pi_label:
                        # basically assume (0, 0) center coord if no box predicted which is pretty harsh penalty, may need to adjust
                        # but we compute median instead of mean so it should help mitigate outliers
                        diff_dict[f"{each_level}_x"].append(gt_xc)
                        diff_dict[f"{each_level}_y"].append(gt_yc)
                        continue 
                    tmp_pred = tmp_pred[0] # highest scored box, since they should be sorted in descending order by score
                    # print(pi_box, pi_label, tmp_pred)
                    assert tmp_gt.ndim == tmp_pred.ndim == 1, f"tmp_gt.ndim is {tmp_gt.ndim} and tmp_pred.ndim is {tmp_pred.ndim}"
                    assert tmp_gt.shape[0] == tmp_pred.shape[0] == 4
                    p_xc, p_yc = (tmp_pred[0] + tmp_pred[2]) / 2, (tmp_pred[1] + tmp_pred[3]) / 2
                    diff_dict[f"{each_level}_x"].append(np.abs(gt_xc - p_xc))
                    diff_dict[f"{each_level}_y"].append(np.abs(gt_yc - p_yc))
        diff_dict = {k: np.median(v) for k, v in diff_dict.items()}
        diff_dict["avg_diff"] = np.mean([v for v in diff_dict.values()])
        diff_dict["avg_x"] = np.mean([v for k, v in diff_dict.items() if "_x" in k])
        diff_dict["avg_y"] = np.mean([v for k, v in diff_dict.items() if "_y" in k])
        return diff_dict

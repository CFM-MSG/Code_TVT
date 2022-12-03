import numpy as np


# metric for sake
# programed by numpy
def accuracy(output, target, topk=(1,)):  # compute accuracy for one modality
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    print(target.view(1, -1).expand_as(pred))
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def sake_metric(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores,
                k=None):
    if k == None:
        k = {'precision': 100, 'map': predicted_features_gallery.shape[0]}
    if k['precision'] == None:
        k['precision'] = 100
    if k['map'] == None:
        k['map'] = predicted_features_gallery.shape[0]
    gt_labels_query = gt_labels_query.flatten()
    gt_labels_gallery = gt_labels_gallery.flatten()
    str_sim = np.expand_dims(gt_labels_query, axis=1) == np.expand_dims(gt_labels_gallery, axis=0) * 1
    aps = map_sake(predicted_features_gallery, gt_labels_gallery,
                   predicted_features_query, gt_labels_query, scores, k=k['map'])
    prec = prec_sake(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query,
                     scores, k=k['precision'])
    return aps, prec, scores, str_sim


def map_sake(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=None):
    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    mean_mAP = []
    for fi in range(predicted_features_query.shape[0]):
        mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=k)
        mAP_ls[gt_labels_query[fi]].append(mapi)
        mean_mAP.append(mapi)
    for i in range(len(mAP_ls)):
        mAP_ls[i] = np.mean(mAP_ls[i])
    # print("map for all classes: ", np.nanmean(mean_mAP))
    print(mAP_ls)
    print('top 10 maximal map: ', np.argsort(-np.array(mAP_ls)))
    return mean_mAP


def prec_sake(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=None):
    # compute precision for two modalities
    prec_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    mean_prec = []
    for fi in range(predicted_features_query.shape[0]):
        prec = eval_precision(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=k)
        prec_ls[gt_labels_query[fi]].append(prec)
        mean_prec.append(prec)
    # print("precision for all samples: ", np.nanmean(mean_prec))
    return np.nanmean(mean_prec)


def eval_AP_inner(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]  # total retrieved samples
    tot_pos = np.sum(pos_flag)  # total true position

    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]  # sorted true positive
    fp = np.logical_not(tp)  # sorted false positive

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]  # select top-k true position
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)
    return ap


def VOCap(rec, prec):
    mrec = np.append(0, rec)  # put 0 in the first element
    mrec = np.append(mrec, 1)  # put 1 in the last element

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):  # sort mpre, the smaller, the latter
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap


def eval_precision(inst_id, scores, gt_labels, top=100):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]

    top = min(top, tot)

    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top]) / top

import torch

def classification_with_tr_mse(pred_logit,
                               gt_label,
                               tr_prediction, #translation and rotation vector (1,2) or (2,)
                               gt_tr):

    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_logit, gt_label)
    #Use gt_label to mask out the MSE loss for the regressor
    bool_mask = torch.gt(gt_label,0)[:, None]    #Create bool mask from 0s and 1s
    masked_tr_prediction = torch.masked_select(tr_prediction, bool_mask).view(-1,tr_prediction.shape[1])
    masked_gt_tr = torch.masked_select(gt_tr, bool_mask).view(-1,gt_tr.shape[1])
    mse_loss = torch.nn.functional.mse_loss(masked_tr_prediction, masked_gt_tr)
    return classification_loss, mse_loss


def classification_BCE_with_logits(pred_logit, gt_label, device, use_pos_weight=True):
    if use_pos_weight:
        num_pos = gt_label.sum().item()
        num_neg = len(gt_label)-num_pos
        pos_weight = torch.tensor([num_neg/num_pos], dtype=torch.float32).to(device)
        classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logit, gt_label, pos_weight=pos_weight)      
    else:
        classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logit, gt_label)  
    return classification_loss
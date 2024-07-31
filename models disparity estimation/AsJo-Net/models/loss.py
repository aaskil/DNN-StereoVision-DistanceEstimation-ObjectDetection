import torch.nn.functional as F
import torch

def model_loss_train(disp_ests, disp_gts, img_masks):
    weights = [1.0, 0.3]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(disp_ests, disp_gts, weights, img_masks):
        # Check for NaN values in inputs
        assert not torch.isnan(disp_est).any(), "NaN in disp_est before mask"
        assert not torch.isnan(disp_gt).any(), "NaN in disp_gt before mask"

        # Ensure masks are not entirely false
        assert mask_img.any(), "Empty mask, no True values"

        # Ensure masked tensors are not empty
        masked_disp_est = disp_est[mask_img]
        masked_disp_gt = disp_gt[mask_img]
        assert masked_disp_est.nelement() > 0, "Empty tensor for disp_est after applying mask"
        assert masked_disp_gt.nelement() > 0, "Empty tensor for disp_gt after applying mask"
        
        # Check for infinite values in inputs
        assert not torch.isinf(masked_disp_est).any(), "Inf in masked disp_est"
        assert not torch.isinf(masked_disp_gt).any(), "Inf in masked disp_gt"

        loss = F.smooth_l1_loss(masked_disp_est, masked_disp_gt, reduction='mean')

        # Check if loss calculation resulted in NaN
        assert not torch.isnan(loss).any(), "NaN in calculated loss"

        all_losses.append(weight * loss)
    return sum(all_losses)

def model_loss_test(disp_ests, disp_gts,img_masks):
    weights = [1.0]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(disp_ests, disp_gts, weights, img_masks):
        all_losses.append(weight * F.l1_loss(disp_est[mask_img], disp_gt[mask_img], size_average=True))
    return sum(all_losses)  
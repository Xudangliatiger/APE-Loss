import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class APELoss(nn.Module):

    def __init__(self, lamb=4, topk=100000, loss_weight=1):
        """
        Args:
         lamb: parameter of sigmoid
         topk: parameter to save GPU memories.
         loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(APELoss, self).__init__()
        self.lamb = lamb
        # To avoid a huge number of samples in a batch,
        # AP loss ignores the input x of H(x) that is smaller than -\delta, since when x< -\delta, H(x) = 0.
        # After replacing H(x) with S(x), we can also ignore input x when x is very small;
        #   \lambda     \delta    threshold    AP of S(x)/H(x)
        #    4           1           -1         36.9/37.0
        #    8           0.5         -0.5       37.3/37.4
        #    16          0.25        -0.25      36.5/36.8
        # Here we simply use -4/self.lamb so that H(x) and S(x) can have same thresholds.
        # You can use a smaller th if you have a large GPU memory.
        self.th = -4/self.lamb
        self.topk = topk
        self.loss_weight = loss_weight

    def forward(self, logits, targets, ious, delta=1., eps=1e-5):
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)
        ious = ious

        # Get valid bg logits, ignore the distances smaller than th
        p_min, index = torch.min(fg_logits, 0)
        relevant_bg_labels = ((targets == 0) & (logits >= p_min + self.th))
        relevant_bg_logits = logits[relevant_bg_labels]

        # Use topk to save the GPU memory; you can use a larger topk if you have more GPU memories.
        if relevant_bg_logits.numel()> self.topk:
            relevant_bg_logits, _ = relevant_bg_logits.topk(self.topk, dim=0, largest=True, sorted=False)

        ranking_loss = []
        # Loops over each Positive Sample following the order
        for ii in range(fg_num):

            # Select Complete Negative Samples
            neg_idx = (relevant_bg_logits > (fg_logits[ii]+self.th))


            pos_idx = (ious < ious[ii]) & (fg_logits > (fg_logits[ii]+self.th))
            adaptive_neg_logits = torch.cat((relevant_bg_logits[neg_idx],fg_logits[pos_idx]),0)

            # Calculate the Sigmoid Distance between the Positive (i.e., ii) and Negative Samples
            FP_dis = torch.sigmoid_((adaptive_neg_logits-fg_logits[ii])*self.lamb)

            # Balance Constant
            TP_idx = (ious >= ious[ii]) & (fg_logits > (fg_logits[ii]+self.th))
            TP_dis = torch.sigmoid_((fg_logits[TP_idx]-fg_logits[ii])*self.lamb)
            rank = torch.sum(FP_dis)+torch.sum(TP_dis)
            rank = rank.clone().detach_()

            if FP_dis.numel():
                # Distance Function employing a simple CE loss
                zeros = torch.zeros_like(FP_dis)
                distances = F.binary_cross_entropy(FP_dis, zeros, reduction='sum')

                # Pair-wise Error = Distance Function/Balance Constant
                ranking_loss.append(distances.view(-1)*ious[ii] /rank)
                # Here we use IoU weight following arxiv:1908.05641

        ranking_loss = torch.cat(ranking_loss, dim=0)
        return self.loss_weight*ranking_loss.mean()/self.lamb
        # We divide loss by lambda since lambda is in the loss gradients due to chain rule, refer to the Appendix of ARPS.
        # After dividing, CE loss should have the same gradient as that of Error-Driven Update.
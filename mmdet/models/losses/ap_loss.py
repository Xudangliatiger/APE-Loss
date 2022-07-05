import torch


class APLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ious=None, delta=0.5, lamb = 8, soft = True):
        classification_grads = torch.zeros(logits.shape).cuda()

        # Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        # Do not use bg with scores less than minimum fg logit
        # since changing its score does not have an effect on precision
        if soft == True:
            threshold_logit = torch.min(fg_logits)-4/lamb
        else:
            threshold_logit = torch.min(fg_logits)-delta

        # Get valid bg logits
        relevant_bg_labels = ((targets == 0) & (logits >= threshold_logit))
        relevant_bg_logits = logits[relevant_bg_labels]
        relevant_bg_grad = torch.zeros(len(relevant_bg_logits)).cuda()
        rank = torch.zeros(fg_num).cuda()
        prec = torch.zeros(fg_num).cuda()
        fg_grad = torch.zeros(fg_num).cuda()

        max_prec = 0
        # sort the fg logits
        order = torch.argsort(fg_logits)
        # Loops over each positive following the order
        for ii in order:
            # if soft == True:
            #     threshold_ii = fg_logits[ii]- 0.25 #4 / lamb
            # else:
            #     threshold_ii = fg_logits[ii]-delta
            # #
            # ii_bg_idx = (relevant_bg_logits > threshold_ii)
            # #N_neg = ii_bg_idx.numel()
            # # ii_bg_logits = relevant_bg_logits[ii_bg_idx]
            # bg_relations_th = relevant_bg_logits[ii_bg_idx]-fg_logits[ii]
            # # Apply piecewise linear function and determine relations with bgs
            # if soft == True:
            #     bg_relations_th = torch.sigmoid_(bg_relations_th * lamb)
            # else:
            #     bg_relations_th = torch.clamp(bg_relations_th / (2 * delta)+0.5, min=0, max=1)


            # x_ij s as score differences with fgs
            fg_relations = fg_logits-fg_logits[ii]
            # Apply piecewise linear function and determine relations with fgs
            if soft ==True:
                fg_relations = torch.sigmoid_(fg_relations * lamb)
            else:
                fg_relations = torch.clamp(fg_relations / (2 * delta)+0.5, min=0, max=1)


            # Discard i=j in the summation in rank_pos
            fg_relations[ii] = 0

            # x_ij s as score differences with bgs
            bg_relations = relevant_bg_logits-fg_logits[ii]
            # Apply piecewise linear function and determine relations with bgs
            if soft ==True:
                bg_relations = torch.sigmoid_(bg_relations * lamb)
            else:
                bg_relations = torch.clamp(bg_relations / (2 * delta)+0.5, min=0, max=1)

            # BC
            bc = rank[ii]
            # rank_pos + FP_num_th

            # DENS
            # valid = (ious < ious[ii])  # & (fg_logits >= threshold_ii)
            # fg_error = fg_logits[valid]-fg_logits[ii]
            # if soft == True:
            #     fg_error = torch.sigmoid_(fg_error * lamb)
            # else:
            #     fg_error = torch.clamp(fg_error / (2 * delta)+0.5, min=0, max=1)



            # Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos = 1 + torch.sum(fg_relations) #- torch.sum(fg_error)
            FP_num = torch.sum(bg_relations) #+ torch.sum(fg_error)
            #FP_num_th = torch.sum(bg_relations_th)
            # Store the total since it is normalizer also for aLRP Regression error
            rank[ii] = rank_pos + FP_num

            # Compute precision for this example
            current_prec = rank_pos / rank[ii]



            # Compute interpolated AP and store gradients for relevant bg examples
            if (max_prec <= current_prec):
                max_prec = current_prec
                relevant_bg_grad += (bg_relations / bc)
                fg_grad[ii] -= (FP_num/bc)
                #fg_grad[valid] += (fg_error/bc)
            else:
                relevant_bg_grad += (bg_relations / bc) * (((1-max_prec) / (1-current_prec)))
                fg_grad[ii] -= (FP_num / bc)* (((1-max_prec) / (1-current_prec)))
                #fg_grad[valid] += (fg_error / bc)* (((1-max_prec) / (1-current_prec)))
            # Store fg gradients

            prec[ii] = max_prec

            # aLRP with grad formulation fg gradient
        classification_grads[fg_labels] = fg_grad
        # aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels] = relevant_bg_grad

        classification_grads /= fg_num

        cls_loss = 1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss

    @staticmethod
    def backward(ctx, out_grad1):
        g1, = ctx.saved_tensors
        return g1 * out_grad1, None, None
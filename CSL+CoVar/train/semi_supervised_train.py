from util.recorder import Recorder
from .supervised_train import SupervisedModule
from torch import nn
import torch
import os
import csv
import matplotlib
# Use a non-interactive backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from util.PCOS import batch_class_stats, get_max_confidence_and_residual_variance
from util.classes import CLASSES

class SemiModule(SupervisedModule):   
    def __init__(self, batch_iters, alpha, nclass, **kwargs):
        # Extract clustering selection configs (do not forward to super)
        self.select_mode = kwargs.pop('select_mode', 'neglog')
        self.select_lam = kwargs.pop('select_lam', 0.5)
        super(SemiModule, self).__init__(nclass=nclass, **kwargs)
        self.batch_iters = batch_iters
        self.avg_log_interval = batch_iters // 8
        self.alpha = alpha
        self.num_classes = nclass
        # counters to keep track of pseudo-label pixels produced and selected during training
        # these are local to the process; on distributed training we aggregate in on_train_end
        self.total_pseudo_pixels = 0
        self.total_selected_pixels = 0
        # per-epoch counters (reset at epoch end)
        self.epoch_pseudo_pixels = 0
        self.epoch_selected_pixels = 0
        self.criterion_u = nn.CrossEntropyLoss(reduction='none')
        self.save_hyperparameters(ignore=['model'])

        self.loss_recorder = Recorder('total_loss', self.log, True)
        self.loss_x_recorder = Recorder('loss_x', self.log, False)
        self.loss_u_s_recorder = Recorder('loss_u_s', self.log, False)
        self.loss_u_m_recorder = Recorder('loss_u_m', self.log, False)
        self.loss_u_fp_recorder = Recorder('loss_u_fp', self.log, False)
        self.mask_ratio_recorder = Recorder('mask_ratio', self.log, False)
        self.recorders = [
            self.loss_recorder,
            self.loss_x_recorder,
            self.loss_u_s_recorder,
            self.loss_u_m_recorder,
            self.loss_u_fp_recorder,
            self.mask_ratio_recorder,
        ]
        # tracking for plotting: choose one random epoch to produce a scatter (seeded for reproducibility)
        self._chosen_scatter_epoch = None
        self._scatter_created = False
        self._scatter_candidate_count = 0
        self._scatter_buffer = None  # will hold (rcv_array, mc_array, sel_array)
        self._rng = random.Random(42)
        # record per-epoch progress for final line plot
        self._epochs = []
        self._fractions = []
        # per-epoch reservoir sampling buffers (store up to _epoch_sample_size points per epoch)
        self._epoch_sample_size = 100000
        self._epoch_seen_count = 0
        self._epoch_rcv_buf = []
        self._epoch_mc_buf = []
        self._epoch_sel_buf = []
        # per-epoch per-class counters for generated pseudo-labels and selected pseudo-labels
        # lists of length num_classes (ints)
        try:
            self._epoch_class_generated = [0] * self.num_classes
            self._epoch_class_selected = [0] * self.num_classes
        except Exception:
            # fallback to empty lists; we'll create on demand
            self._epoch_class_generated = []
            self._epoch_class_selected = []
        # Special single-shot scatter at a target epoch (only one plot)
        self._special_epoch = 1
        self._special_scatter_created = False
        # accumulate per-image rows from special_epoch .. end
        self._special_accum_rows = []  # list of tuples (epoch, img_id, dom, name, is_maj, mc_img, acc)
        
    def on_train_batch_start(self, batch, batch_idx):   
        (img_u_w_mix, _, _, ignore_mask_mix, _, _) = batch['mixed']
        with torch.no_grad():
            self.pred_u_w_mix = self(img_u_w_mix, False, False).detach()
            self.weight_u_w_mix = self.get_weight(self.pred_u_w_mix.softmax(dim=1), ignore_mask_mix, num_classes=self.num_classes)
            self.mask_u_w_mix = self.pred_u_w_mix.argmax(dim=1)


    def training_step(self, batch, batch_idx):   
        ((img_x, mask_x),
        (img_u_w, img_u_s, img_u_m, ignore_mask, cutmix_box, cutmix_box2),
        (_, img_u_s_mix, _, ignore_mask_mix, _, _)) = batch['labeled'], batch['unlabeled'], batch['mixed']

        num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
        preds, preds_fp = self(torch.cat((img_x, img_u_w)), True)
        pred_x, pred_u_w = preds.split([num_lb, num_ulb])
        pred_u_fp = preds_fp[num_lb:]

        pred_u_w = pred_u_w.detach()
        weight_u_w = self.get_weight(pred_u_w.softmax(dim=1), ignore_mask, num_classes=self.num_classes)
        mask_u_w = pred_u_w.argmax(dim=1)

        conf_mask = weight_u_w == 1
        # cutmix_box2 = conf_mask & (cutmix_box2 == 1)

        # img_u_m[cover_mask.unsqueeze(1).expand_as(img_u_m)] = 0 

        img_u_m[cutmix_box2.unsqueeze(1).expand(img_u_m.shape) == 1] = \
            img_u_s_mix[cutmix_box2.unsqueeze(1).expand(img_u_m.shape) == 1]

        img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = \
            img_u_s_mix[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]

        pred_u_s, pred_u_m = self(torch.cat((img_u_s, img_u_m)), False, False).chunk(2)

        mask_u_w_cutmixed, weight_u_w_cutmixed, ignore_mask_cutmixed = \
            mask_u_w.clone(), weight_u_w.clone(), ignore_mask.clone()

        mask_u_w_cutmixed2, weight_u_w_cutmixed2, ignore_mask_cutmixed2 = \
            mask_u_w.clone(), weight_u_w.clone(), ignore_mask.clone()

        mask_u_w_cutmixed[cutmix_box == 1] = self.mask_u_w_mix[cutmix_box == 1]
        weight_u_w_cutmixed[cutmix_box == 1] = self.weight_u_w_mix[cutmix_box == 1]
        ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask_mix[cutmix_box == 1]

        mask_u_w_cutmixed2[cutmix_box2 == 1] = self.mask_u_w_mix[cutmix_box2 == 1]
        weight_u_w_cutmixed2[cutmix_box2 == 1] = self.weight_u_w_mix[cutmix_box2 == 1]
        ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

        loss_x = self.criterion_l(pred_x, mask_x)
        loss_u_s = self.semi_loss(pred_u_s, mask_u_w_cutmixed, weight_u_w_cutmixed, ignore_mask_cutmixed)
        loss_u_m = self.semi_loss(pred_u_m, mask_u_w_cutmixed2, weight_u_w_cutmixed2, ignore_mask_cutmixed2)
        loss_u_fp = self.semi_loss(pred_u_fp, mask_u_w, weight_u_w, ignore_mask)
        
        loss = (loss_x + loss_u_s * 0.25 + loss_u_m * 0.25 + loss_u_fp * 0.5) / 2.0
        
        mask_ratio = (conf_mask & (ignore_mask != 255)).sum().item() / \
            (ignore_mask != 255).sum()
        
        self.loss_recorder(loss.item())
        self.loss_x_recorder(loss_x.item())
        self.loss_u_s_recorder(loss_u_s.item())
        self.loss_u_m_recorder(loss_u_m.item())
        self.loss_u_fp_recorder(loss_u_fp.item())
        self.mask_ratio_recorder(mask_ratio)

        # ---- accumulate pseudo-label counts for diagnostics ----
        try:
            # valid pixels in unlabeled batch
            valid_mask = (ignore_mask != 255)
            batch_pseudo = int(valid_mask.sum().item())
            # selected pseudo-label pixels where conf_mask==True and valid
            batch_selected = int((conf_mask & valid_mask).sum().item())
        except Exception:
            batch_pseudo = 0
            batch_selected = 0
        # accumulate both total and per-epoch counters
        self.total_pseudo_pixels += batch_pseudo
        self.total_selected_pixels += batch_selected
        self.epoch_pseudo_pixels += batch_pseudo
        self.epoch_selected_pixels += batch_selected

        # For each batch, perform reservoir sampling across pixels to collect up to
        # _epoch_sample_size points per epoch (rcv, mc, selected). This allows immediate
        # per-epoch scatter plotting at epoch end without writing large CSVs.
        try:
            epoch_idx = int(getattr(self, 'current_epoch', 0))
        except Exception:
            epoch_idx = 0

        if batch_pseudo > 0:
            try:
                probs = pred_u_w.softmax(dim=1)
                max_conf, scaled_res_var = get_max_confidence_and_residual_variance(probs, valid_mask, self.num_classes)
                mc_vals = max_conf[valid_mask].detach().cpu().numpy().ravel()
                rcv_vals = scaled_res_var[valid_mask].detach().cpu().numpy().ravel()
                sel_mask = (conf_mask & valid_mask)[valid_mask].detach().cpu().numpy().ravel().astype(int)

                # accumulate per-class generated and selected counts for this batch
                try:
                    labels_np = mask_u_w[valid_mask].detach().cpu().numpy().ravel()
                    if labels_np.size > 0:
                        gen_counts = np.bincount(labels_np, minlength=self.num_classes)
                        sel_counts = np.bincount(labels_np, weights=sel_mask, minlength=self.num_classes)
                        # ensure buffers initialized
                        if len(self._epoch_class_generated) != self.num_classes:
                            self._epoch_class_generated = [0] * self.num_classes
                            self._epoch_class_selected = [0] * self.num_classes
                        for ci in range(self.num_classes):
                            self._epoch_class_generated[ci] += int(gen_counts[ci])
                            self._epoch_class_selected[ci] += int(sel_counts[ci])
                except Exception:
                    pass

                # iterate a small random subset of pixels per batch to reduce per-batch work
                m = mc_vals.shape[0]
                if m > 0:
                    # number to sample from this batch: min(m, _epoch_sample_size)
                    k = min(m, self._epoch_sample_size)
                    try:
                        # use python RNG to sample indices without replacement
                        chosen_idx = self._rng.sample(range(m), k)
                    except Exception:
                        chosen_idx = list(range(k))

                    for idx in chosen_idx:
                        self._epoch_seen_count += 1
                        if len(self._epoch_rcv_buf) < self._epoch_sample_size:
                            self._epoch_rcv_buf.append(float(rcv_vals[idx]))
                            self._epoch_mc_buf.append(float(mc_vals[idx]))
                            self._epoch_sel_buf.append(int(sel_mask[idx]))
                        else:
                            # reservoir replace probability
                            r = self._rng.randint(1, self._epoch_seen_count)
                            if r <= self._epoch_sample_size:
                                replace_pos = r - 1
                                self._epoch_rcv_buf[replace_pos] = float(rcv_vals[idx])
                                self._epoch_mc_buf[replace_pos] = float(mc_vals[idx])
                                self._epoch_sel_buf[replace_pos] = int(sel_mask[idx])
                    # (No special per-pixel reservoir maintained anymore; per-image aggregation is used at epoch end)
            except Exception:
                pass

        if self.trainer.is_global_zero and batch_idx % self.avg_log_interval == 0:
            msg = f'Iters: {self.global_step}'
            for recorder in self.recorders:
                msg += recorder.log_metrics()
            self.logging_callback.info(msg)
        return loss


    def on_train_end(self) -> None:
        """Aggregate counters across processes (if distributed) and print summary."""
        # Try to aggregate totals across ranks when using distributed training
        total_pseudo = self.total_pseudo_pixels
        total_selected = self.total_selected_pixels
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                tensor = torch.tensor([total_pseudo, total_selected], dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
                torch.distributed.barrier()
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                total_pseudo = int(tensor[0].item())
                total_selected = int(tensor[1].item())
        except Exception:
            # if distributed failing for some reason, fall back to local counts
            pass

        # --- Create final line plot from in-memory per-epoch lists and, if reservoir sampled one batch,
        # create the scatter plot from the buffered batch. This avoids writing CSVs to disk.
        try:
            is_global_zero = getattr(self, 'trainer', None) is None or getattr(self.trainer, 'is_global_zero', True)
        except Exception:
            is_global_zero = True

        if is_global_zero:
            try:
                out_dir = None
                try:
                    out_dir = getattr(self.trainer, 'log_dir', None)
                except Exception:
                    out_dir = None
                if out_dir is None:
                    out_dir = os.path.join(os.getcwd(), 'pseudo_stats')
                os.makedirs(out_dir, exist_ok=True)

                # Line plot from in-memory lists
                if self._epochs and self._fractions:
                    try:
                        order = np.argsort(np.array(self._epochs))
                        epochs_arr = np.array(self._epochs)[order]
                        fr_arr = np.array(self._fractions)[order]
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(epochs_arr, fr_arr, marker='o', linestyle='-')
                        ax.set_xlabel('epoch')
                        ax.set_ylabel('selected / generated (fraction)')
                        ax.set_title('Pseudo-label selection fraction per epoch')
                        ax.grid(True)
                        progress_path = os.path.join(out_dir, 'pseudo_fraction_final.png')
                        fig.tight_layout()
                        fig.savefig(progress_path)
                        plt.close(fig)
                    except Exception:
                        pass

                # Scatter from buffered batch (if any)
                if (not self._scatter_created) and (self._scatter_buffer is not None):
                    try:
                        rcv_vals, mc_vals, sel_mask = self._scatter_buffer
                        rcv_a = np.array(rcv_vals)
                        mc_a = np.array(mc_vals)
                        sel_a = np.array(sel_mask)
                        if rcv_a.size and mc_a.size:
                            fig2, ax2 = plt.subplots(figsize=(6, 6))
                            ax2.scatter(rcv_a[sel_a == 0], mc_a[sel_a == 0], s=6, c='blue', label='unselected', alpha=0.6)
                            if (sel_a == 1).any():
                                ax2.scatter(rcv_a[sel_a == 1], mc_a[sel_a == 1], s=6, c='red', label='selected', alpha=0.6)
                            ax2.set_xlabel('RCV (residual class variance)')
                            ax2.set_ylabel('MC (max confidence)')
                            ax2.set_title(f'Pseudo samples scatter (epoch={self._chosen_scatter_epoch})')
                            ax2.legend(markerscale=3)
                            ax2.grid(True)
                            scatter_path = os.path.join(out_dir, f'pseudo_scatter_epoch_{self._chosen_scatter_epoch}.png')
                            fig2.tight_layout()
                            fig2.savefig(scatter_path)
                            plt.close(fig2)
                            self._scatter_created = True
                    except Exception:
                        pass
            except Exception:
                try:
                    if hasattr(self, 'logging_callback') and self.logging_callback is not None:
                        self.logging_callback.warning(f"Failed to create pseudo plots")
                except Exception:
                    pass

            # If we accumulated special per-image rows from _special_epoch onwards, create a combined plot
            try:
                if hasattr(self, '_special_accum_rows') and len(self._special_accum_rows) > 0:
                    try:
                        # build arrays
                        epochs_list = [int(r[0]) for r in self._special_accum_rows]
                        img_ids = [r[1] for r in self._special_accum_rows]
                        doms = [int(r[2]) for r in self._special_accum_rows]
                        names = [r[3] for r in self._special_accum_rows]
                        is_majs = [int(r[4]) for r in self._special_accum_rows]
                        mcs = [float(r[5]) for r in self._special_accum_rows]
                        accs = [float(r[6]) for r in self._special_accum_rows]

                        # colors: majority -> pink-ish, minority -> blue-ish
                        colors = [(230/255,169/255,177/255) if im == 1 else (112/255,153/255,211/255) for im in is_majs]

                        # plot combined scatter
                        figc, axc = plt.subplots(figsize=(7, 6))
                        axc.scatter(accs, mcs, c=colors, s=10, alpha=0.6)
                        axc.set_xlabel('pseudo-label accuracy (per-image)')
                        axc.set_ylabel('mean max confidence (per-image)')
                        axc.set_title(f'Combined special pseudo scatter (epochs >= {getattr(self, "_special_epoch", 1)})')
                        axc.grid(True)
                        combined_png = os.path.join(out_dir, f'special_pseudo_scatter_epochs_{getattr(self, "_special_epoch", 1)}_to_end.png')
                        figc.tight_layout()
                        figc.savefig(combined_png)
                        plt.close(figc)

                        # save combined text file
                        combined_txt = os.path.join(out_dir, f'special_pseudo_scatter_epochs_{getattr(self, "_special_epoch", 1)}_to_end.txt')
                        try:
                            with open(combined_txt, 'w') as f:
                                f.write(f"# n_points\t{len(accs)}\n")
                                f.write('epoch\timg_id\tdom_class_idx\tdom_class_name\tis_majority\tmean_mc\taccuracy\n')
                                for ep, img_id, dom, name, is_maj, mc_img, acc in self._special_accum_rows:
                                    f.write(f"{ep}\t{img_id}\t{dom}\t{name}\t{is_maj}\t{mc_img:.6e}\t{acc:.6e}\n")
                        except Exception:
                            pass

                        try:
                            msg3 = f"Wrote combined special scatter: points={len(accs)}, saved={combined_png}"
                            if hasattr(self, 'logging_callback') and self.logging_callback is not None:
                                self.logging_callback.info(msg3)
                            else:
                                print(msg3)
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass

    @torch.no_grad()
    def get_weight(self, pred, ignore, num_classes, epsilon=1e-8, alpha=2.0):
        weight_mask = torch.zeros_like(ignore, device=ignore.device)
        valid_mask = (ignore != 255)

        max_confidence, scaled_residual_variance = get_max_confidence_and_residual_variance(
            pred, valid_mask, num_classes, epsilon
        )
        means, vars = batch_class_stats(
            max_confidence,
            scaled_residual_variance,
            num_classes,
            select_mode=getattr(self, 'select_mode', 'neglog'),
            lam=getattr(self, 'select_lam', 0.25),
        )
        conf_mean = means[:, 0].view(-1, 1, 1)  
        res_mean = means[:, 1].view(-1, 1, 1)  
        conf_var = vars[:, 0].view(-1, 1, 1)  
        res_var = vars[:, 1].view(-1, 1, 1)   

        conf_z = (max_confidence - conf_mean) / torch.sqrt(conf_var + epsilon)
        res_z = (res_mean - scaled_residual_variance) / torch.sqrt(res_var + epsilon)

        weight_conf = torch.exp(- (conf_z ** 2) / alpha) 
        weight_res = torch.exp(- (res_z ** 2) / alpha)   

        weight = weight_conf * weight_res 

        confident_mask = (conf_z > 0) | (res_z > 0) 

        weight = torch.where(confident_mask, torch.ones_like(weight), weight)
        weight_mask = torch.where(valid_mask, weight, torch.zeros_like(weight))

        return weight_mask

    
    def semi_loss(self, pred, label, weight, ignore):
        valid_mask = (ignore != 255)
        loss_high_conf = self.criterion_u(pred, label)
        loss_high_conf = loss_high_conf * weight 
        loss_u = loss_high_conf.sum() / valid_mask.sum()
        return loss_u

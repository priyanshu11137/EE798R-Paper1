import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime
import torch.nn.functional as F 
from datasets.crowd import Crowd_qnrf, Crowd_nwpu, Crowd_sh
from models import vgg19
from losses.ot_loss import OT_Loss
from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    gt_discretes = torch.stack(transposed_batch[2], 0)
    return images, points, gt_discretes


class Trainer:
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        sub_dir = f"input-{args.crop_size}_wot-{args.wot}_wtv-{args.wtv}_reg-{args.reg}_nIter-{args.num_of_iter_in_ot}_normCood-{args.norm_cood}"
        self.save_dir = os.path.join("ckpts", sub_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
        self.logger = log_utils.get_logger(os.path.join(self.save_dir, f"train-{time_str}.log"))
        log_utils.print_config(vars(args), self.logger)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise Exception("GPU is not available")

        downsample_ratio = 8
        self.args.downsample_ratio = downsample_ratio

        if args.dataset.lower() == "qnrf":
            self.datasets = {x: Crowd_qnrf(os.path.join(args.data_dir, x), args.crop_size, downsample_ratio, x) for x in ["train", "val"]}
        elif args.dataset.lower() == "nwpu":
            self.datasets = {x: Crowd_nwpu(os.path.join(args.data_dir, x), args.crop_size, downsample_ratio, x) for x in ["train", "val"]}
        elif args.dataset.lower() in ["sha", "shb"]:
            self.datasets = {
                "train": Crowd_sh(os.path.join(args.data_dir, "train_data"), args.crop_size, downsample_ratio, "train"),
                "val": Crowd_sh(os.path.join(args.data_dir, "test_data"), args.crop_size, downsample_ratio, "val")
            }
        else:
            raise NotImplementedError

        self.dataloaders = {
            x: DataLoader(
                self.datasets[x],
                collate_fn=(train_collate if x == "train" else default_collate),
                batch_size=(args.batch_size if x == "train" else 1),
                shuffle=(x == "train"),
                num_workers=args.num_workers,
                pin_memory=(x == "train")
            )
            for x in ["train", "val"]
        }
        self.model = vgg19().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            self.logger.info(f"Loading pretrained model from {args.resume}")
            checkpoint = torch.load(args.resume, self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1

        self.ot_loss = OT_Loss(args.crop_size, downsample_ratio, args.norm_cood, self.device, args.num_of_iter_in_ot, args.reg)
        self.tv_loss = nn.L1Loss(reduction="none").to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info(f"{'-'*5} Epoch {epoch}/{args.max_epoch} {'-'*5}")
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()
    def compute_hot_loss_single_image(self, outputs_i, outputs_normed_i, points_i):
        levels = [2]  
        alpha = [0.5, 0.3, 0.2]  
        total_ot_loss, total_wd, total_ot_obj_value = 0, 0, 0
        downsample_ratio = self.args.downsample_ratio

        points_i_array = points_i.clone().to(self.device)
        for l, level in enumerate(levels):
            num_cells_per_side = 2 ** level
            H_cell = outputs_i.size(2) // num_cells_per_side
            W_cell = outputs_i.size(3) // num_cells_per_side

            if H_cell == 0 or W_cell == 0:
                continue

            for grid_y in range(num_cells_per_side):
                for grid_x in range(num_cells_per_side):
                    y_start, y_end = grid_y * H_cell, (grid_y + 1) * H_cell
                    x_start, x_end = grid_x * W_cell, (grid_x + 1) * W_cell

                    outputs_cell = outputs_i[:, :, y_start:y_end, x_start:x_end]
                    outputs_normed_cell = outputs_normed_i[:, :, y_start:y_end, x_start:x_end]

                    if outputs_cell.size(2) != self.ot_loss.output_size or outputs_cell.size(3) != self.ot_loss.output_size:
                        # Resize cells to match `self.ot_loss.output_size`
                        outputs_cell = F.interpolate(outputs_cell, size=(self.ot_loss.output_size, self.ot_loss.output_size), mode='bilinear', align_corners=False)
                        outputs_normed_cell = F.interpolate(outputs_normed_cell, size=(self.ot_loss.output_size, self.ot_loss.output_size), mode='bilinear', align_corners=False)

                    y_start_img, y_end_img = y_start * downsample_ratio, y_end * downsample_ratio
                    x_start_img, x_end_img = x_start * downsample_ratio, x_end * downsample_ratio

                    in_cell_mask = (points_i_array[:, 1] >= y_start_img) & (points_i_array[:, 1] < y_end_img) & \
                                (points_i_array[:, 0] >= x_start_img) & (points_i_array[:, 0] < x_end_img)
                    points_cell = points_i_array[in_cell_mask]

                    if points_cell.shape[0] == 0:
                        continue

                    points_cell_normalized = points_cell.clone()
                    points_cell_normalized[:, 0] = (points_cell[:, 0] - x_start_img) / downsample_ratio
                    points_cell_normalized[:, 1] = (points_cell[:, 1] - y_start_img) / downsample_ratio

                    # Calculate OT loss for the current cell
                    ot_loss_cell, wd_cell, ot_obj_value_cell = self.ot_loss(outputs_normed_cell, outputs_cell, [points_cell_normalized])
                    total_ot_loss += alpha[l] * ot_loss_cell
                    total_wd += alpha[l] * wd_cell
                    total_ot_obj_value += alpha[l] * ot_obj_value_cell

        return total_ot_loss, total_wd, total_ot_obj_value


    def train_epoch(self):
        args = self.args
        epoch_ot_loss, epoch_ot_obj_value, epoch_wd = AverageMeter(), AverageMeter(), AverageMeter()
        epoch_count_loss, epoch_tv_loss, epoch_loss = AverageMeter(), AverageMeter(), AverageMeter()
        epoch_mae, epoch_mse = AverageMeter(), AverageMeter()
        epoch_start = time.time()
        self.model.train()

        for step, (inputs, points, gt_discrete) in enumerate(self.dataloaders["train"]):
            inputs, gt_discrete = inputs.to(self.device), gt_discrete.to(self.device)
            outputs, outputs_normed = self.model(inputs)

            # Initialize accumulation variables as scalars
            total_ot_loss = 0.0  
            total_count_loss = 0.0
            total_tv_loss = 0.0
            pred_counts, gd_counts = [], []

            for i in range(inputs.size(0)):
                outputs_i, outputs_normed_i, points_i = outputs[i].unsqueeze(0), outputs_normed[i].unsqueeze(0), points[i]
                ot_loss_i, wd_i, ot_obj_value_i = self.compute_hot_loss_single_image(outputs_i, outputs_normed_i, points_i)
                
                # Accumulate OT losses directly as float values
                total_ot_loss += ot_loss_i * self.args.wot
                total_count_loss += self.mae(outputs_i.sum().reshape(1), torch.tensor([len(points_i)], dtype=torch.float32).to(self.device))

                # Normalize gt_discrete and compute TV loss
                gd_count_tensor_i = torch.tensor([len(points_i)], dtype=torch.float32).to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                gt_discrete_normed_i = gt_discrete[i].unsqueeze(0) / (gd_count_tensor_i + 1e-6)
                tv_loss_i = (self.tv_loss(outputs_normed_i, gt_discrete_normed_i).sum() * gd_count_tensor_i.item()).mean(0) * self.args.wtv
                total_tv_loss += tv_loss_i

                # Update metrics for each image
                epoch_ot_loss.update(float(ot_loss_i))
                epoch_wd.update(float(wd_i))
                epoch_ot_obj_value.update(float(ot_obj_value_i))
                epoch_count_loss.update(float(total_count_loss))
                epoch_tv_loss.update(float(tv_loss_i))

                # Store predicted and ground truth counts for error metrics
                pred_counts.append(outputs_i.sum().item())
                gd_counts.append(len(points_i))

            # Calculate total loss and backpropagate
            loss = total_ot_loss + total_count_loss + total_tv_loss
            loss_tensor = torch.tensor(loss, requires_grad=True, device=self.device)
            self.optimizer.zero_grad()
            loss_tensor.backward()
            self.optimizer.step()

            # Compute epoch-level error metrics
            pred_counts, gd_counts = np.array(pred_counts), np.array(gd_counts)
            pred_err = pred_counts - gd_counts
            epoch_mse.update(np.mean(pred_err ** 2), len(pred_counts))
            epoch_mae.update(np.mean(np.abs(pred_err)), len(pred_counts))
            epoch_loss.update(loss, len(pred_counts))

        # Logging epoch results
        self.logger.info(
            f"Epoch {self.epoch} Train, Loss: {epoch_loss.get_avg():.2f}, OT Loss: {epoch_ot_loss.get_avg():.2e}, "
            f"Wass Distance: {epoch_wd.get_avg():.2f}, OT obj value: {epoch_ot_obj_value.get_avg():.2f}, "
            f"Count Loss: {epoch_count_loss.get_avg():.2f}, TV Loss: {epoch_tv_loss.get_avg():.2f}, "
            f"MSE: {np.sqrt(epoch_mse.get_avg()):.2f} MAE: {epoch_mae.get_avg():.2f}, Cost {time.time() - epoch_start:.1f} sec"
        )

        # Save model checkpoint
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, f"{self.epoch}_ckpt.tar")
        torch.save({
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_state_dict": model_state_dic
        }, save_path)
        self.save_list.append(save_path)




    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()
        epoch_res = []
        for inputs, count, name in self.dataloaders["val"]:
            inputs = inputs.to(self.device)
            assert inputs.size(0) == 1, "Batch size should be 1 in validation mode"
            with torch.no_grad():
                outputs, _ = self.model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.logger.info(f"Epoch {self.epoch} Val, MSE: {mse:.2f} MAE: {mae:.2f}, Cost {time.time() - epoch_start:.1f} sec")

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse, self.best_mae = mse, mae
            self.logger.info(f"Save best mse {self.best_mse:.2f} mae {self.best_mae:.2f} model epoch {self.epoch}")
            torch.save(model_state_dic, os.path.join(self.save_dir, f"best_model_{self.best_count}.pth"))
            self.best_count += 1

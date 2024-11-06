import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime

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

class HOT_Loss(nn.Module):
    def __init__(self, crop_size, downsample_ratio, norm_cood, device, num_of_iter_in_ot, reg, hierarchy_levels=3):
        super(HOT_Loss, self).__init__()
        self.crop_size = crop_size
        self.downsample_ratio = downsample_ratio
        self.norm_cood = norm_cood
        self.device = device
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        self.hierarchy_levels = hierarchy_levels
        self.alpha_weights = self.calculate_alpha_weights(hierarchy_levels)
        
        # Initialize OT_Loss with expected output size based on crop size and downsample ratio
        self.ot_loss_instance = OT_Loss(crop_size, downsample_ratio, norm_cood, device, num_of_iter_in_ot, reg)
        self.expected_output_size = self.ot_loss_instance.output_size

    def calculate_alpha_weights(self, levels):
        return [2 ** i / sum(2 ** j for j in range(levels)) for i in range(levels)]

    def partition_image(self, density_map, level):
        grid_size = 2 ** level
        h, w = density_map.size(-2), density_map.size(-1)
        cell_h, cell_w = h // grid_size, w // grid_size
        cells = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell = density_map[..., i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                
                # Ensure cell has four dimensions (N, C, H, W) before interpolation
                if cell.dim() == 2:  # Only spatial dimensions
                    cell = cell.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                elif cell.dim() == 3:  # No batch dimension
                    cell = cell.unsqueeze(0)  # Add batch dimension
                
                # Resize or pad each cell to match the expected output size for OT_Loss
                cell = nn.functional.interpolate(cell, size=(self.expected_output_size, self.expected_output_size), mode='bilinear', align_corners=False)
                cell = cell.squeeze(0).squeeze(0)  # Remove batch and channel dimensions for further processing
                
                cells.append(cell)
        return cells

    def forward(self, pred, gt, points):
        total_ot_loss, total_wd, total_ot_obj_value = 0, 0, 0
        for level in range(self.hierarchy_levels):
            alpha = self.alpha_weights[level]
            cells_pred = self.partition_image(pred, level)
            cells_gt = self.partition_image(gt, level)
            
            # Accumulate OT loss, wd, and ot_obj_value for each cell
            level_ot_loss, level_wd, level_ot_obj_value = 0, 0, 0
            for pred_cell, gt_cell in zip(cells_pred, cells_gt):
                ot_loss, wd, ot_obj_value = self.ot_loss_instance(pred_cell, gt_cell, points)
                level_ot_loss += ot_loss
                level_wd += wd
                level_ot_obj_value += ot_obj_value
            
            # Apply the weight for the current level
            total_ot_loss += alpha * level_ot_loss
            total_wd += alpha * level_wd
            total_ot_obj_value += alpha * level_ot_obj_value

        return total_ot_loss, total_wd, total_ot_obj_value


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        sub_dir = 'input-{}_hot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}'.format(
            args.crop_size, args.wot, args.wtv, args.reg, args.num_of_iter_in_ot, args.norm_cood)

        self.save_dir = os.path.join('ckpts', sub_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        self.logger = log_utils.get_logger(os.path.join(self.save_dir, 'train-{:s}.log'.format(time_str)))
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            self.logger.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("GPU is not available")

        downsample_ratio = 8
        if args.dataset.lower() == 'qnrf':
            self.datasets = {x: Crowd_qnrf(os.path.join(args.data_dir, x),
                                           args.crop_size, downsample_ratio, x) for x in ['train', 'val']}
        elif args.dataset.lower() == 'nwpu':
            self.datasets = {x: Crowd_nwpu(os.path.join(args.data_dir, x),
                                           args.crop_size, downsample_ratio, x) for x in ['train', 'val']}
        elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
            self.datasets = {'train': Crowd_sh(os.path.join(args.data_dir, 'train_data'),
                                               args.crop_size, downsample_ratio, 'train'),
                             'val': Crowd_sh(os.path.join(args.data_dir, 'test_data'),
                                             args.crop_size, downsample_ratio, 'val')}
        else:
            raise NotImplementedError

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.model = vgg19()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            self.logger.info('loading pretrained model from ' + args.resume)
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))
        else:
            self.logger.info('random initialization')

        self.ot_loss = HOT_Loss(args.crop_size, downsample_ratio, args.norm_cood, self.device, args.num_of_iter_in_ot,
                                args.reg, hierarchy_levels=3)
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    # The rest of the training and validation logic remains the same
    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        for step, (inputs, points, gt_discrete) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)

            with torch.set_grad_enabled(True):
                outputs, outputs_normed = self.model(inputs)
                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points)
                ot_loss = ot_loss * self.args.wot
                ot_obj_value = ot_obj_value * self.args.wot
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)

                # Compute counting loss.
                count_loss = self.mae(outputs.sum(1).sum(1).sum(1),
                                      torch.from_numpy(gd_count).float().to(self.device))
                epoch_count_loss.update(count_loss.item(), N)

                # Compute TV loss.
                gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(
                    2).unsqueeze(3)
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(
                    1) * torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.wtv
                epoch_tv_loss.update(tv_loss.item(), N)

                loss = ot_loss + count_loss + tv_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)

        self.logger.info(
            'Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, '
            'Count Loss: {:.2f}, TV Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                .format(self.epoch, epoch_loss.get_avg(), epoch_ot_loss.get_avg(), epoch_wd.get_avg(),
                        epoch_ot_obj_value.get_avg(), epoch_count_loss.get_avg(), epoch_tv_loss.get_avg(),
                        np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                        time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)

    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()
        epoch_res = []
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs, _ = self.model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.logger.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            self.best_count += 1

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
from models import DMCount
from losses.ot_loss import OT_Loss
from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    gt_discretes = torch.stack(transposed_batch[2], 0)
    return images, points, gt_discretes

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.scale_weights = {
            '4x4':1,  # or some other default weight
            '8x8':1,
            '16x16':1
        }

    def setup(self):
        args = self.args
        sub_dir = 'input-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}'.format(
            args.crop_size, args.wot, args.wtv, args.reg, args.num_of_iter_in_ot, args.norm_cood)

        self.save_dir = os.path.join('ckpts', sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        self.logger = log_utils.get_logger(os.path.join(self.save_dir, 'train-{:s}.log'.format(time_str)))
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            self.logger.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

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
                                             args.crop_size, downsample_ratio, 'val'),
                             }
        else:
            raise NotImplementedError

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate if x == 'train' else default_collate),
                                          batch_size=(args.batch_size if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}

        self.model = DMCount()
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

        # Initialize three OT losses for different scales
        self.ot_loss_4x4 = OT_Loss(args.crop_size, self.device, args.norm_cood, args.num_of_iter_in_ot, args.reg)
        self.ot_loss_8x8 = OT_Loss(args.crop_size, self.device, args.norm_cood, args.num_of_iter_in_ot, args.reg)
        self.ot_loss_16x16 = OT_Loss(args.crop_size, self.device, args.norm_cood, args.num_of_iter_in_ot, args.reg)
        
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0


    def compute_scale_loss(self, pred_density, pred_normed, points, gt_discrete, gd_count, scale_size):
        """Compute losses for a specific scale"""
        N = pred_density.size(0)
        
        # Compute OT loss
        if scale_size == 4:
            ot_loss, wd, ot_obj_value = self.ot_loss_4x4(pred_normed, pred_density, points, 4)
        elif scale_size == 8:
            ot_loss, wd, ot_obj_value = self.ot_loss_8x8(pred_normed, pred_density, points, 8)
        else:  # 16x16
            ot_loss, wd, ot_obj_value = self.ot_loss_16x16(pred_normed, pred_density, points, 16)
            
        ot_loss = ot_loss * self.args.wot
        ot_obj_value = ot_obj_value * self.args.wot

        # Compute counting loss
        count_loss = self.mae(pred_density.sum(1).sum(1).sum(1),
                            torch.from_numpy(gd_count).float().to(self.device))

        # Compute TV loss with shape alignment
        gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)

        # Resize gt_discrete_normed to match pred_normed's spatial dimensions
        gt_discrete_normed_resized = F.interpolate(gt_discrete_normed, size=pred_normed.shape[2:], mode='bilinear', align_corners=False)

        # Now `gt_discrete_normed_resized` has the same shape as `pred_normed`
        tv_loss = (self.tv_loss(pred_normed, gt_discrete_normed_resized).sum(1).sum(1).sum(1) * 
                torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.wtv

        return ot_loss, ot_obj_value, wd, count_loss, tv_loss
    
    def compute_weighted_count(self, mu_4x4, mu_8x8, mu_16x16):
        """Compute weighted count from all scales"""
        # Interpolate all scales to the largest scale
        B = mu_4x4.size(0)
        mu_4x4_up = F.interpolate(mu_4x4, size=mu_16x16.shape[2:], mode='bilinear', align_corners=False)
        mu_8x8_up = F.interpolate(mu_8x8, size=mu_16x16.shape[2:], mode='bilinear', align_corners=False)
        
        # Weighted combination
        weighted_density = (
            self.scale_weights['4x4'] * mu_4x4_up + 
            self.scale_weights['8x8'] * mu_8x8_up + 
            self.scale_weights['16x16'] * mu_16x16
        )
        
        return weighted_density
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
        self.model.train()

        for step, (inputs, points, gt_discrete) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)

            with torch.set_grad_enabled(True):
                # Get predictions for all scales
                (mu_4x4, mu_4x4_normed), (mu_8x8, mu_8x8_normed), (mu_16x16, mu_16x16_normed) = self.model(inputs)

                # Compute losses for each scale
                loss_4x4 = self.compute_scale_loss(mu_4x4, mu_4x4_normed, points, gt_discrete, gd_count, 4)
                loss_8x8 = self.compute_scale_loss(mu_8x8, mu_8x8_normed, points, gt_discrete, gd_count, 8)
                loss_16x16 = self.compute_scale_loss(mu_16x16, mu_16x16_normed, points, gt_discrete, gd_count, 16)

                # Combine losses with weights
                ot_loss = (
                    self.scale_weights['4x4'] * loss_4x4[0] + 
                    self.scale_weights['8x8'] * loss_8x8[0] + 
                    self.scale_weights['16x16'] * loss_16x16[0]
                )
                
                ot_obj_value = (
                    self.scale_weights['4x4'] * loss_4x4[1] + 
                    self.scale_weights['8x8'] * loss_8x8[1] + 
                    self.scale_weights['16x16'] * loss_16x16[1]
                )
                
                wd = (
                    self.scale_weights['4x4'] * loss_4x4[2] + 
                    self.scale_weights['8x8'] * loss_8x8[2] + 
                    self.scale_weights['16x16'] * loss_16x16[2]
                )
                
                count_loss = (
                    self.scale_weights['4x4'] * loss_4x4[3] + 
                    self.scale_weights['8x8'] * loss_8x8[3] + 
                    self.scale_weights['16x16'] * loss_16x16[3]
                )
                
                tv_loss = (
                    self.scale_weights['4x4'] * loss_4x4[4] + 
                    self.scale_weights['8x8'] * loss_8x8[4] + 
                    self.scale_weights['16x16'] * loss_16x16[4]
                )

                loss = ot_loss + count_loss + tv_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Use weighted combination of all scales for count prediction
                weighted_density = self.compute_weighted_count(mu_4x4, mu_8x8, mu_16x16)
                pred_count = torch.sum(weighted_density.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gd_count

                # Update metrics
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)
                epoch_count_loss.update(count_loss.item(), N)
                epoch_tv_loss.update(tv_loss.item(), N)
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

        # Save checkpoint
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)
    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()
        epoch_res = []

        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                # Get predictions for all scales
                (mu_4x4, _), (mu_8x8, _), (mu_16x16, _) = self.model(inputs)
                
                # Compute weighted density map
                weighted_density = self.compute_weighted_count(mu_4x4, mu_8x8, mu_16x16)
                pred_count = torch.sum(weighted_density).item()
                res = count[0].item() - pred_count
                epoch_res.append(res)


        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        
        self.logger.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            self.best_count += 1

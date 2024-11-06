import torch
from torch.nn import Module
from .bregman_pytorch import sinkhorn

class OT_Loss(Module):
    def __init__(self, c_size, device, norm_cood=True, num_of_iter_in_ot=100, reg=10.0):
        super(OT_Loss, self).__init__()
        
        # Initialize parameters
        self.c_size = c_size
        self.device = device
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        
        # Create coordinate grids for different scales
        self.scale_configs = {
            4: {'stride': c_size // 4},
            8: {'stride': c_size // 8},
            16: {'stride': c_size // 16}
        }
        
        # Initialize coordinate grids for each scale
        self.scale_coords = {}
        for scale, config in self.scale_configs.items():
            cood = torch.arange(0, c_size, step=config['stride'],
                              dtype=torch.float32, device=device) + config['stride'] / 2
            cood.unsqueeze_(0)  # [1, #cood]
            
            if self.norm_cood:
                cood = cood / c_size * 2 - 1  # map to [-1, 1]
                
            self.scale_coords[scale] = cood
            config['output_size'] = cood.size(1)

    def compute_distance_matrix(self, points, scale):
        """Compute L2 square distance matrix for a specific scale"""
        cood = self.scale_coords[scale]
        output_size = self.scale_configs[scale]['output_size']
        
        if self.norm_cood:
            points = points / self.c_size * 2 - 1  # map to [-1, 1]
            
        x = points[:, 0].unsqueeze_(1)  # [#gt, 1]
        y = points[:, 1].unsqueeze_(1)
        
        x_dis = -2 * torch.matmul(x, cood) + x * x + cood * cood  # [#gt, #cood]
        y_dis = -2 * torch.matmul(y, cood) + y * y + cood * cood
        
        y_dis.unsqueeze_(2)
        x_dis.unsqueeze_(1)
        dis = y_dis + x_dis
        
        return dis.view((dis.size(0), -1)), output_size  # [#gt, #cood * #cood]

    def compute_ot_loss(self, normed_density, unnormed_density, points, scale):
        """Compute OT loss for a specific scale"""
        batch_size = normed_density.size(0)
        output_size = self.scale_configs[scale]['output_size']
        assert len(points) == batch_size
        assert output_size == normed_density.size(2)
        
        loss = torch.zeros([1]).to(self.device)
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0  # wasserstein distance
        
        for idx, im_points in enumerate(points):
            if len(im_points) > 0:
                # Compute distance matrix for current scale
                dis, output_size = self.compute_distance_matrix(im_points, scale)
                
                # Prepare source and target probabilities
                source_prob = normed_density[idx][0].view([-1]).detach()
                target_prob = (torch.ones([len(im_points)]) / len(im_points)).to(self.device)
                
                # Compute optimal transport
                P, log = sinkhorn(target_prob, source_prob, dis, self.reg, 
                                maxIter=self.num_of_iter_in_ot, log=True)
                beta = log['beta']
                
                # Compute OT objective values
                ot_obj_values += torch.sum(normed_density[idx] * 
                                         beta.view([1, output_size, output_size]))
                
                # Compute gradients
                source_density = unnormed_density[idx][0].view([-1]).detach()
                source_count = source_density.sum()
                im_grad_1 = source_count / (source_count * source_count + 1e-8) * beta
                im_grad_2 = (source_density * beta).sum() / (source_count * source_count + 1e-8)
                im_grad = im_grad_1 - im_grad_2
                im_grad = im_grad.detach().view([1, output_size, output_size])
                
                # Compute final loss
                loss += torch.sum(unnormed_density[idx] * im_grad)
                wd += torch.sum(dis * P).item()
        
        return loss, wd, ot_obj_values

    def forward(self, normed_density, unnormed_density, points, scale):
        """Forward pass with scale selection"""
        assert scale in self.scale_configs, f"Invalid scale {scale}. Must be one of {list(self.scale_configs.keys())}"
        return self.compute_ot_loss(normed_density, unnormed_density, points, scale)

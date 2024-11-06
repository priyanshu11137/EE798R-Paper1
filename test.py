import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
from models import DMCount
import multiprocessing

def main():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the test image')
    parser.add_argument('--model-path', type=str, default='pretrained_models/model_qnrf.pth',
                        help='saved model path')
    parser.add_argument('--data-path', type=str,
                        default='data/QNRF-Train-Val-Test',
                        help='path to dataset')
    parser.add_argument('--dataset', type=str, default='qnrf',
                        help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--pred-density-map-path', type=str, default='',
                        help='save predicted density maps when pred-density-map-path is not empty.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = args.model_path
    crop_size = args.crop_size
    data_path = args.data_path

    # Dataset loading
    if args.dataset.lower() == 'qnrf':
        dataset = crowd.Crowd_qnrf(os.path.join(data_path, 'test'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'nwpu':
        dataset = crowd.Crowd_nwpu(os.path.join(data_path, 'val'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
        dataset = crowd.Crowd_sh(os.path.join(data_path, 'test_data'), crop_size, 8, method='val')
    else:
        raise NotImplementedError

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    if args.pred_density_map_path:
        import cv2
        if not os.path.exists(args.pred_density_map_path):
            os.makedirs(args.pred_density_map_path)

    def load_model(model, model_path, device):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model

    model = DMCount()
    model = load_model(model, model_path, device)
    model.to(device)
    model.eval()

    image_errs = []
    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            # Get multi-scale density maps from the model
            (mu_4x4, _), (mu_8x8, _), (mu_16x16, _) = model(inputs)
            
            # Compute average predicted count across scales
            a=torch.sum(mu_4x4).item()
            b=torch.sum(mu_8x8).item()
            c=torch.sum(mu_16x16).item()
            pred_count = (a+b+c)/ 3
            img_err = count[0].item() - pred_count

        print(name, img_err, count[0].item(), pred_count, a, b, c)
        image_errs.append(img_err)

        if args.pred_density_map_path:
            # Save the visualized density map for the largest scale (16x16)
            vis_img = mu_16x16[0, 0].cpu().numpy()
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + '.png'), vis_img)

    image_errs = np.array(image_errs)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print('{}: MAE: {:.2f}, MSE: {:.2f}'.format(model_path, mae, mse))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

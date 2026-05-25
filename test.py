import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from test_utils.metric_new import SigmoidMetric, SamplewiseSigmoidMetric, PD0_FA0
import datasets
import models

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sam-vit-b-token-TSIRMT.yaml')
    parser.add_argument('--model', default="weights/model_TSIRMT_SAM1-B.pth")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--bin-thresh', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=8)
    return parser.parse_args()

def load_state_dict(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt, strict=True)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    model = models.make(config['model']).to(device)
    load_state_dict(model, args.model, device)
    model.eval()

    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=args.bin_thresh)
    pd_fa_metric = PD0_FA0(nclass=1, thre=args.bin_thresh)

    frame_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', ncols=120):
            inp = batch['inp'].to(device)
            gt = batch['gt'].to(device)
            pred_prob = torch.sigmoid(model.infer(inp, boxes=None))

            for b in range(pred_prob.shape[0]):
                prob_tensor = pred_prob[b:b + 1]
                gt_tensor = gt[b:b + 1]

                if 'gt_orig' in batch:
                    gt_orig = batch['gt_orig'].to(device)[b:b + 1]
                else:
                    gt_orig = gt_tensor

                orig_h, orig_w = gt_orig.shape[-2:]
                prob_resized = F.interpolate(prob_tensor, size=(orig_h, orig_w), mode='area')

                pred_np = (prob_resized[0, 0].cpu().numpy() > args.bin_thresh).astype(np.uint8)
                gt_np = (gt_orig[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

                iou_metric.update(pred_np, gt_np)
                niou_metric.update(pred_np, gt_np)
                pd_fa_metric.update(pred_np, gt_np)
                frame_idx += 1

    _, iou, _ = iou_metric.get()
    niou = niou_metric.get()
    fa, pd = pd_fa_metric.get()

    print(f'\n===== Evaluation frames: {frame_idx} =====')
    print(f'IoU:  {iou:.4f}')
    print(f'nIoU: {niou:.4f}')
    print(f'PD:   {pd:.4f}')
    print(f'FA:   {fa:.8f}')
    print('======================================')
    print(args.model)

if __name__ == '__main__':
    main()

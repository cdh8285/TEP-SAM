import bisect
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from datasets import register

def _locate_seq(global_idx, seq_offsets, seq_lens):
    seq_idx = bisect.bisect_right(seq_offsets, global_idx) - 1
    base = seq_offsets[seq_idx]
    last = base + seq_lens[seq_idx] - 1
    return base, last

def _get_meta(dataset, idx):
    metas = getattr(dataset, 'metas', None)
    if metas is None:
        return {}
    return metas[idx]

@register('seqwin')
class SeqWinDataset(Dataset):
    def __init__(self, dataset, inp_size, window_size=7, augment=False, **kwargs):
        assert window_size % 2 == 1, 'window_size must be odd.'
        self.dataset = dataset
        self.inp_size = inp_size
        self.window = window_size
        self.half = window_size // 2
        self.augment = augment

        self.seq_offsets = getattr(dataset, 'seq_offsets', None)
        self.seq_lens = getattr(dataset, 'seq_lens', None)
        assert self.seq_offsets is not None and self.seq_lens is not None, \
            'The base dataset must provide seq_offsets and seq_lens.'

        self.total = len(dataset)
        self.center_indices = list(range(self.total))

        self.img_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        self.mask_transform_middle = transforms.Compose([
            transforms.Resize((inp_size // 4, inp_size // 4), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        center = self.center_indices[idx]
        base, last = _locate_seq(center, self.seq_offsets, self.seq_lens)

        do_hflip = self.augment and random.random() < 0.5
        do_vflip = self.augment and random.random() < 0.5
        rot_angle = random.choice([0, 90, 180, 270]) if self.augment else 0
        do_t_reverse = self.augment and random.random() < 0.5

        def spatial_aug(x):
            if do_hflip:
                x = x.transpose(Image.FLIP_LEFT_RIGHT)
            if do_vflip:
                x = x.transpose(Image.FLIP_TOP_BOTTOM)
            if rot_angle:
                x = x.rotate(rot_angle, expand=False)
            return x

        imgs_pil = []
        for offset in range(-self.half, self.half + 1):
            frame_idx = min(max(center + offset, base), last)
            img, _ = self.dataset[frame_idx]
            imgs_pil.append(spatial_aug(img))

        _, mask_center = self.dataset[center]
        mask_center = spatial_aug(mask_center)

        if do_t_reverse:
            imgs_pil = list(reversed(imgs_pil))

        inp = torch.stack([self.img_transform(img) for img in imgs_pil], dim=0)
        gt = self.mask_transform(mask_center)
        gt_middle = self.mask_transform_middle(mask_center)
        single_inp = self.img_transform(imgs_pil[self.half])

        return {
            'inp': inp,
            'gt': gt,
            'single_inp': single_inp,
            'gt_middle': gt_middle,
        }

@register('seqwin-test')
class SeqWinTestDataset(Dataset):
    def __init__(self, dataset, inp_size, window_size=7, augment=False, **kwargs):
        assert window_size % 2 == 1, 'window_size must be odd.'
        self.dataset = dataset
        self.inp_size = inp_size
        self.window = window_size
        self.half = window_size // 2

        self.seq_offsets = getattr(dataset, 'seq_offsets', None)
        self.seq_lens = getattr(dataset, 'seq_lens', None)
        assert self.seq_offsets is not None and self.seq_lens is not None, \
            'The base dataset must provide seq_offsets and seq_lens.'

        self.total = len(dataset)
        self.center_indices = list(range(self.total))

        self.img_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        self.mask_transform_middle = transforms.Compose([
            transforms.Resize((inp_size // 4, inp_size // 4), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        center = self.center_indices[idx]
        base, last = _locate_seq(center, self.seq_offsets, self.seq_lens)

        imgs_pil = []
        for offset in range(-self.half, self.half + 1):
            frame_idx = min(max(center + offset, base), last)
            img, _ = self.dataset[frame_idx]
            imgs_pil.append(img)

        _, mask_center = self.dataset[center]
        inp = torch.stack([self.img_transform(img) for img in imgs_pil], dim=0)
        gt = self.mask_transform(mask_center)
        gt_middle = self.mask_transform_middle(mask_center)
        gt_orig = self.to_tensor(mask_center)
        single_inp = self.img_transform(imgs_pil[self.half])

        meta = _get_meta(self.dataset, center)
        orig_h, orig_w = gt_orig.shape[-2:]

        return {
            'inp': inp,
            'gt': gt,
            'single_inp': single_inp,
            'gt_middle': gt_middle,
            'gt_orig': gt_orig,
            'seq_name': meta.get('seq_name', ''),
            'frame_name': meta.get('frame_name', ''),
            'img_path': meta.get('img_path', ''),
            'gt_path': meta.get('gt_path', ''),
            'orig_h': orig_h,
            'orig_w': orig_w,
        }

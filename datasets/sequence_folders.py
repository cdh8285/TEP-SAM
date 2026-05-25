import os
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from datasets import register

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

def _list_images(path):
    if not os.path.isdir(path):
        return []
    files = []
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMG_EXTS:
            files.append(p)
    return sorted(files, key=lambda x: _natural_key(os.path.basename(x)))

def _parse_seq_name(line):
    token = line.strip().split()[0].strip('"\'')
    token = token.replace('\\', '/').rstrip('/')
    ext = os.path.splitext(token)[1].lower()
    if ext in IMG_EXTS:
        return os.path.basename(os.path.dirname(token))
    return os.path.basename(token)

def _read_seq_list(seq_list):
    if seq_list is None:
        return None
    names = []
    seen = set()
    with open(seq_list, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            name = _parse_seq_name(line)
            if name and name not in seen:
                names.append(name)
                seen.add(name)
    return names

@register('paired-sequence-folders')
class PairedSequenceFolders(Dataset):
    def __init__(self,
                 root_path_1,
                 root_path_2=None,
                 layout='auto',
                 image_dir_name='images',
                 mask_dir_name='masks',
                 seq_list=None,
                 cache='none',
                 split_key=None,
                 allow_missing_mask=False,
                 binarize_masks=True,
                 **kwargs):
        self.root_path_1 = root_path_1
        self.root_path_2 = root_path_2 if root_path_2 is not None else root_path_1
        self.layout = self._detect_layout(layout, image_dir_name, mask_dir_name)
        self.image_dir_name = image_dir_name
        self.mask_dir_name = mask_dir_name
        self.allow_missing_mask = allow_missing_mask
        self.binarize_masks = binarize_masks
        seq_names = _read_seq_list(seq_list)
        self.files_img = []
        self.files_mask = []
        self.metas = []
        self.seq_offsets = []
        self.seq_lens = []
        if self.layout == 'seq_subdirs':
            seq_dirs = self._collect_seq_dirs(self.root_path_1, seq_names)
            for seq_name, seq_dir in seq_dirs:
                img_dir = os.path.join(seq_dir, image_dir_name)
                mask_dir = os.path.join(seq_dir, mask_dir_name)
                self._append_sequence(seq_name, img_dir, mask_dir)
        elif self.layout == 'split_images_masks':
            seq_dirs = self._collect_seq_dirs(self.root_path_1, seq_names)
            for seq_name, img_dir in seq_dirs:
                mask_dir = os.path.join(self.root_path_2, seq_name)
                self._append_sequence(seq_name, img_dir, mask_dir)
        else:
            raise ValueError('Unknown layout: {}'.format(self.layout))
        if len(self.files_img) == 0:
            raise RuntimeError('No image-mask pairs found. Please check dataset paths, layout and seq_list.')
        self.dataset_1 = SimpleNamespace(files=self.files_img)
        self.dataset_2 = SimpleNamespace(files=self.files_mask)

    def _detect_layout(self, layout, image_dir_name, mask_dir_name):
        if layout != 'auto':
            return layout
        if not os.path.isdir(self.root_path_1):
            raise FileNotFoundError('root_path_1 does not exist: {}'.format(self.root_path_1))
        for name in sorted(os.listdir(self.root_path_1), key=_natural_key):
            seq_dir = os.path.join(self.root_path_1, name)
            if not os.path.isdir(seq_dir):
                continue
            if os.path.isdir(os.path.join(seq_dir, image_dir_name)) and os.path.isdir(os.path.join(seq_dir, mask_dir_name)):
                return 'seq_subdirs'
        return 'split_images_masks'

    def _collect_seq_dirs(self, root, seq_names=None):
        if not os.path.isdir(root):
            raise FileNotFoundError('Directory does not exist: {}'.format(root))
        if seq_names is None:
            names = [n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n))]
            names = sorted(names, key=_natural_key)
        else:
            names = seq_names
        seq_dirs = []
        missing = []
        for name in names:
            p = os.path.join(root, name)
            if os.path.isdir(p):
                seq_dirs.append((name, p))
            else:
                missing.append(name)
        if missing:
            raise FileNotFoundError('Sequences listed in seq_list are missing under {}: {}'.format(root, ', '.join(missing[:10])))
        return seq_dirs

    def _append_sequence(self, seq_name, img_dir, mask_dir):
        img_paths = _list_images(img_dir)
        mask_paths = _list_images(mask_dir)
        mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}
        start = len(self.files_img)
        added = 0
        for img_path in img_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = mask_map.get(stem, None)
            if mask_path is None and not self.allow_missing_mask:
                raise FileNotFoundError('Missing mask for image: {}'.format(img_path))
            self.files_img.append(img_path)
            self.files_mask.append(mask_path)
            self.metas.append({
                'seq_name': seq_name,
                'frame_name': stem,
                'img_path': img_path,
                'gt_path': mask_path if mask_path is not None else '',
            })
            added += 1
        if added > 0:
            self.seq_offsets.append(start)
            self.seq_lens.append(added)

    def __len__(self):
        return len(self.files_img)

    def __getitem__(self, idx):
        img = Image.open(self.files_img[idx]).convert('RGB')
        mask_path = self.files_mask[idx]
        if mask_path is None:
            mask = Image.new('L', img.size, 0)
        else:
            mask = Image.open(mask_path).convert('L')
            if self.binarize_masks:
                arr = np.array(mask)
                arr = (arr > 0).astype(np.uint8) * 255
                mask = Image.fromarray(arr, mode='L')
        return img, mask

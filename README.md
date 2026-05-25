# TEP-SAM1

This repository contains a implementation of TEP-SAM for multi-frame infrared small target segmentation.

## Dataset Layout

NUDT-MIRSDT:

```text
NUDT-MIRSDT/
  sequence_train/
    Sequence001/
      images/
      masks/
  sequence_test/
    Sequence081/
      images/
      masks/
```

TSIRMT:

```text
TSIRMT/
  images/
    Sequence001/
    Sequence002/
  masks/
    Sequence001/
    Sequence002/
  ImageSets/
    train_new.txt
    val_new.txt
    val_snr_smaller_than_10.txt
```

For TSIRMT, the loader reads sequence names from the `ImageSets/*.txt` files and then loads frames from `images/<seq_name>` and `masks/<seq_name>`. To use another split, modify the `seq_list` field in `configs/sam-vit-b-token-TSIRMT.yaml`.

## Environment

```bash
pip install -r requirements.txt
```

Put the SAM ViT-B checkpoint here:

```text
pretrained/sam_vit_b_01ec64.pth
```

## Training

Edit dataset paths in the config file first.

```bash
torchrun --nproc_per_node=1 train.py \
  --config configs/sam-vit-b-token.yaml \
  --name tep_sam1_nudt \
  --tag run1
```

For TSIRMT:

```bash
torchrun --nproc_per_node=1 train.py \
  --config configs/sam-vit-b-token-TSIRMT.yaml \
  --name tep_sam1_tsirmt \
  --tag run1
```

## Testing

```bash
python test.py \
  --config configs/sam-vit-b-token.yaml \
  --model path/to/model_epoch_epoch_40.pth
```

The test script reports IoU, nIoU, PD, and FA.

## Trained Weights

| Dataset | Google Drive | IoU | nIoU | PD | FA |
|:--|:--|--:|--:|--:|--:|
| NUDT-MIRSDT | [Download](https://drive.google.com/file/d/1_hbtkHZiVZeT-h9chdUagmAPfsTBlqoa/view?usp=sharing) | 0.8615 | 0.8628 | 0.9971 | 0.00000051 |
| TSIRMT | [Download](https://drive.google.com/file/d/1SwHd19ph0nVjw_dGPhtl0HliyQBNlzSN/view?usp=drive_link) | 0.7434 | 0.7321 | 0.9204 | 0.00010897 |

import argparse
import os
import os.path as osp

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert IDRiD 'A. Segmentation' data to mmsegmentation format "
            "(compatible with FGADRDataset and configs using train_idrid/test_idrid splits)."
        ))
    parser.add_argument(
        'seg_root',
        help="Path to the 'A. Segmentation' folder from the IDRiD dataset.")
    parser.add_argument(
        '-o',
        '--out_dir',
        help="Output root directory. "
        "Defaults to 'data/My_dataset' to match existing configs.",
    )
    args = parser.parse_args()
    return args


def _find_class_dirs(gt_set_root):
    """Infer lesion subdirectories for each class under a train/test GT root.

    Expected structure (names may vary slightly, we match by substrings):

        gt_set_root/
            ...Microaneurysm...
            ...Hard Exudate...
            ...Soft Exudate...
            ...Haemorrhage/Hemorrhage...
    """
    subdirs = [
        d for d in os.listdir(gt_set_root)
        if osp.isdir(osp.join(gt_set_root, d))
    ]
    mapping = {}
    for d in subdirs:
        name = d.lower()
        if 'microaneurysm' in name or 'micro aneurysm' in name:
            mapping['MA'] = d
        elif 'hard' in name and 'exud' in name:
            mapping['EX'] = d
        elif 'soft' in name and 'exud' in name:
            mapping['SE'] = d
        elif 'haemorrhage' in name or 'hemorrhage' in name:
            mapping['HE'] = d

    required = {'EX', 'MA', 'SE', 'HE'}
    missing = required - set(mapping.keys())
    if missing:
        raise RuntimeError(
            f'Could not find lesion GT folders {missing} under {gt_set_root}. '
            'Please check that you pointed seg_root to the "A. Segmentation" '
            'folder and that its subdirectory names are unmodified.')
    return mapping


def _find_mask_path(gt_dir, stem):
    """Return path to a mask file if it exists, otherwise None."""
    for ext in ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'):
        p = osp.join(gt_dir, stem + ext)
        if osp.exists(p):
            return p
    return None


def _process_split(img_src_root,
                   gt_set_root,
                   dst_img_dir,
                   dst_ann_dir,
                   split_list):
    """Process one of {train, test} sets.

    - Copies/converts images into dst_img_dir as JPG.
    - Builds a 5-class segmentation mask (0: background,
      1: EX, 2: MA, 3: SE, 4: HE) into dst_ann_dir as PNG.
    - Appends image stems to split_list.
    """
    class_dirs = _find_class_dirs(gt_set_root)
    class_labels = {'EX': 1, 'MA': 2, 'SE': 3, 'HE': 4}

    img_files = sorted(os.listdir(img_src_root))
    prog_bar = mmcv.ProgressBar(len(img_files))

    for filename in img_files:
        lower = filename.lower()
        if not lower.endswith(
                ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
            prog_bar.update()
            continue

        stem = osp.splitext(filename)[0]
        src_img_path = osp.join(img_src_root, filename)

        # Load and save image as JPG to match FGADRDataset(img_suffix='.jpg')
        img = mmcv.imread(src_img_path)
        dst_img_path = osp.join(dst_img_dir, stem + '.jpg')
        mmcv.imwrite(img, dst_img_path)

        # Build segmentation map from individual lesion masks.
        # Use EX mask (if present) to determine size; otherwise fall back
        # to the first available lesion mask.
        ref_mask_path = None
        if _find_mask_path(
                osp.join(gt_set_root, class_dirs['EX']), stem) is not None:
            ref_mask_path = _find_mask_path(
                osp.join(gt_set_root, class_dirs['EX']), stem)
        else:
            # Fallback: first existing lesion mask
            for cls in ['MA', 'SE', 'HE']:
                candidate = _find_mask_path(
                    osp.join(gt_set_root, class_dirs[cls]), stem)
                if candidate is not None:
                    ref_mask_path = candidate
                    break

        if ref_mask_path is None:
            # No lesion masks at all for this image; create an all-background map
            # with the same height/width as the source image.
            h, w = img.shape[:2]
            seg = np.zeros((h, w), dtype=np.uint8)
        else:
            ref_mask = mmcv.imread(ref_mask_path, flag='grayscale')
            h, w = ref_mask.shape[:2]
            seg = np.zeros((h, w), dtype=np.uint8)

        for cls, label in class_labels.items():
            gt_dir = osp.join(gt_set_root, class_dirs[cls])
            mask_path = _find_mask_path(gt_dir, stem)
            if mask_path is None:
                continue

            mask = mmcv.imread(mask_path, flag='grayscale')
            if mask.shape[0] != seg.shape[0] or mask.shape[1] != seg.shape[1]:
                # Resize mask to reference size if needed
                mask = mmcv.imresize(mask, (seg.shape[1], seg.shape[0]))

            seg[mask > 0] = label

        dst_seg_path = osp.join(dst_ann_dir, stem + '.png')
        mmcv.imwrite(seg, dst_seg_path)

        split_list.append(stem)
        prog_bar.update()


def main():
    args = parse_args()
    seg_root = args.seg_root
    if args.out_dir is None:
        out_dir = osp.join('data', 'My_dataset')
    else:
        out_dir = args.out_dir

    # Expected IDRiD A. Segmentation structure:
    #   seg_root/
    #       1. Original Images/
    #           a. Training Set/
    #           b. Testing Set/
    #       2. All Segmentation Groundtruths/
    #           a. Training Set/
    #           b. Testing Set/
    orig_root = osp.join(seg_root, '1. Original Images')
    gt_root = osp.join(seg_root, '2. All Segmentation Groundtruths')

    train_img_src = osp.join(orig_root, 'a. Training Set')
    test_img_src = osp.join(orig_root, 'b. Testing Set')
    train_gt_root = osp.join(gt_root, 'a. Training Set')
    test_gt_root = osp.join(gt_root, 'b. Testing Set')

    print('Making directories...')
    img_dir = osp.join(out_dir, 'img_dir', 'train_idrid')
    ann_dir = osp.join(out_dir, 'ann_dir', 'train_idrid')
    splits_dir = osp.join(out_dir, 'splits')
    mmcv.mkdir_or_exist(img_dir)
    mmcv.mkdir_or_exist(ann_dir)
    mmcv.mkdir_or_exist(splits_dir)

    train_ids, test_ids = [], []

    print('Processing training set...')
    _process_split(train_img_src, train_gt_root, img_dir, ann_dir, train_ids)

    print('\nProcessing test set...')
    _process_split(test_img_src, test_gt_root, img_dir, ann_dir, test_ids)

    # Write split files used by configs (e.g. HACDR_idrid.py)
    train_split_path = osp.join(splits_dir, 'train_idrid.txt')
    test_split_path = osp.join(splits_dir, 'test_idrid.txt')

    with open(train_split_path, 'w') as f:
        f.writelines(stem + '\n' for stem in train_ids)

    with open(test_split_path, 'w') as f:
        f.writelines(stem + '\n' for stem in test_ids)

    print(f'\nWrote {len(train_ids)} training ids to {train_split_path}')
    print(f'Wrote {len(test_ids)} test ids to {test_split_path}')
    print('Done!')


if __name__ == '__main__':
    main()


import os.path as osp
import os
import numpy as np
from tqdm import tqdm
import cv2


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = '/mnt/sdb/DataSet/objectTracking/Fish_01/images/train'
label_root = 'D:/CompileCode/Dataset/DatasetProecss/2D/new_videoToFrame/Fish/labels_with_ids/train'

seqs = os.listdir(seq_root)
print(seqs)

# seqs = ['fish01-1']

for seq in tqdm(seqs):

    seq_label_root = osp.join(label_root, seq, 'img1')
    all_labels = os.listdir(seq_label_root)
    all_labels = sorted(all_labels)

    if len(all_labels) == 0:
        continue

    fm = cv2.imread(osp.join(seq_root, seq, 'img1', all_labels[0].replace('.txt', '.jpg')))
    seq_height, seq_width, c = fm.shape

    gt_folder = osp.join(seq_root, seq, 'gt')
    mkdirs(gt_folder)

    gt_txt = osp.join(gt_folder, 'gt.txt')

    gt_data = []

    for fid, label_file in enumerate(all_labels):
        # fid = int(label_file.replace('.txt', ''))
        fid = fid + 1
        with open(osp.join(seq_label_root, label_file), 'r') as f:
            labels = f.readlines()
            tid = 0
            for label in labels:
                parts = label.strip().split()
                tid += 1
                x = float(parts[2]) * seq_width
                y = float(parts[3]) * seq_height
                w = float(parts[4]) * seq_width
                h = float(parts[5]) * seq_height
                x -= w / 2
                y -= h / 2
                gt_data.append([fid, tid, x, y, w, h, 1, 1, 1])

    gt_data = np.array(gt_data, dtype=np.float64)
    idx = np.lexsort(gt_data.T[:2, :])
    gt_data = gt_data[idx, :]

    np.savetxt(gt_txt, gt_data, fmt='%.0f', delimiter=',')

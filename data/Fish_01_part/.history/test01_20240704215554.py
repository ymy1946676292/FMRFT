# rename  符合MOT的名字
import os
# labels = os.listdir(
#     '/mnt/sdb/DataSet/objectTracking/Fish_01/labels_with_ids/train')


image_root = '/mnt/sdb/DataSet/objectTracking/Fish_01/images/train'

label_root = '/mnt/sdb/DataSet/objectTracking/Fish_01/labels_with_ids/train'

seqs = os.listdir(image_root)

seqs = sorted(seqs)


for seq in seqs:

    seq_path = os.path.join(image_root, seq,'img1')

    label_path = os.path.join(label_root, seq,'img1')

    images = os.listdir(seq_path)

    if len(images) == 0:
        print(seq)
        continue

    images = sorted(images)

    for i, image in enumerate(images):
        i += 1
        number_name = format(i, '06d')

        new_img_name = os.path.join(seq_path, number_name+'.jpg')

        new_label_name = os.path.join(label_path, number_name+'.txt')

        os.rename(os.path.join(seq_path, image), new_img_name)

        os.rename(os.path.join(
            label_path, image.split('.')[0]+'.txt'), new_label_name)

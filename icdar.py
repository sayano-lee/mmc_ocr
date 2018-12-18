import os
# import argparse
import numpy as np
import csv
from PIL import Image
from os.path import join as opj


# parser = argparse.ArgumentParser()
# parser.add_argument("--train_root", type=str, default="./data/icdar2015/train")

# args = parser.parse_args()

def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.

def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print poly
            print('invalid poly')
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def crop_area(im, polys, tags, crop_background=False, max_tries=50):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h//10
    pad_w = w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx+pad_w:maxx+pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny+pad_h:maxy+pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        if xmax - xmin < args.min_crop_side_ratio*w or ymax - ymin < args.min_crop_side_ratio*h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax+1, xmin:xmax+1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        im = im[ymin:ymax+1, xmin:xmax+1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags

    return im, polys, tags


def convert_array_to_tuple(arr):
    assert type(arr) is np.ndarray
    tmp = tuple(tuple(arr[i].astype("uint64")) for i in range(4))
    return tmp


def generate_crop_boxes(im, bboxes):
    try:
        assert type(bboxes) is np.ndarray
        num_bboxes = bboxes.shape[0]
    except:
        print("Wrong type of bboxes")

    outer_poly = []
    for i in range(num_bboxes):
        crop_bboxes = tuple(convert_array_to_tuple(bboxes[i]) for i in range(num_bboxes))
        left  = min(p[0] for p in crop_bboxes[i])
        upper = max(p[1] for p in crop_bboxes[i])
        right = max(p[0] for p in crop_bboxes[i])
        lower = min(p[1] for p in crop_bboxes[i])
        outer_poly.append((left, lower, right, upper))

    return outer_poly

def convert_bool_to_label(tags):
    label = np.ones(len(tags), dtype="uint8")
    for cnt, i in enumerate(tags):
        if i: label[cnt] = 0
    return label


class icdar(object):

    def __init__(self, root):
        super(icdar, self).__init__()
        self.root = root
        self.fns = [i.split('.')[0] for i in os.listdir(opj(root, 'img'))]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, index):
        # for cnt, fn in enumerate(self.fns):
        fn = self.fns[index]

        ann_fn = opj(self.root, 'gt', fn + '.txt')
        im_fn = opj(self.root, 'img', fn + '.jpg')

        im = Image.open(im_fn)
        h = im.size[1]
        w = im.size[0]
        text_polys, text_tags = load_annoataion(ann_fn)

        text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

        crops = generate_crop_boxes(im, text_polys)
        # labels = convert_bool_to_label(text_tags)

        cropped_imgs = []
        for crop in crops:
            cropped_imgs.append(im.crop(crop))

        return cropped_imgs, text_tags

    def __len__(self):
        return len(self.fns)


if __name__ == '__main__':
    # dataset = icdar(args.train_root)

    # for cnt, data in enumerate(dataset):
    #     import ipdb
    #     ipdb.set_trace()
    pass

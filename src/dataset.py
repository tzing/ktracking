import glob
import os
import cv2
import numpy

class Dataset:
    def __init__(self, folder):
        if not os.path.exists(folder):
            raise ValueError('Not a valid folder')

        # images
        fn_imgs = glob.glob(f'{folder}/*.jpg')
        fn_imgs.sort()
        self.imgs = [cv2.imread(f) for f in fn_imgs]

        # ground truth
        with open(f'{folder}/groundtruth.txt', 'r') as fp:
            groundtruth = fp.read().split('\n')

        def norm_boundary(pts_str):
            pts = numpy.array(pts_str.split(','), dtype=numpy.float64).reshape(4,2)
            x, y = zip(*pts)
            return numpy.array([
                [min(x), min(y)],
                [max(x), max(y)],
            ])

        self.gtruth = [norm_boundary(g) for g in groundtruth if g != '']

        # recheck
        assert len(self.imgs) == len(self.gtruth)

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        return f'<DataSet with {len(self)} imgs>'

    def frame(self, idx):
        return self.imgs[idx], self.gtruth[idx]
    
    def target(self, idx):
        img, gt = self.frame(idx)
        (sx, sy), (ex, ey) = gt.astype(int)
        return img[sy:ey, sx:ex]
    
    def draw_gtruth(self, frame):
        img = self.imgs[frame]
        x0, y0 = self.gtruth[frame][0]
        x1, y1 = self.gtruth[frame][1]

        plt.imshow(img)
        plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'r-')
        plt.show()
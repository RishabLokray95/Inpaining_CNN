import numpy as np
import cv2

from torch.utils.data.dataset import Dataset


class LoadImageFromFolder(Dataset):

    # Generates masked_image, masks, and target images for training
    def __init__(self, images):
        # Initialize the constructor

        self.input_image = images
        self.length_data = len(self.input_image)
        self.indexes = np.arange(len(self.input_image))
        self.X_unmasked = []
        self.mask_input = []
        self.y_output = []
        self.T = []
        for im in self.input_image:
            Y_unmasked = im / 255
            X_masked, Mask = self.__data_generation(im)
            self.T.append((X_masked, Mask, Y_unmasked))

    def __getitem__(self, index):
        return self.T[index]

    def __len__(self):
        return self.length_data

    def __data_generation(self, im):
        image_copy = im
        masked_image, mask = self.__createMask(image_copy)
        masked_image = masked_image / 255
        mask = mask / 255
        return masked_image, mask

    def __createMask(self, im):
        ## White background
        mask = np.full((32, 32, 3), 255, np.uint8)

        for _ in range(np.random.randint(1, 3)):
            # Get random x locations to start line
            x1, x2 = np.random.randint(1, 32), np.random.randint(1, 32)
            # Get random y locations to start line
            y1, y2 = np.random.randint(1, 32), np.random.randint(1, 32)
            # Get random thickness of the line drawn
            thickness = np.random.randint(1, 3)
            # Draw black line on the white mask
            cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 0), thickness)
            ## Mask the image

        masked_image = im
        masked_image[mask == 0] = 255
        return masked_image, mask


class AverageMeter(object):
    """An easy way to compute and store both average and current values"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


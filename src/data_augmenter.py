import numpy as np
import matplotlib.pyplot as plt
import cv2

class DataAugmenter():
    def __init__(self, images, labels, transform_type, special_preprocessing=False):
        self.images = images
        self.labels = labels
        self.transform_type = transform_type
        self.special_preprocessing = special_preprocessing

        self.augmented_images = []
        self.augmented_labels = []

        assert len(labels) == len(images)

        for i in range(len(images)):
            image, label = images[i], labels[i]
            aug_images, aug_labels = self.apply_transform(image, label, transform_type)
            self.augmented_images += aug_images
            self.augmented_labels += aug_labels

        self.augmented_images = np.asarray(self.augmented_images, dtype=np.float32)
        self.augmented_labels = np.asarray(self.augmented_labels)

    def adaptive_threshold(self, color_image=None, gray_image=None):
        assert not (color_image is None and gray_image is None)
        if gray_image is None:
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        res = gray_image
        #res = cv2.GaussianBlur(res, (3, 3), 0)
        res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 3)
        return res

    def erosion_dilation(self, color_image=None, gray_image=None):
        assert not (color_image is None and gray_image is None)
        if gray_image is None:
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        res = gray_image
        res = cv2.erode(res, np.ones((2, 11), np.uint8), iterations=1) # <--- remove non-horizontal lines
        res = cv2.dilate(res, np.ones((2, 51), np.uint8), iterations=1) # <--- recompose horizontal lines
        return res

    def scale(self, image):
        return image / 255.0
        #return image / 127.5 - 1

    def identity(self, image):
        return np.copy(image)

    def flip_horizontal(self, image):
        return np.flip(image, axis=0)

    def flip_vertical(self, image):
        return np.flip(image, axis=1)

    def rotate(self, image):
        return np.rot90(image, k=2)

    def apply_transform(self, image, label, transform_type):
        if self.special_preprocessing:
            image = self.adaptive_threshold(color_image=image)
            image = self.erosion_dilation(gray_image=image)
            image = np.asarray(image, dtype=np.float32)
        else:
            image = np.asarray(image, dtype=np.float32)
            image = self.scale(image)

        augmented_images, labels = [image], [label]

        if transform_type == "train": # <--- special token to augment the images
            augmented_images, labels = [], []
            for f in (self.identity, self.flip_vertical, self.flip_horizontal, self.rotate):
                augmented_images.append(f(image))
                labels.append(label)

        return augmented_images, labels

    def random_transform(self, image):
        images, labels = self.apply_transform(image, label=0, transform_type="train")

        for i in range(min(len(images), 4)):
            plt.subplot(2, 2, i + 1)
            plt.imshow(images[i])
        plt.show()


if __name__ == "__main__":
    array = cv2.imread("poza.png")

    augmenter = DataAugmenter(images=[array], labels=[0], transform_type="train", special_preprocessing=False)
    augmenter.random_transform(array)
    
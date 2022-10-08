import cv2
import numpy as np


class left_right:

    def __init__(self, image):
        self.image = image
        self.height, self.width, _ = self.image.shape
        self.crop = 100
        self.thresh = 200
        self.kernel = np.ones((5, 5), np.uint8)
        self.iterations = 1

    def _processing_image(self):
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_crop = image_gray[self.crop:self.height - self.crop, self.crop:self.width - self.crop]
        _, thresh = cv2.threshold(img_crop.copy(), self.thresh, 255, 0)
        img_dilation = cv2.dilate(thresh, self.kernel, iterations=self.iterations)
        contours, _ = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _processing_points(self):
        contours = self._processing_image()
        list_point = list(contours[0])

        for l in contours[2:]:
            list_point += list(l)

        list_point_2 = []
        for p in list_point:
            list_point_2.append(p[0])

        list_point_3 = np.array(list_point_2)
        return list_point_3

    def _get_corner_points(self):
        list_point = self._processing_points()
        i_a = np.argmin(list_point[:, 0])
        i_b = np.argmax(list_point[:, 1])
        i_c = np.argmin(list_point[:, 1])
        A = list_point[i_a]
        B = list_point[i_b]
        C = list_point[i_c]
        td = [(B[0] - C[0]) // 2 + C[0], (B[1] - C[1]) // 2 + C[1]]
        return td, A

    def classify(self):
        td, A = self._get_corner_points()

        if td[0] - A[0] > (self.width - self.crop*2) / 2:
            return True
        else:
            return False


if __name__ == '__main__':
    image = '/home/sonnda/ANSON/Mouse-Video-Using-Yolact/video/24_/out_video/55.jpg'
    left_right_object = left_right(image)

    if left_right_object.classify():
        print('LEFT')
    else:
        print('RIGHT')

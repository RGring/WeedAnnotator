import numpy as np
import cv2


class PostProcessor():
    @staticmethod
    def post_open(pred_mask):
        post_mask = np.zeros_like(pred_mask)
        ind = np.where((np.any(pred_mask != (0, 0, 0), axis=-1)))
        if len(ind[0]) > 0:
            maskcolor = pred_mask[ind[0][0], ind[1][0]]
        else:
            return post_mask

        kernel = np.ones((25, 25), np.uint8)
        ret, mask = cv2.threshold(cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY), 0, 255, 0)
        post_mask_temp = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        post_mask[post_mask_temp == 255, :] = maskcolor
        return post_mask

    @staticmethod
    def post_graphcut(img, pred_mask):
        post_mask = np.zeros_like(pred_mask)
        ind = np.where((np.any(pred_mask != (0, 0, 0), axis=-1)))
        if len(ind[0]) > 0:
            maskcolor = pred_mask[ind[0][0], ind[1][0]]
        else:
            return post_mask

        # Getting binary mask
        ret, mask = cv2.threshold(cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY), 0, 255, 0)
        # Foreground
        erosion_size = 50
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                            (erosion_size, erosion_size))
        sure_fg = cv2.erode(mask, element, iterations=1)

        # Background
        bg = cv2.bitwise_not(mask)
        erosion_size = 90
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                            (erosion_size, erosion_size))
        sure_bg = cv2.erode(bg, element, iterations=1)

        markers = np.ones(img.shape[:2], np.uint8) * 2
        markers[mask == 255] = 3
        markers[sure_fg == 255] = 1
        markers[sure_bg == 255] = 0
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            new_mask, bgdModel, fgdModel = cv2.grabCut(img, markers, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        except:
            return post_mask
        post_mask[new_mask == 1, :] = maskcolor
        post_mask[new_mask == 3, :] = maskcolor
        return post_mask

    @staticmethod
    def post_watershed(img, pred_mask):
        post_mask = np.zeros_like(pred_mask)
        ind = np.where((np.any(pred_mask != (0, 0, 0), axis=-1)))
        if len(ind[0]) > 0:
            maskcolor = pred_mask[ind[0][0], ind[1][0]]
        else:
            return post_mask
        # Getting binary mask
        ret, mask = cv2.threshold(cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY), 0, 255, 0)

        # Background
        erosion_size = 90
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                            (erosion_size, erosion_size))
        sure_bg = cv2.dilate(mask, element, iterations=1)

        # Foreground
        erosion_size = 50
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                            (erosion_size, erosion_size))
        sure_fg = cv2.dilate(cv2.bitwise_not(mask), element, iterations=1)

        # Marker Creation
        unknown = cv2.subtract(sure_bg, cv2.bitwise_not(sure_fg))
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        new_mask = cv2.watershed(img, markers)

        post_mask[new_mask == 1, :] = maskcolor
        return post_mask

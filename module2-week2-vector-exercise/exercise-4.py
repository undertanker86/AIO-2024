import numpy as np
import cv2


def compute_difference(bg_img, input_img):
    difference = cv2.absdiff(bg_img, input_img)
    difference_single_channel = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    return difference_single_channel


def compute_binary_mask(difference_single_channel):
    _, difference_binary = cv2.threshold(
        difference_single_channel, 50, 255, cv2.THRESH_BINARY)
    return difference_binary


def replace_background(bg1_image, bg2_image, ob_image):
    difference_single_channel = compute_difference(bg1_image, ob_image)
    binary_mask = compute_binary_mask(difference_single_channel)
    output = np.where(binary_mask[:, :, np.newaxis]
                      == 255, ob_image, bg2_image)
    return output


if __name__ == "__main__":

    bg1_image = cv2.imread(
        'D:\AIO-2024-WORK\AIO-2024\module2-week2-vector-exercise\GreenBackground.png', 1)

    bg1_image = cv2.resize(bg1_image, (678, 381))

    ob_image = cv2.imread(
        'D:\AIO-2024-WORK\AIO-2024\module2-week2-vector-exercise\Object.png', 1)
    ob_image = cv2.resize(ob_image, (678, 381))

    bg2_image = cv2.imread(
        'D:\AIO-2024-WORK\AIO-2024\module2-week2-vector-exercise\Object.png', 1)
    bg2_image = cv2.resize(bg2_image, (678, 381))

    difference_single_channel = compute_difference(bg1_image, ob_image)

    output_image = replace_background(bg1_image, bg2_image, ob_image)

    cv2.imshow('Final Output', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

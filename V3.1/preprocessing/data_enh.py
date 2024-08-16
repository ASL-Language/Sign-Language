import os
import cv2
import numpy as np

input_path="./datas/asl_alphabet_train_sampled"
output_path="./datas/asl_alphabet_test"
if not os.path.exists(output_path):
    os.makedirs(output_path)


def single_scale_retinex(img, sigma):
    retinex = np.log10(img + 1.0) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1.0)
    return retinex

def retinex_enhancement(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    retinex = single_scale_retinex(img, sigma=50)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(retinex, cv2.COLOR_GRAY2BGR)

# def main():
for root, dirs, files in os.walk(input_path):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            
            if img is not None:
                enhanced_img = retinex_enhancement(img)
                
                relative_path = os.path.relpath(root, input_path)
                output_dir = os.path.join(output_path, relative_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                output_file_path = os.path.join(output_dir, file)
                cv2.imwrite(output_file_path, enhanced_img)


# main()

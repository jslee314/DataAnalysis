import glob
import cv2
import numpy as np


def getDicFromMask(src_absorption="D:/2. data/total_iris/dataVoucher_right_mask/absorptionring/",
                   src_cholesterol="D:/2. data/total_iris/dataVoucher_right_mask/cholesterolring/",
                   src_stress="D:/2. data/total_iris/dataVoucher_right_mask/stressring/"):
    absorption_paths = sorted(glob.glob(src_absorption + "*.bmp"))
    cholesterol_paths = sorted(glob.glob(src_cholesterol + "*.bmp"))
    stress_paths = sorted(glob.glob(src_stress + "*.bmp"))

    absorption_dic, cholesterol_dic, stress_dic = {}, {}, {}

    for absorption_path, cholesterol_path, stress_path in zip(absorption_paths, cholesterol_paths, stress_paths):
        imageName = absorption_path.split("\\")[-1][:-4]

        img_absorption = cv2.imread(absorption_path, 0)
        img_cholesterol = cv2.imread(cholesterol_path, 0)
        img_stress = cv2.imread(stress_path, 0)

        array_absorption = np.array(img_absorption)
        array_cholesterol = np.array(img_cholesterol)
        array_stress = np.array(img_stress)

        absorption_dic[imageName] = (array_absorption > 0).sum()
        cholesterol_dic[imageName] = (array_cholesterol > 0).sum()
        stress_dic[imageName] = (array_stress > 0).sum()

        # cv2.putText(img_absorption, str(absorption_dic[imageName]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(img_cholesterol, str(cholesterol_dic[imageName]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(img_stress, str(stress_dic[imageName]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.imshow("img_absorption", img_absorption)
        # cv2.imshow("img_cholesterol", img_cholesterol)
        # cv2.imshow("img_stress", img_stress)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    return absorption_dic, cholesterol_dic, stress_dic



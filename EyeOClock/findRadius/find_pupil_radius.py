import cv2
import glob
import numpy as np
import math

class FindRadius:
    def __init__(self, pupilSeg, irisSeg):
        self.pupilSeg = pupilSeg
        self.irisSeg = irisSeg

    def undesired_objects (self, image):
        image = image.astype('uint8')
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=4)
        sizes = stats[:, -1]

        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        largest_connectedComponent = np.zeros(image.shape, dtype='uint8')
        largest_connectedComponent[output == max_label] = 255
        # cv2.imshow("Biggest component", largest_connectedComponent)
        # cv2.waitKey()
        return largest_connectedComponent

    def get_centor(self, gray):
            contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            contour = contours[0]
            contour = np.array(contour).reshape(-1, 2)

            # 모멘트란,
            # 어떤 종류의 "물리적 효과"가 하나의 물리량 뿐만 아니라 그 물리량의 "분포상태"에 따라서 정해질 때, 정의되는 양
            # n차 모멘트 = (위치)^n(물리량)
            M = cv2.moments(contour)

            # 도심(무게중심)이란,
            # 단면의 직교좌표축에 대한 단면 1차 모멘트가 0이 되는 점이다.
            # 직교좌표축에서 도심까지의 거리를 구하는 방법은, 단면 1차 모멘트를 도형의 면적으로 나눈다.
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            return cX, cY

    def get_radius(self, gray, cX, cY):
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contour = contours[0]
        contour = np.array(contour).reshape(-1, 2)

        # 원의 반지름 : 도형의 contour의 좌표들과 도심간의 거리의 평균
        sum_radius = 0
        for x, y in contour:
            sum_radius = sum_radius + math.sqrt((x - cX) * (x - cX) + (y - cY) * (y - cY))

        mean_radius = round(sum_radius / len(contour))

        return mean_radius

    def getAutoNerveWaveRadious(self, pupilRadius,irisRadius):
        aa = (irisRadius - pupilRadius)/3;
        a = round(aa *1.2);
        return pupilRadius + a;

    def RadiusAndCentor(self):
        pupilLargestConnectedComponent = self.undesired_objects(self.pupilSeg)
        irisLargestConnectedComponent = self.undesired_objects(self.irisSeg)
        # cv2.imshow("pupilLargestConnectedComponent", pupilLargestConnectedComponent)
        # cv2.imshow("irisLargestConnectedComponent", irisLargestConnectedComponent)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 2차원 분포의 1차 모멘트값 산정하기 -> 예측원의 중심 구하기
        cX, cY = self.get_centor(pupilLargestConnectedComponent)

        # 도형의 contour의 좌표들과 도심간의 거리의 평균 -> 예측원의 반지름 구하기
        pupilRadius = self.get_radius(pupilLargestConnectedComponent, cX, cY)
        irisRadius = self.get_radius(irisLargestConnectedComponent, cX, cY)
        autoNerveWaveRadius = self.getAutoNerveWaveRadious(pupilRadius,irisRadius)
        return cX, cY, autoNerveWaveRadius, irisRadius

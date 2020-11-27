import cv2
import numpy as np
import matplotlib.pyplot as plt

organImages = {"brain": [], "kidney": [], "liver": [], "lung": []}
# (동공: 0, 자율신경환: 1, 첫번째 섹터: 2, 두번째 섹터: 3, 세번째 섹터: 4, 홍채: 5)

organRightSectors = {"brain": {"startAngle": 336, "endAngle": 28.5, "startSector": 1, "endSector": 5},
                    "kidney": {"startAngle": 163.5, "endAngle": 175.5, "startSector": 1, "endSector": 5},
                    "liver": {"startAngle": 229.5, "endAngle": 237, "startSector": 1, "endSector": 5},
                    "lung": {"startAngle": 270, "endAngle": 297, "startSector": 1, "endSector": 4}}
organLeftSectors = {"brain": {"startAngle": 331.5, "endAngle": 24, "startSector": 1, "endSector": 5},
                    "kidney": {"startAngle": 184.5, "endAngle": 196.5, "startSector": 1, "endSector": 5},
                    "lung": {"startAngle": 63, "endAngle": 90, "startSector": 1, "endSector": 5},}
SectorValue = {}


def calculateAngle(angle):
    angle = angle - 90  # 안드로이드와 파이썬과의 기준 차이
    if 0<=angle and angle<=360:
        return angle
    else:
        return (angle+360) % 360


def undesired_objects(image):
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

import copy
def cropRegion(img, thr):
    thr = undesired_objects(thr)
    temp_thr = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
    blue_img = np.where(temp_thr == 255, img, (255, 0, 0))
    blue_img = blue_img.astype(np.uint8)
    # cv2.imshow("blue_img", blue_img)
    # cv2.imshow("g", thr)
    # cv2.waitKey()

    contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    len = w if w > h else h
    dst = blue_img[y:y + len, x:x + len]
    return dst


def getOrganSector(total_segmentation, cX, cY, autoNerveWaveRadius, irisRadius, pattern_input_width, pattern_input_height):
    # 장기별 Sector값(딕셔너리) 초기화
    btwRadius = round((irisRadius - autoNerveWaveRadius)/4)
    for idx in range(1, 6):
        SectorValue[idx] = autoNerveWaveRadius + (btwRadius*(idx-1))
    #print(SectorValue)

    # 장기별 ROI이미지 저장
    for organ, organSector in organRightSectors.items():
        sector = total_segmentation.copy()
        #print(organ)
        startSector = SectorValue[organSector["startSector"]]
        endSector = SectorValue[organSector["endSector"]]
        startAngle = calculateAngle(organSector["startAngle"])
        endAngle = calculateAngle(organSector["endAngle"])
        #print(str(startSector) + ", " + str(endSector) + ", " + str(startAngle) + ", " + str(endAngle))

        cv2.ellipse(img=sector, center=(cX, cY), axes=(endSector, endSector), angle=0, startAngle=startAngle, endAngle=endAngle, color=(255, 0, 0), thickness=-1)
        cv2.ellipse(img=sector, center=(cX, cY), axes=(startSector, startSector), angle=0, startAngle=startAngle, endAngle=endAngle, color=(0, 0, 0), thickness=-1)
        image_sector = np.where(sector == (255, 0, 0), total_segmentation, 0)

        # # 1> 특정각도에서 홍채이미지
        image_sector_gray = cv2.cvtColor(image_sector, cv2.COLOR_BGR2GRAY)
        rct, thr = cv2.threshold(image_sector_gray, 1, 255, cv2.THRESH_BINARY)
        dstImg = cropRegion(image_sector, thr)
        resized_dstImg = cv2.resize(dstImg, (pattern_input_height, pattern_input_width))

        dstorganImg = cv2.cvtColor(resized_dstImg, cv2.COLOR_BGR2RGB)

        organImages[organ] = dstorganImg

    return organImages

def getOrganRegion(imageBGR, pupilSeg, irisSeg, cX, cY, autoNerveWaveRadius, irisRadius, pattern_input_width, pattern_input_height):

    # pupil_seg = np.where(pupilSeg > 0.5, 0, imageBGR)
    # iris_seg = np.where(irisSeg > 0.5, pupil_seg, 0)

    # ROI 그리기
    pupil_segmentation = cv2.circle(img=np.zeros((480, 640, 3)), center=(cX, cY), radius=autoNerveWaveRadius, color=(255, 255, 255), thickness=-1)
    iris_segmentation = cv2.circle(img=np.zeros((480, 640, 3)), center=(cX, cY), radius=irisRadius, color=(255, 255, 255), thickness=-1)

    total_segmentation = np.where(pupil_segmentation == 255, 0, imageBGR)
    total_segmentation = np.where(iris_segmentation == 255, total_segmentation, 0)
    total_segmentation_rgb = cv2.cvtColor(total_segmentation, cv2.COLOR_BGR2RGB)

    # 장기 영역 섹터별로 구하기
    return getOrganSector(total_segmentation, cX, cY, autoNerveWaveRadius, irisRadius, pattern_input_width, pattern_input_height)









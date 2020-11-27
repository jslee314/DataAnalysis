import glob
from PIL import Image
import numpy as np
from EyeOClock.findRadius.PupilIrisSegmentation import SegPupilIris
from EyeOClock.findRadius.find_pupil_radius import FindRadius
from EyeOClock.organPattern.OrganPatternClassifier import PatternClassifier
from EyeOClock.organPattern.getOrganRegion import getOrganRegion
from EyeOClock.unknown.UnknownClassifer import UnknownClassifier
import cv2

input_width = 640
input_height = 480

pattern_input_width = 240
pattern_input_height = 240
size = input_width, input_height


def getPatternData(src_dir="D:/2. data/total_iris/dataVoucherSample/"):

    imagePaths = sorted(glob.glob(src_dir + "*.png"))

    imageNames, imageRGBs, imageBGRs, imageNumpy = [], [], [], []

    # 이미지 파일 읽어어서 리스트(imageLists)로 저장
    for imagePath in imagePaths:
        imageNames.append(imagePath.split("\\")[-1][:-4])
        imageRGB = Image.open(imagePath).convert("RGB")
        img_sector = np.where(imageRGB == (255, 0, 0), imageRGB, 0)
        # rgb = Image.new("RGB", argb.size, (255, 255, 255))
        # rgb.paste(argb, mask=argb.split()[3])  # 3 is the alpha channel
        imageRGB = imageRGB.resize(size, resample=Image.BICUBIC)
        imageRGBs.append(imageRGB)
        imageNumpy.append(np.array(imageRGB))
        # for show
        image_bgr = cv2.imread(imagePath)
        imageBGRs.append(image_bgr)
    data = np.array(imageNumpy, dtype="float") / 255.0

    # 동공과 홍채영역 segmentation
    segPupilIris = SegPupilIris(data=data, imageLists=imageRGBs,
                                input_width=input_width, input_height=input_height)
    segPupilIris.getPupilMasks(iris_h5model="findRadius/model/h5/unet_pupil_weight.h5")
    segPupilIris.getIrisMasks(iris_h5model="findRadius/model/h5/unet_iris_640_480_weight.h5")
    pupilSegs, irisSegs = segPupilIris.getMasks()
    # brainImageList, kidneyImageList, liverImageList, lungImageList = [], [], [], []
    organImageList = []
    # 동공의 중심, 반지름, 홍채의 반지름 구함
    for imageBGR, pupilSeg, irisSeg in zip(imageBGRs, pupilSegs, irisSegs):
        findRadius = FindRadius(pupilSeg=pupilSeg, irisSeg=irisSeg)
        cX, cY, autoNerveWaveRadius, irisRadius = findRadius.RadiusAndCentor()
        #print(cX, cY, autoNerveWaveRadius, irisRadius)
        # 예측원과 중심 그리기
        # cv2.circle(img=image, center=(cX, cY), radius=3, color=(255, 0, 0), thickness=-1)
        # cv2.circle(img=image, center=(cX, cY), radius=pupilRadius, color=(0, 0, 255), thickness=1)
        # cv2.circle(img=image, center=(cX, cY), radius=irisRadius, color=(0, 255, 0), thickness=1)
        # cv2.imshow("pupilDst", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # 장기 영역 분석
        # 장기별 crop된 영상
        # brainImage, kidneyImage, liverImage, lungImage = getOrganRegion(imageBGR, pupilSeg, irisSeg, cX, cY, autoNerveWaveRadius, irisRadius, pattern_input_width, pattern_input_height)
        # brainImageList.append(brainImage)
        # kidneyImageList.append(kidneyImage)
        # liverImageList.append(liverImage)
        # lungImageList.append(lungImage)
        organImages = getOrganRegion(imageBGR, pupilSeg, irisSeg, cX, cY, autoNerveWaveRadius, irisRadius, pattern_input_width, pattern_input_height)
        organImageList.extend(organImages.values())
        for organ, organImage in organImages.items():
            organImage_rgb = cv2.cvtColor(organImage, cv2.COLOR_BGR2RGB)
            # cv2.imshow(organ, organImage)
            # cv2.imshow(organ+"_rgb", organImage_rgb)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    ################### unknown data 걸러냄
    classifyUnknownData = UnknownClassifier(input_width=pattern_input_width, input_height=pattern_input_width,
                                        h5_path="unknown/unknown_mobilenetv3.h5")
    data = np.array(organImageList, dtype="float") / 255.0
    unknownDataResult = classifyUnknownData.classifyUnknownData(data=data, organImageList=organImageList)

    ################### organPattern
    classifyPattern = PatternClassifier(input_width=pattern_input_width, input_height=pattern_input_height,
                                        h5_path="organPattern/pattern_efficientnet-b1.h5")
    data = np.array(organImageList, dtype="float") / 255.0
    organResult = classifyPattern.classifyPattern(data=data, organImageList=organImageList)

    # # 해당 장기 영역이 unknown 인 경우 normal 에 -1 값 넣는다.
    # for organ, unknown in zip(organResult, unknownDataResult):
    #     if unknown[1] > 50:
    #         organ[2] = -1

    organResult = np.reshape(organResult, (-1, 4, 5))   # (data수=-1, organ수=4, pattern수=5)

    brain_defects = organResult[:, 0, 0]
    brain_lacunas = organResult[:, 0, 1]
    brain_normals = organResult[:, 0, 2]
    brain_spokes = organResult[:, 0, 3]
    brain_spots = organResult[:, 0, 4]

    kidney_defects = organResult[:, 1, 0]
    kidney_lacunas = organResult[:, 1, 1]
    kidney_normals = organResult[:, 1, 2]
    kidney_spokes = organResult[:, 1, 3]
    kidney_spots = organResult[:, 1, 4]

    liver_defects = organResult[:, 2, 0]
    liver_lacunas = organResult[:, 2, 1]
    liver_normals = organResult[:, 2, 2]
    liver_spokes = organResult[:, 2, 3]
    liver_spots = organResult[:, 2, 4]

    lung_defects = organResult[:, 3, 0]
    lung_lacunas = organResult[:, 3, 1]
    lung_normals = organResult[:, 3, 2]
    lung_spokes = organResult[:, 3, 3]
    lung_spots = organResult[:, 3, 4]


    (brain_defect_dic, brain_lacuna_dic, brain_normal_dic, brain_spoke_dic, brain_spot_dic,
     kidney_defect_dic, kidney_lacuna_dic, kidney_normal_dic, kidney_spoke_dic, kidney_spot_dic,
     liver_defect_dic, liver_lacuna_dic, liver_normal_dic, liver_spoke_dic, liver_spot_dic,
         lung_defect_dic, lung_lacuna_dic, lung_normal_dic, lung_spoke_dic, lung_spot_dic) = ({}, {}, {}, {}, {},    {}, {}, {}, {}, {},   {}, {}, {}, {}, {},    {}, {}, {}, {}, {})

    for (imgName, imageBGR,
         brain_defect, brain_lacuna, brain_normal, brain_spoke, brain_spot,
         kidney_defect, kidney_lacuna, kidney_normal, kidney_spoke, kidney_spot,
         liver_defect, liver_lacuna, liver_normal, liver_spoke, liver_spot,
         lung_defect, lung_lacuna, lung_normal, lung_spoke, lung_spot) in zip(imageNames, imageBGRs,
                                                                              brain_defects, brain_lacunas, brain_normals, brain_spokes, brain_spots,
                                                                              kidney_defects, kidney_lacunas, kidney_normals, kidney_spokes, kidney_spots,
                                                                              liver_defects, liver_lacunas, liver_normals, liver_spokes, liver_spots,
                                                                              lung_defects, lung_lacunas, lung_normals, lung_spokes, lung_spots):
        (brain_defect_dic[imgName], brain_lacuna_dic[imgName], brain_normal_dic[imgName], brain_spoke_dic[imgName], brain_spot_dic[imgName],
         kidney_defect_dic[imgName], kidney_lacuna_dic[imgName], kidney_normal_dic[imgName], kidney_spoke_dic[imgName], kidney_spot_dic[imgName],
         liver_defect_dic[imgName], liver_lacuna_dic[imgName], liver_normal_dic[imgName], liver_spoke_dic[imgName], liver_spot_dic[imgName],
         lung_defect_dic[imgName], lung_lacuna_dic[imgName], lung_normal_dic[imgName], lung_spoke_dic[imgName], lung_spot_dic[imgName]) \
            = (brain_defect, brain_lacuna, brain_normal, brain_spoke, brain_spot,
         kidney_defect, kidney_lacuna, kidney_normal, kidney_spoke, kidney_spot,
         liver_defect, liver_lacuna, liver_normal, liver_spoke, liver_spot,
         lung_defect, lung_lacuna, lung_normal, lung_spoke, lung_spot)

        # cv2.putText(imageBGR, "brain_defect " + str(brain_defect_dic[imgName]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(imageBGR, "brain_lacuna " + str(brain_lacuna_dic[imgName]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(imageBGR, "brain_normal " + str(brain_normal_dic[imgName]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(imageBGR, "brain_spoke " + str(brain_spoke_dic[imgName]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(imageBGR, "brain_spot " + str(brain_spot_dic[imgName]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        #
        # cv2.putText(imageBGR,  "kidney_defect " + str(kidney_defect_dic[imgName]), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(imageBGR,  "kidney_lacuna " + str(kidney_lacuna_dic[imgName]), (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(imageBGR,  "kidney_normal " + str(kidney_normal_dic[imgName]), (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(imageBGR,  "kidney_spoke " + str(kidney_spoke_dic[imgName]), (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(imageBGR,  "kidney_spot " + str(kidney_spot_dic[imgName]), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        #
        # cv2.imshow("txt" + imgName, imageBGR)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    return (brain_defect_dic, brain_lacuna_dic, brain_normal_dic, brain_spoke_dic, brain_spot_dic,
     kidney_defect_dic, kidney_lacuna_dic, kidney_normal_dic, kidney_spoke_dic, kidney_spot_dic,
     liver_defect_dic, liver_lacuna_dic, liver_normal_dic, liver_spoke_dic, liver_spot_dic,
         lung_defect_dic, lung_lacuna_dic, lung_normal_dic, lung_spoke_dic, lung_spot_dic)

    ################### Stress
    # from EyeOClock.stressCholesterolAbsorption import Loss, DropConnect, DropBlock, instance_normalization, DeepLabV3PlusWithFPN
    # import EyeOClock.modelUtil as util
    # import tensorflow as tf

    # stress_model = models.load_model('stressCholesterolAbsorption/model/stressring_model_datavoucher2.h5',
    #                                  custom_objects={
    #                            'InstanceNormalization': instance_normalization.InstanceNormalization,
    #                            'focal_tversky_loss': Loss.focal_tversky_loss,
    #                            'dsc': Loss.dsc,
    #                            'DropConnect': DropConnect.DropConnect,
    #                            'DropBlock2D': DropBlock.DropBlock2D,
    #                            'HardSwish': util.HardSwish,
    #                            'resize_images': tf.keras.backend.resize_images,
    #                            'DeepLabV3PlusWithFPN': DeepLabV3PlusWithFPN.DeepLabV3PlusWithFPN})

    # stress_model = models.load_model('stressCholesterolAbsorption/model/stressring_model_datavoucher2.h5',
    #                                  custom_objects={
    #                            'focal_tversky_loss': Loss.focal_tversky_loss,
    #                            'DropConnect': DropConnect.DropConnect,
    #                            'DropBlock2D': DropBlock.DropBlock2D,
    #                            'dsc': Loss.dsc,
    #                            'HardSwish': util.HardSwish,
    #                            'resize_images': tf.keras.backend.resize_images,
    #                            'DeepLabV3PlusWithFPN': DeepLabV3PlusWithFPN.DeepLabV3PlusWithFPN
    #                                  })
    # stress_model.summary()
    # stressPredicts = stress_model.predict(data, batch_size=4)
    #
    # for image, stressPredict in zip(imageRGBs, stressPredicts):
    #     cv2.imshow("imageRGB", imageRGB)
    #     cv2.imshow("stressPredict", stressPredict)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

        # cv2.imshow("image", image.astype(np.uint8))
        # cv2.imshow("stressPredict", stressPredict.astype(np.uint8))

    ################### Cholesterol
    # cholesterol_model = models.load_model("stressCholesterolAbsorption/model/cholesterolring_model_deeplabv3plus-with-fpn.h5")
    # cholesterolPredicts = cholesterol_model.predict(data, batch_size=4)

    ################### Absorption
    # absorption_model = models.load_model("stressCholesterolAbsorption/model/absorptionring_model_deeplabv3plus-with-fpn-2.h5")
    # absorptionPredicts = absorption_model.predict(data, batch_size=4)
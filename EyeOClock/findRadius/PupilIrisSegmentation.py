import numpy as np
from EyeOClock.findRadius.model.UNet import UNet
import glob
from PIL import Image
import cv2


class SegPupilIris:
    def __init__(self, data, imageLists, input_width, input_height):
        self.data = data
        self.imageLists = imageLists
        self.input_width = input_width
        self.input_height = input_height

    def getPupilMasks(self, iris_h5model):
        ####### pupil segmentation #######
        m = UNet(nClasses=1, input_height=self.input_height, input_width=self.input_width)
        m.load_weights(iris_h5model)
        self.pupilSegPredicts = m.predict(self.data, batch_size=2)

    def getIrisMasks(self, iris_h5model):
        ####### iris segmentation #######
        m = UNet(nClasses=1, input_height=self.input_height, input_width=self.input_width)
        m.load_weights(iris_h5model)
        self.irisSegPredicts = m.predict(self.data, batch_size=4)


    def getMasks(self):
        pupilSegs = []
        irisSegs = []
        for image, pupilSegPredict, irisSegPredict in zip(self.imageLists, self.pupilSegPredicts, self.irisSegPredicts):
            # cv2.imshow("image", image)
            # cv2.imshow("irisSegPredict", irisSegPredict)
            # cv2.imshow("pupilSegPredict", pupilSegPredict)
            pupilSeg = np.where(pupilSegPredict > 0.5, 255, 0)
            irisSeg = np.where(irisSegPredict > 0.5, 255, 0)
            # ret, pupilSeg = cv2.threshold(pupilSegPredict, 0.5, 255, cv2.THRESH_BINARY)
            # ret, irisSeg = cv2.threshold(irisSegPredict, 0.5, 255, cv2.THRESH_BINARY)

            # cv2.imshow("pupilSegs", pupilSeg.astype(np.uint8))
            # cv2.imshow("irisSegs", irisSeg.astype(np.uint8))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            pupilSegs.append(pupilSeg)
            irisSegs.append(irisSeg)
        return pupilSegs, irisSegs
    
def undesired_objects (image):
    image = image.astype('uint8')
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]
    print(len(sizes))
    largest_connectedComponent = image
    if len(sizes) > 1:
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
    else:
        largest_connectedComponent = image
    return largest_connectedComponent


if __name__ == '__main__':
    src_dir = "D:/2. data/wt/unidentifiedData144/"
    dst_iris = "D:/2. data/wt/unidentifiedData_dst_iris/"
    dst_pupil = "D:/2. data/wt/dst_pupil/"
    input_width = 640
    input_height = 480
    size = input_width, input_height
    imagePaths = sorted(glob.glob(src_dir + "*.png"))
    imageNames, imageRGBs, imageBGRs, imageNumpy, imageDims = [], [], [], [], []

    # 이미지 파일 읽어어서 리스트(imageLists)로 저장
    for imagePath in imagePaths:
        imageNames.append(imagePath.split("\\")[-1][:-4])
        imageRGB = Image.open(imagePath).convert("RGB")
        imageDims.append([np.array(imageRGB).shape[1], np.array(imageRGB).shape[0]])
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
    segPupilIris.getPupilMasks(iris_h5model="model/h5/unet_pupil_weight.h5")
    segPupilIris.getIrisMasks(iris_h5model="model/h5/unet_iris_640_480_weight.h5")
    pupilSegs, irisSegs = segPupilIris.getMasks()
    # pupilSegs, irisSegs = segPupilIris.getMaskImages()

    for imageBGR, imageName, pupilSeg, irisSeg, imageDim in zip(imageBGRs, imageNames, pupilSegs, irisSegs, imageDims):
        pupilLargestConnectedComponent = undesired_objects(pupilSeg)
        irisLargestConnectedComponent = undesired_objects(irisSeg)
        resized_pupil = cv2.resize(pupilLargestConnectedComponent.astype(np.uint8), tuple(imageDim))
        resized_iris = cv2.resize(irisLargestConnectedComponent.astype(np.uint8), tuple(imageDim))

        BGR_pupil = cv2.cvtColor(resized_pupil, cv2.COLOR_GRAY2BGR)
        BGR_iris = cv2.cvtColor(resized_iris, cv2.COLOR_GRAY2BGR)

        pupilSeg = np.where(BGR_pupil==[255, 255, 255], [255, 0, 0], imageBGR)
        irisSeg = np.where(BGR_iris==[255, 255, 255], [255, 0, 0], imageBGR)

        # cv2.imshow("pupilSegs", resized_irisSeg)
        # cv2.imshow("irisSegs", resized_pupil)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # cv2.imwrite(dst_pupil+imageName+'.png', pupilSeg)
        cv2.imwrite(dst_iris+imageName+'.png', irisSeg)


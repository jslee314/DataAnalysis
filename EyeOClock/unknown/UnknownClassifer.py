import EyeOClock.modelUtil as util


from tensorflow.keras import models
import cv2
class UnknownClassifier:
    def __init__(self, input_width, input_height, h5_path):
        self.input_width = input_width
        self.input_height = input_height
        self.model = models.load_model(filepath=h5_path,
                                  custom_objects={'Relu6': util.Relu6,
                                                  'HardSwish': util.HardSwish,
                                                  "DropConnect": util.DropConnect,
                                                  "RectifiedAdam": util.RectifiedAdam})


    def classifyUnknownData(self, data, organImageList):
        self.model.summary()
        preds = self.model.predict(data, batch_size=32)
        organ_lists = ['brain', 'kidney', 'liver', 'lung']

        pattern_lists = ['known', 'unknown']
        for i, prediction in enumerate(preds):
            # print(organ_lists[i%4])
            # organImage_rgb = cv2.cvtColor(organImageList[i], cv2.COLOR_BGR2RGB)
            # cv2.putText(organImage_rgb, "{} {}: {:.2f}%".format(organ_lists[i%4], pattern_lists[0], preds[i][0] * 100), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.imshow(organ_lists[i%4]+" / " + pattern_lists[0], organImage_rgb)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            for pattern_idx, pattern_list in enumerate(pattern_lists):
                print("{} >> {} {}: {:.2f}%".format(i, pattern_idx, pattern_list, preds[i][pattern_idx] * 100))
                preds[i][pattern_idx] = round(preds[i][pattern_idx], 2)*100
        return preds




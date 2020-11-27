import pandas as pd
from EyeOClock.stressCholesterolAbsorption.getDataFromMask import getDicFromMask
from EyeOClock.organPattern.getPatternData import getPatternData



(brain_defect_dic, brain_lacuna_dic, brain_normal_dic, brain_spoke_dic, brain_spot_dic,
 kidney_defect_dic, kidney_lacuna_dic, kidney_normal_dic, kidney_spoke_dic, kidney_spot_dic,
 liver_defect_dic, liver_lacuna_dic, liver_normal_dic, liver_spoke_dic, liver_spot_dic,
 lung_defect_dic, lung_lacuna_dic, lung_normal_dic, lung_spoke_dic, lung_spot_dic) = getPatternData(src_dir="D:/2. data/total_iris/dataVoucherSample/")
#src_dir="D:/2. data/total_iris/dataVoucher_right/"

brain_defect = pd.Series(brain_defect_dic)
brain_lacuna = pd.Series(brain_lacuna_dic)
brain_normal = pd.Series(brain_normal_dic)
brain_spoke = pd.Series(brain_spoke_dic)
brain_spot = pd.Series(brain_spot_dic)

kidney_defect = pd.Series(kidney_defect_dic)
kidney_lacuna = pd.Series(kidney_lacuna_dic)
kidney_normal = pd.Series(kidney_normal_dic)
kidney_spoke = pd.Series(kidney_spoke_dic)
kidney_spot = pd.Series(kidney_spot_dic)

liver_defect = pd.Series(liver_defect_dic)
liver_lacuna = pd.Series(liver_lacuna_dic)
liver_normal = pd.Series(liver_normal_dic)
liver_spoke = pd.Series(liver_spoke_dic)
liver_spot = pd.Series(liver_spot_dic)

lung_defect = pd.Series(lung_defect_dic)
lung_lacuna = pd.Series(lung_lacuna_dic)
lung_normal = pd.Series(lung_normal_dic)
lung_spoke = pd.Series(lung_spoke_dic)
lung_spot = pd.Series(lung_spot_dic)

absorption_dic, cholesterol_dic, stress_dic = getDicFromMask(
    src_absorption="D:/2. data/total_iris/dataVoucherSample_mask/absorptionring/",
    src_cholesterol="D:/2. data/total_iris/dataVoucherSample_mask/cholesterolring/",
    src_stress="D:/2. data/total_iris/dataVoucherSample_mask/stressring/")

absorption = pd.Series(absorption_dic)
cholesterol = pd.Series(cholesterol_dic)
stress = pd.Series(stress_dic)



irisImageData = pd.DataFrame({
        'brain_defect': brain_defect, 'brain_lacuna': brain_lacuna, 'brain_normal': brain_normal, 'brain_spoke': brain_spoke, 'brain_spot': brain_spot,
        'kidney_defect': kidney_defect, 'kidney_lacuna': kidney_lacuna, 'kidney_normal': kidney_normal, 'kidney_spoke': kidney_spoke, 'kidney_spot': kidney_spot,
        'liver_defect': liver_defect, 'liver_lacuna': liver_lacuna, 'liver_normal': liver_normal, 'liver_spoke': liver_spoke, 'liver_spot': liver_spot,
        'lung_defect': lung_defect, 'lung_lacuna': lung_lacuna, 'lung_normal': lung_normal, 'lung_spoke': lung_spoke, 'lung_spot': lung_spot,
        'absorption': absorption, 'cholesterol': cholesterol, 'stress': stress})


print(irisImageData.index)
print(irisImageData.columns)
irisImageData.to_excel('irisImageDataSample.xlsx')


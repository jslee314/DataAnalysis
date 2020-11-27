import glob
import os.path
import shutil
import pandas as pd
# srcDir = "D:/2. data/irisWithEmr/DataVoucher/"
# dstDir = "D:/4. sourceRepository/python_repository/OCR/dataVoucher_/"
# files = os.listdir(srcDir)
# irisPngFiles = glob.glob(os.path.join(srcDir, '*/*/iris_*_right_*.png'))
#
# for file in irisPngFiles:
#     print(file)
#     shutil.copy(file, dstDir)






import openpyxl
srcDir = "D:/2. data/total_iris/dataVoucher_right/"
dstDir = "D:/2. data/total_iris/dataVoucher_right_id/"

files = os.listdir(srcDir)
irisPngFiles = glob.glob(os.path.join(srcDir, 'iris_*_right_*.png'))

xles_name = 'id.xlsx'
id = pd.read_excel(xles_name)
list = id['id'].tolist()
print(len(list))

for file in irisPngFiles:
    filename = file.split("\\")[-1][:-4]
    print(filename)

    if filename in list:
        print('리스트에 값이 있습니다.')
        shutil.copy(file, dstDir)
    else:
        print('없습니다.')


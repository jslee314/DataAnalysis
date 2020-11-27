import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dtype = {'brain_defect': float, 'brain_lacuna': float, 'brain_normal': float, 'brain_spoke': float, 'brain_spot': float,
#         'kidney_defect': float, 'kidney_lacuna': float, 'kidney_normal': float, 'kidney_spoke': float, 'kidney_spot': float,
#         'liver_defect': float, 'liver_lacuna': float, 'liver_normal': float, 'liver_spoke': float, 'liver_spot': float,
#         'lung_defect': float, 'lung_lacuna': float, 'lung_normal': float, 'lung_spoke': float, 'lung_spot': float,
#          'absorption': float, 'cholesterol': float, 'stress': float,
#          'height': float,	'weight': float, 'bmi': float, 'systolic_bp': float, 'diastolic_bp': float,
#          'hemoglobin': float, 'Glucose': float, 'total_chol': float, 'high_chol': float, 'triglyceride': float, 'low_chol':float,
#          'creatinine': float, 'egfr': float, 'AST': float, 'ALT': float, 'GTP': float,
#          'age':float, 'survey_stress': float, 'survey_depression':float, 'survey_diabetes':float, 'survey_dementia':float, 'survey_liver':float}
#
# totalData_df = pd.read_excel('totalData.xlsx', sheet_name='total', dtype=dtype, index_col='iris_image')
# totalData_df_corr = totalData_df.corr()
#
# byidData_df = pd.read_excel('totalData.xlsx', sheet_name='byid', dtype=dtype, index_col='iris_image')
# byidData_df_corr = byidData_df.corr()
#
# byid_editData_df = pd.read_excel('totalData.xlsx', sheet_name='byid_edit', dtype=dtype, index_col='iris_image')
# byid_editData_df_corr = byid_editData_df.corr()


# with pd.ExcelWriter('totalData_corr.xlsx') as w:
#     totalData_df_corr.to_excel(w, sheet_name='total_correlation')
#     byidData_df_corr.to_excel(w, sheet_name='byid_correlation')
#     byid_editData_df_corr.to_excel(w, sheet_name='byid_edit_correlation')

# sns.heatmap(byidData_df_corr, annot=True)
# plt.show()





xles_name = 'totalData_id.xlsx'
# brain = pd.read_excel(xles_name, sheet_name='brain')
# sns.pairplot(brain)
# plt.savefig("brain11.png")


kidney = pd.read_excel(xles_name, sheet_name='kidney')
sns.pairplot(kidney, corner=True)
plt.savefig("kidney.png")
#
# liver = pd.read_excel(xles_name, sheet_name='liver')
# sns.pairplot(liver, corner=True)
# plt.savefig("liver.png")
#
# stress = pd.read_excel(xles_name, sheet_name='stress')
# sns.pairplot(stress, corner=True)
# plt.savefig("stress.png")




'''iris_259_right_0902_28d188dc-fff8-4d6d-9024-a14c979bc617
iris_255_right_0902_62deaf95-b58d-41c2-8780-d603072af510
iris_235_right_0908_479dbeaa-4fc5-4711-8c67-5ea2f0c36e3a
iris_234_right_0908_966ca7b2-644d-40b6-8d97-cb4a591b65df
iris_234_right_0908_acf3fc1f-c98f-4454-8436-31b68a7b1676
iris_234_right_0908_f63d09c8-239a-4b30-8fd5-cdf63422691a
iris_232_right_0908_1f0c5101-d4d2-40ec-85b8-187e893c2a31
iris_231_right_0908_6b076e5a-f6c3-418a-92c3-025637d780e6
iris_229_right_0908_6c69743d-4c28-4f15-a463-e5d20d8119da
iris_228_right_0908_23553b08-2658-432b-ac58-23decbd318f6
iris_227_right_0908_38f87559-aa18-4d16-a33e-aa93778fe46a
iris_224_right_0908_1f69486b-ad3e-4d51-a421-5bdb9335db37
iris_223_right_0908_912cf34c-38bd-498d-a50f-84302505be6c
iris_223_right_0908_bcccf29d-b5d6-4d84-a8d3-8fe0199312d9
iris_222_right_0908_9880fcc2-8b05-481d-bf4d-6892e8cf8af4
iris_221_right_0908_b7d70ac7-d07a-4281-bdda-dac2fb3f0f5e
iris_219_right_0909_ab690883-2b25-41ec-97ec-5d8a5e2aee1a
iris_216_right_0909_6440f955-468d-4cc4-8402-a8935044de68
iris_215_right_0909_5a1c07db-5ce7-4ff5-a5e2-1a696adab484
iris_214_right_0909_79274844-b453-4d17-a387-e7a0f4a2c17b
iris_212_right_0909_f1cc2f5b-88d1-4c60-9381-7434ac79468e
iris_210_right_0909_0f5ecddb-4afe-4090-a5c3-046e8431f27b
iris_209_right_0909_b55edd5f-757f-4bb9-9eac-f174202a507a
iris_207_right_0910_9c1488ca-7c04-4f0b-9b25-6c9ced56ddc2
iris_206_right_0910_c5575136-41e3-4388-9798-45884557ba2f
iris_205_right_0910_5394e0da-5231-4336-b5b6-e06e870367b4
iris_204_right_0910_c1788716-4e03-43a8-9253-7c5cedc9cdc9
iris_202_right_0910_80fa98ef-2812-4566-8955-36ad71646df6
iris_197_right_0911_728772ac-39fa-4668-9c51-bb3b2ad12334
iris_196_right_0911_46961c10-2099-4e77-a278-58e20c9198e7
iris_195_right_0911_82742eb3-ddfb-45e3-ac81-1de481c92522
iris_194_right_0911_70753ed8-3618-4764-b188-0f76f09cfa3a
iris_193_right_0911_b3f4155d-a56b-4645-89a5-fd0f47fdf803
iris_190_right_0911_079050a2-70a5-44cf-aa14-e8b662174997
iris_188_right_0911_825d3f00-f1b5-44f1-97b3-e77cee58b1d3
iris_187_right_0911_f12249c1-b115-4214-aa60-1b94621e861e
iris_184_right_0914_851fa1bb-7375-47ff-adde-a79e2148eab9
iris_183_right_0914_c5ce8086-d583-4020-8199-148c2c316e1e
iris_181_right_0914_8f9b1179-fe33-481b-8c0c-1ce0dcbdc658
iris_178_right_0914_73ee81cb-2ae4-482c-a419-2a32aa530f2f
iris_178_right_0914_4c175dda-0177-4855-ab40-f6e1951b7a6c
iris_177_right_0914_a46d1980-3880-404e-8287-1be6862d1678
iris_164_right_0916_6d562f94-93b0-4ddf-8148-832df414465a
iris_157_right_0917_a6d3e493-8064-46e9-8772-257215947355
iris_156_right_0917_4a081da4-667f-474b-baa1-43d88d2c02a2
iris_155_right_0917_0c9de659-926f-49e8-839c-4ac347b1140f
iris_151_right_0917_b5c6b2b9-d939-4864-ad36-cc2e618cf83b
iris_148_right_0917_25d3df62-95f7-4361-938f-98ce242272c6
iris_144_right_0918_19ffcbd1-ade4-4065-99c6-ee683f28e36e
iris_143_right_0918_1308e446-bab2-4a24-8e3f-647dcda91dd5
iris_142_right_0918_86332c21-0ed0-476d-bee4-56efb76572e2
iris_140_right_0918_d71bfe70-d9ae-4b65-8dc8-05aea94581e3
iris_139_right_0918_50eb6ce9-13d3-4328-9f18-7b81a88b796d
iris_136_right_0918_05e719bf-6890-4610-9656-85f2a00b0eef
iris_134_right_0918_a4d70d61-cde9-41a8-9931-a61e71dc48b4
iris_133_right_0918_55fc260f-22c1-4987-a6d7-7a36806454cc
iris_132_right_0918_3e9ffe72-2faf-4786-a65c-e94208b7ce85
iris_130_right_0918_38b0b923-48cb-4224-9e8e-e1ba16b36f30
iris_129_right_0918_f9588f13-fa2f-4ad7-bb29-b1c087920ff9
iris_128_right_0918_5c4bd1eb-5f22-4fdb-856a-04c8c2e86001
iris_128_right_0918_91ef7f4b-3cc9-4c56-81eb-6ab6b8ed81de
iris_126_right_0921_291f1a63-884b-492f-9ab8-ff1ec98a5cad
iris_125_right_0921_ddd62c82-4ea3-426e-a7f4-e72330f85dd9
iris_124_right_0921_be4e1871-8da0-4ac8-8e2f-b267dafd476b
iris_123_right_0921_0e06e553-8ac2-4ade-b2da-ef436373fd6d
iris_120_right_0921_5199cf72-5b24-4033-b47b-12958b850e5d
iris_119_right_0921_fc7276a5-5a47-41d5-90a1-650040a3ea69
iris_118_right_0922_7a2f036a-2cde-4a53-8ce7-76ce1c430b2c
iris_117_right_0922_efd33ade-333f-4dd0-9c32-6c810baeee46
iris_116_right_0922_9c22101c-5d45-4a68-b134-e5f884417cc1
iris_114_right_0922_4d4fa113-b35c-4057-81a2-4ac39761b7ed
iris_111_right_0925_703275ab-bb85-4d90-bfae-c30b66559029
iris_110_right_0925_1937e700-ff3a-4e5a-a49c-c04577d34dfa
iris_107_right_0925_e1d258c8-d35a-4db5-801b-f8f736fa3b92
iris_107_right_0925_c3dcba19-f268-4462-8100-61c69d74e5bb
iris_103_right_0925_2af684a0-7b67-4895-9937-5436d3c212d3
iris_101_right_0925_7a1c86b1-012f-4ce4-b01d-830cce3f717b
iris_101_right_0925_03a44164-54af-4a8b-9d47-48d6bb5823c8
'''
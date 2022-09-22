# ResNet50의 이미지분류로 X-ray이미지 분류하기

Chest_xray_Corona_Metadata.csv의 데이터 예

X_ray_image_name|Label|Dataset_type|Label_2_Virus_category|Label_1_Virus_category
---|---|---|---|---|
IM-01~~.jpeg|Normal|TRAIN|Null|bacteria
IM-03~~.jpeg|Normal|Test|Null|virus
IM-00~~.jpeg|Normal|TRAIN|Null|Covid-19

X-ray 이미지 파일

* Coronahack-Chest-XRay-Dataset
  * test
  * train


분류를 위해 Metadata Label에 있는 Train과 Test의 데이터를 나눠줬습니다.

'''python
train_data = meta[meta['Dataset_type']=='TRAIN'] \n
test_data = meta[meta['Dataset_type']=='TEST']
'''

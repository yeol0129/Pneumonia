X-ray Images classification
=============
***
#### [code](https://github.com/yeol0129/xray_ResNet50_Pneumonia/blob/main/pneumonia_resnet50.ipynb), [Data(kaggle)](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset
<img width="341" alt="image" src="https://user-images.githubusercontent.com/111839344/217168467-0eabe747-f830-4050-ab68-3263ff7c10a4.png">
)
## Data 
>  ### Chest_xray_Corona_Metadata.csv sample
>  X_ray_image_name|Label|Dataset_type|Label_2_Virus_category|Label_1_Virus_category
> ---|---|---|---|---|
> IM-01~~.jpeg|Normal|TRAIN|Null|bacteria
> IM-03~~.jpeg|Normal|Test|Null|virus
> IM-00~~.jpeg|Pnemonia|TRAIN|Null|Covid-19
>  ### Prepare data
>  null to 0
>  ```python
>  meta.fillna('0', inplace = True)
>  meta.isnull().sum()
>  ```
>  ### Separate train data and test data
>  ```python
>  train_data = meta[meta['Dataset_type']=='TRAIN']
>  test_data = meta[meta['Dataset_type']=='TEST']
>  ```
>  ### Separate Pneumonia and Normal data
>  ```python
>  Pneumonia  = train_df[train_df['Label']=='Pnemonia']  
>  Normal = train_df[train_df['Label']=='Normal']
>  ```
> ### image sample
>  ```python
>  Nor_sample = Image.open(os.path.join(train_img, Normal['X_ray_image_name'].iloc[1]))
>  Pne_sample = Image.open(os.path.join(train_img, Pneumonia['X_ray_image_name'].iloc[1]))
>  ```
> Pneumonia|Normal
> ---|---|
> <img src="https://user-images.githubusercontent.com/111839344/191780342-da945fb9-a1e2-4c58-b157-a8c2ce632917.png" width="200" height="200">|<img src="https://user-images.githubusercontent.com/111839344/191781073-e5f198af-63ae-4ddb-a794-01026e13f7e4.png" width="200" height="200">
> ### Train Data Value
> ```python
> train_df['Label'].value_counts()
> ```
> output : 
> Pnemonia    3944
> Normal      1342

## Data train
> ### Split train and validation data sets (train:val = 8:2)
> ```python
> train_df, valid_df = train_test_split(train_df, train_size=0.8, random_state=0)
> ```
> ### Image data generator
> ```python
> train_datagen = ImageDataGenerator(rescale = 1/255,rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.2, 
>                                  shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, vertical_flip =True)
> test_datagen = ImageDataGenerator(rescale = 1/255)
> ```
> ```python
> train_gen = train_datagen.flow_from_dataframe(dataframe = train_df, directory=train_img, x_col='X_ray_image_name', 
>                                             y_col='Label', target_size=(224,224), batch_size=64, 
>                                              class_mode='binary')
> valid_gen = test_datagen.flow_from_dataframe(dataframe = valid_df, directory=train_img, x_col='X_ray_image_name',
>                                            y_col='Label', target_size=(224,224), batch_size=64, 
>                                           class_mode='binary')
> test_gen = test_datagen.flow_from_dataframe(dataframe = test_df, directory=test_img, x_col='X_ray_image_name', 
>                                           y_col='Label', target_size=(224,224), batch_size=64,
>                                            class_mode='binary')
> ```
> ### ResNet50 model
> ```python
> Resnet_model = tf.keras.applications.ResNet50V2(weights='imagenet', input_shape = (224,224,3),
>                                                     include_top=False)
> ```
> 
> 
>
> 
> ### training ResNet50 model
> ```python
> history = model.fit(train_gen, 
>                    validation_data=valid_gen, epochs=100, 
>                    callbacks=[early_stopping_callback,checkpointer])
> ```
> output :
> ```
> Epoch 9/100
> 67/67 [==============================] - ETA: 0s - loss: 0.0985 - accuracy: 0.9633
> Epoch 9: val_loss did not improve from 0.10675
> 67/67 [==============================] - 54s 805ms/step - loss: 0.0985 - accuracy: 0.9633 - val_loss: 0.1365 - val_accuracy: 0.9461
> ```
> <img src="https://user-images.githubusercontent.com/111839344/191777801-97fd13aa-7f06-47ec-a510-f38a3b107e27.png" width="400" height="400"/>


X-ray Images classification
=============
***
#### [code](https://github.com/yeol0129/xray_ResNet50_Pneumonia/blob/main/pneumonia_resnet50.ipynb)
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
> ### X-ray images

> * Coronahack-Chest-XRay-Dataset
>   * test
>   * train
> ### image sample
> Pneumonia|Normal
> ---|---|
> <img src="https://user-images.githubusercontent.com/111839344/191780342-da945fb9-a1e2-4c58-b157-a8c2ce632917.png" width="200" height="200">|<img src="https://user-images.githubusercontent.com/111839344/191781073-e5f198af-63ae-4ddb-a794-01026e13f7e4.png" width="200" height="200">

> ### Pneumonia Data
> ```python
> train_df['Label'].value_counts()
> ```
> output : 
> Pnemonia    3944
> Normal      1342

## Data train
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

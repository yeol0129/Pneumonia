ResNet50의 이미지분류로 X-ray이미지 분류하기
=============
#### 자세한 코드는 [여기](https://github.com/yeol0129/xray_ResNet50_Pneumonia/blob/main/pneumonia_resnet50.ipynb)
## Data 
>  ### Chest_xray_Corona_Metadata.csv의 데이터 예
>  X_ray_image_name|Label|Dataset_type|Label_2_Virus_category|Label_1_Virus_category
> ---|---|---|---|---|
> IM-01~~.jpeg|Normal|TRAIN|Null|bacteria
> IM-03~~.jpeg|Normal|Test|Null|virus
> IM-00~~.jpeg|Pnemonia|TRAIN|Null|Covid-19

> ### X-ray 이미지 파일

> * Coronahack-Chest-XRay-Dataset
>   * test
>   * train


> ### 분류를 위해 Metadata Label에 있는 Train과 Test의 데이터를 나눈 후 저장합니다.

> ```python
> train_data = meta[meta['Dataset_type']=='TRAIN']
> test_data = meta[meta['Dataset_type']=='TEST']
> train_data.to_csv('train_data_corona.csv')
> test_data.to_csv('test_data_corona.csv')
> ```

> ### train과 test 각각 불러옵니다.

> ```python
> train_df=pd.read_csv('train_data_corona.csv')
> test_df=pd.read_csv('test_data_corona.csv')
> train_img='./Coronahack-Chest-XRay-Dataset/train'
> test_img='./Coronahack-Chest-XRay-Dataset/test'
> ```

> ### 폐렴데이터 확인
> ```python
> train_df['Label'].value_counts()
> ```
> output : 
> Pnemonia    3944
> Normal      1342

> ### 폐렴데이터와 정상데이터를 분리합니다.
> ```python
> Pneumonia  = train_df[train_df['Label']=='Pnemonia']  
> Normal = train_df[train_df['Label']=='Normal']
> ```

## 데이터 전처리
 > ### Train의 데이터를 Train set과 Validation set 8:2의 비율로 분할합니다.
 > ```python
 > train_df, valid_df = train_test_split(train_df, train_size=0.8, random_state=0)
 > ```

> ### ImageDataGenerator를 통한 이미지 증식과 정규화
> ```python
> train_datagen = ImageDataGenerator(rescale = 1/255,rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.2, 
>                                   shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, vertical_flip =True)
> test_datagen = ImageDataGenerator(rescale = 1/255)
> ```

> ### flow_from_dataframe 통해 train,validation,test에 사용할 이미지데이터 불러옵니다.
> ```python
> train_gen = train_datagen.flow_from_dataframe(dataframe = train_df, directory=train_img, x_col='X_ray_image_name', 
>                                              y_col='Label', target_size=(224,224), batch_size=64, 
>                                               class_mode='binary')
> valid_gen = test_datagen.flow_from_dataframe(dataframe = valid_df, directory=train_img, x_col='X_ray_image_name',
>                                             y_col='Label', target_size=(224,224), batch_size=64, 
>                                            class_mode='binary')
> test_gen = test_datagen.flow_from_dataframe(dataframe = test_df, directory=test_img, x_col='X_ray_image_name', 
>                                           y_col='Label', target_size=(224,224), batch_size=64,
>                                             class_mode='binary')
> ```
> output :
> ```
> Found 4228 validated image filenames belonging to 2 classes.
> Found 1058 validated image filenames belonging to 2 classes.
> Found 624 validated image filenames belonging to 2 classes.
> ```

## 데이터 훈련
> ### ResNet50모델을 사용하기위해 모델을 불러옵니다.
> ```python
> Resnet_model = tf.keras.applications.ResNet50V2(weights='imagenet', input_shape = (224,224,3),
>                                                     include_top=False)
> for layer in Resnet_model.layers:
>    layer.trainable = False
> ```
> ### ResNet50을 추가하여 모델 학습합니다.
> ```python
> model = tf.keras.Sequential([
>    Resnet_model, 
>    tf.keras.layers.GlobalAveragePooling2D(), 
>    tf.keras.layers.Dense(128, activation='relu'),
>    tf.keras.layers.BatchNormalization(), 
>    tf.keras.layers.Dropout(0.2), 
>    tf.keras.layers.Dense(1, activation='sigmoid')
> ])
>```
> ### 모델 컴파일
> ```python
> model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001),
>              loss = 'binary_crossentropy',
>              metrics=['accuracy'])
>              MODEL_DIR='./pne_model/'
> ```
>
> ### 과적합방지하기위해 학습자동중단 설정
> ```python
> if not os.path.exists(MODEL_DIR):
>    os.mkdir(MODEL_DIR)
> modelpath="./pne_model/{epoch:02d}-{val_loss:.4f}.hdf5"
> checkpointer=ModelCheckpoint(filepath=modelpath,monitor='val_loss',verbose=1,save_best_only=True)
> early_stopping_callback=EarlyStopping(monitor='val_loss',patience=3)
> ```
> 
> ### 모델 학습
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

> ### 모델의 정확도와 손실 시각화
> ```python
> plt.plot(history.history['accuracy'],'y')
> plt.plot(history.history['val_accuracy'],'r')
> plt.plot(history.history['loss'],'g')
> plt.plot(history.history['val_loss'],'b')
> plt.legend(['train_acc', 'test_acc','train_loss', 'test_loss'], loc='center left')
> plt.show()
> ```
> output :
> ![graph](https://user-images.githubusercontent.com/111839344/191777801-97fd13aa-7f06-47ec-a510-f38a3b107e27.png)

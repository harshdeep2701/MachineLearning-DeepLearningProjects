
from keras.layers import Convolution2D, Dropout, BatchNormalization
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint

# 1. Initialize CNN
classifier=Sequential()
 
# 2. Convolution 
classifier.add(Convolution2D(input_shape=(180,180,3),filters=32,kernel_size=(3,3),strides=(1,1),activation='relu'))
# 3. Max Pooling 
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(BatchNormalization())
# 4. Convolution 
classifier.add(Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu'))
# 5. Max Pooling 
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(BatchNormalization())
# 4. Convolution 
classifier.add(Convolution2D(filters=128,kernel_size=(3,3),strides=(1,1),activation='relu'))
# 5. Max Pooling 
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(BatchNormalization())
# 4. Convolution 
classifier.add(Convolution2D(filters=256,kernel_size=(3,3),strides=(1,1),activation='relu'))
# 5. Max Pooling 
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(BatchNormalization())
# 6. Flattening 
classifier.add(Flatten())
 
# 7. Full Connection 
classifier.add(Dense(units=128,activation='relu',kernel_initializer='uniform'))
classifier.add(Dropout(0.6))
classifier.add(Dense(units=96,activation='relu',kernel_initializer='uniform'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=64,activation='relu',kernel_initializer='uniform'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))

# 8. Compliling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# 9. Fitting CNN with Images 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1./255)
 
training_set = train_datagen.flow_from_directory(
        '../input/dogs-cats-images/dataset/training_set',
        target_size=(180,180),
        batch_size=128,
        class_mode='binary')
 
test_set = test_datagen.flow_from_directory(
        '../input/dogs-cats-images/dataset/test_set',
        target_size=(180,180),
        batch_size=128,
        class_mode='binary',
        shuffle=False)
filepath="bestcatvsdogmodel.hdf5"
save_best_model=ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
ReduceLR=ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=5, verbose=1,cooldown=1, min_delta=0.0020, min_lr=0.)

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/128,
        epochs=100,
        validation_data=test_set,
        validation_steps=2000/128,
        callbacks=[ReduceLR,save_best_model])
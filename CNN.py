# CNN

# Part 1 - Building our CNN

# importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initialising CNN
classifier = Sequential()

# step 1 - convolution
classifier.add( Convolution2D(32 , 3 , padding = 'same' , input_shape = (64,64,3) , activation = 'relu' ) )

# step 2 - max pooling
classifier.add( MaxPooling2D( pool_size = (2,2)))

# Adding a layer
classifier.add( Convolution2D(32 , 3 , padding = 'same', activation = 'relu' ) )
classifier.add( MaxPooling2D( pool_size = (2,2)))

# step 3 - flattening
classifier.add(Flatten())

# step 4 - full connection
classifier.add(Dense(output_dim = 128 , activation = 'relu'))
classifier.add(Dense(output_dim = 1 , activation = 'sigmoid' ))

# compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

# Part 2 - Fitting CNN 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
                                   
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')
                                                    
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')
        
classifier.fit( training_set,
            steps_per_epoch = 8000,
            epochs = 25,
            validation_data=test_set,
            validation_steps = 2000)
       
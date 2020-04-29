import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
import pickle
path = 'Train'
testRatio=0.2
valRatio =0.2
imageDimensions = (32,32,3)

images=[]
classNo =[]
myList = os.listdir(path)
print("Total number of classes", len(myList))
noOfCLasses = len(myList)
print("Importing Classes")
for x in range (1, noOfCLasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg, (32,32))
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")
print(" ")


images = np.array(images)
classNo = np.array(classNo)

print(images.shape)
print(classNo.shape)

######Splitting Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation =train_test_split(X_train,y_train, test_size=valRatio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numOfSamples=[]
for x in range(0, noOfCLasses):
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0, noOfCLasses), numOfSamples)
plt.title("Number of images")
plt.xlabel=("ID")
plt.ylabel("no")
plt.show()
def preProcessing(img):
    img=cv2.cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1],X_validation.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train, noOfCLasses)
y_test = to_categorical(y_test, noOfCLasses)
y_validation = to_categorical(y_validation, noOfCLasses)





model = Sequential()
model.add((Conv2D(60, (5,5), input_shape=(32,32, 1), activation= 'relu')))
model.add((Conv2D(60, (5,5), activation='relu')))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add((Conv2D(60//2, (3,3), activation='relu')))
model.add((Conv2D(60//2, (3,3), activation= 'relu')))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(43, activation='softmax'))
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])



print(model.summary())
epochVal = 10
stepsPerEpoch = 2000
history = model.fit_generator(dataGen.flow(X_train,y_train,
                                 batch_size=50), steps_per_epoch=stepsPerEpoch, epochs=epochVal,
                              validation_data=(X_validation, y_validation),
                                shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel ='epoch'
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel= 'epoch'
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])


pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

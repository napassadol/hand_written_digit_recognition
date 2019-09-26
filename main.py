from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
import cv2, time


# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
 
# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
 
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()
	# fit model
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
	# save model
	model.save('model.h5', overwrite=True)

def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
 
def process(filename):
	img = load_image(filename)
	model = load_model('model.h5')
	digit = model.predict_classes(img)
	print(digit)
	return digit[0]

# if __name__ == "__main__":
def main(image_name):
	input_image = image_name
	image = cv2.imread(input_image)
	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
	ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	digit = None

	if len(contours) > 0:
		c = max(contours, key = cv2.contourArea)
		x,y,w,h = cv2.boundingRect(c)
		cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2)
		sub_image = thresh[y - 20:y+h + 20, x - 20:x+w + 20]
		cv2.imwrite('input.' + input_image.split('.')[-1], sub_image)
		digit = process('input.' + input_image.split('.')[-1])
		cv2.putText(image, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

	cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
	# cv2.imshow('contours', image)
	# cv2.imwrite('output.' + input_image.split('.')[-1], image)
	# if cv2.waitKey(0) & 0xff == 27:  
	# 	cv2.destroyAllWindows()
	return digit
	



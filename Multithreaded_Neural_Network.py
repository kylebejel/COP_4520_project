import tensorflow
import threading
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD


def evaluate_model(dataX, dataY, model, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

def define_model(layerNum, lossType):
	model = Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Flatten())
	model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(layers.Dense(layerNum, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss=lossType, metrics=['accuracy'])
	return model

def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

__name__ == '__main__':
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    
    sizes = [0 for i in range(10)]
    for i in range(len(yTrain)):
        sizes[yTrain[i]] += 1
    yTrainSep = [10][0 for i in range(len(yTrain))]
    for i in range(len(yTrain)):
        if(yTrain[i] == 0):
            yTrainSep[0][i] = 1
            yTrainSep[1][i],yTrainSep[2][i],yTrainSep[3][i],yTrainSep[4][i],yTrainSep[5][i],yTrainSep[6][i],yTrainSep[7][i],yTrainSep[8][i],yTrainSep[9][i] = 0
        if(yTrain[i] == 1):
            yTrainSep[1][i] = 1
            yTrainSep[0][i],yTrainSep[2][i],yTrainSep[3][i],yTrainSep[4][i],yTrainSep[5][i],yTrainSep[6][i],yTrainSep[7][i],yTrainSep[8][i],yTrainSep[9][i] = 0
        if(yTrain[i] == 2):
            yTrainSep[2][i] = 1
            yTrainSep[1][i],yTrainSep[0][i],yTrainSep[3][i],yTrainSep[4][i],yTrainSep[5][i],yTrainSep[6][i],yTrainSep[7][i],yTrainSep[8][i],yTrainSep[9][i] = 0
        if(yTrain[i] == 3):
            yTrainSep[3][i] = 1
            yTrainSep[1][i],yTrainSep[2][i],yTrainSep[0][i],yTrainSep[4][i],yTrainSep[5][i],yTrainSep[6][i],yTrainSep[7][i],yTrainSep[8][i],yTrainSep[9][i] = 0
        if(yTrain[i] == 4):
            yTrainSep[4][i] = 1
            yTrainSep[1][i],yTrainSep[2][i],yTrainSep[3][i],yTrainSep[0][i],yTrainSep[5][i],yTrainSep[6][i],yTrainSep[7][i],yTrainSep[8][i],yTrainSep[9][i] = 0
        if(yTrain[i] == 5):
            yTrainSep[5][i] = 1
            yTrainSep[1][i],yTrainSep[2][i],yTrainSep[3][i],yTrainSep[4][i],yTrainSep[0][i],yTrainSep[6][i],yTrainSep[7][i],yTrainSep[8][i],yTrainSep[9][i] = 0
        if(yTrain[i] == 6):
            yTrainSep[6][i] = 1
            yTrainSep[1][i],yTrainSep[2][i],yTrainSep[3][i],yTrainSep[4][i],yTrainSep[5][i],yTrainSep[0][i],yTrainSep[7][i],yTrainSep[8][i],yTrainSep[9][i] = 0
        if(yTrain[i] == 7):
            yTrainSep[7][i] = 1
            yTrainSep[1][i],yTrainSep[2][i],yTrainSep[3][i],yTrainSep[4][i],yTrainSep[5][i],yTrainSep[6][i],yTrainSep[0][i],yTrainSep[8][i],yTrainSep[9][i] = 0
        if(yTrain[i] == 8):
            yTrainSep[8][i] = 1
            yTrainSep[1][i],yTrainSep[2][i],yTrainSep[3][i],yTrainSep[4][i],yTrainSep[5][i],yTrainSep[6][i],yTrainSep[7][i],yTrainSep[0][i],yTrainSep[9][i] = 0
        if(yTrain[i] == 9):
            yTrainSep[9][i] = 1
            yTrainSep[1][i],yTrainSep[2][i],yTrainSep[3][i],yTrainSep[4][i],yTrainSep[5][i],yTrainSep[6][i],yTrainSep[7][i],yTrainSep[8][i],yTrainSep[0][i] = 0
    
    model = define_model(10,'categorical_crossentropy')
    modelSep = []
    for i in range(10):
        modelSep.append(define_model(2, 'binary_crossentropy'))

    xTrain = xTrain.reshape((xTrain.shape[0], 28, 28, 1))
    xTest = xTest.reshape((xTest.shape[0], 28, 28, 1))
    xTrain, xTest = prep_pixels(xTrain, xTest)
    t0 = threading.Thread(target=evaluate_model, args=(xTrain, yTrainSep[0], modelSep[0]))
    t1 = threading.Thread(target=evaluate_model, args=(xTrain, yTrainSep[1], modelSep[1]))
    t2 = threading.Thread(target=evaluate_model, args=(xTrain, yTrainSep[2], modelSep[2]))
    t3 = threading.Thread(target=evaluate_model, args=(xTrain, yTrainSep[3], modelSep[3]))
    t4 = threading.Thread(target=evaluate_model, args=(xTrain, yTrainSep[4], modelSep[4]))
    t5 = threading.Thread(target=evaluate_model, args=(xTrain, yTrainSep[5], modelSep[5]))
    t6 = threading.Thread(target=evaluate_model, args=(xTrain, yTrainSep[6], modelSep[6]))
    t7 = threading.Thread(target=evaluate_model, args=(xTrain, yTrainSep[7], modelSep[7]))
    t8 = threading.Thread(target=evaluate_model, args=(xTrain, yTrainSep[8], modelSep[8]))
    t9 = threading.Thread(target=evaluate_model, args=(xTrain, yTrainSep[9], modelSep[9]))
    t10 = threading.Thread(target=evaluate_model, args=(xTrain, yTrain, model))

    t0.start
    t1.start
    t2.start
    t3.start
    t4.start
    t5.start
    t6.start
    t7.start
    t8.start
    t9.start
    t10.start

    t0.join
    t1.join
    t2.join
    t3.join
    t4.join
    t5.join
    t6.join
    t7.join
    t8.join
    t9.join
    t10.join

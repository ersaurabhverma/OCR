import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist=tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test)=mnist.load_data()
#print(x_train[0])
#plt.imshow(x_train[0],cmap=plt.cm.binary)
#plt.show()
#print(y_train[0])
#print(x_train[0].shape)
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
model=tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10)
loss,accuracy=model.evaluate(x_test,y_test)
print('Loss='+str(loss*100)+'%')
print('Accuracy='+str(accuracy*100)+'%')
model.save('mnist.model')
print('Model Saved!!!')



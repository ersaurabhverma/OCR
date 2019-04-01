import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

mnist=tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test)=mnist.load_data()

saved_model=tf.keras.models.load_model('mnist.model')
predictions=saved_model.predict(x_test)
print('Predicted outcome=',np.argmax(predictions[0]))
plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()

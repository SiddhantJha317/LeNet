
# LeNet5

I try to implement the LeNet5 algorithm , the first CNN using python and tensorflow without the OOP version.

Paper Link:https://arxiv.org/abs/1609.04112



Importing requisite libraries
```
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from keras import layers
from keras import models
from keras import losses
```
Importing and structuring Data, for training and testing
```
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
x_train.shape
```
Padding the photograph file to fit the model 
```
x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255
x_train.shap
```
Increasing dimensions to meet requisite requirements
```
x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)
x_train.shape
```
Splitting Data into Validation and training sets

```
x_val = x_train[-2000:,:,:,:]
y_val = y_train[-2000:]
x_train = x_train[:-2000,:,:,:]
y_train = y_train[:-2000]
```
Constructing the model
```
model = models.Sequential()
model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=x_train.shape[1:]))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(16, 5, activation='tanh'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(120, 5, activation='tanh'))
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
-----------------------------------------------------------------
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 6)         156       
                                                                 
 average_pooling2d (AverageP  (None, 14, 14, 6)        0         
 ooling2D)                                                       
                                                                 
 activation (Activation)     (None, 14, 14, 6)         0         
                                                                 
 conv2d_1 (Conv2D)           (None, 10, 10, 16)        2416      
                                                                 
 average_pooling2d_1 (Averag  (None, 5, 5, 16)         0         
 ePooling2D)                                                     
                                                                 
 activation_1 (Activation)   (None, 5, 5, 16)          0         
                                                                 
 conv2d_2 (Conv2D)           (None, 1, 1, 120)         48120     
                                                                 
 flatten (Flatten)           (None, 120)               0         
                                                                 
 dense (Dense)               (None, 84)                10164     
                                                                 
 dense_1 (Dense)             (None, 10)                850       
                                                                 
=================================================================
Total params: 61,706
Trainable params: 61,706
Non-trainable params: 0
_________________________________________________________________
```
Compiling the Model

```
model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
```

Training the model
```
history = model.fit(x_train, y_train, batch_size=64, epochs=40, validation_data=(x_val, y_val))
-----------------------------------------------------------------------------------------------------------------------------------------
Epoch 1/40
907/907 [==============================] - 34s 36ms/step - loss: 1.2429 - accuracy: 0.5571 - val_loss: 0.3480 - val_accuracy: 0.8920
Epoch 2/40
907/907 [==============================] - 32s 36ms/step - loss: 0.3673 - accuracy: 0.8838 - val_loss: 0.2291 - val_accuracy: 0.9370
Epoch 3/40
907/907 [==============================] - 32s 35ms/step - loss: 0.2764 - accuracy: 0.9111 - val_loss: 0.1587 - val_accuracy: 0.9555
Epoch 4/40
907/907 [==============================] - 31s 35ms/step - loss: 0.2346 - accuracy: 0.9256 - val_loss: 0.1501 - val_accuracy: 0.9530
Epoch 5/40
907/907 [==============================] - 32s 35ms/step - loss: 0.2080 - accuracy: 0.9346 - val_loss: 0.1227 - val_accuracy: 0.9660
Epoch 6/40
907/907 [==============================] - 31s 35ms/step - loss: 0.1810 - accuracy: 0.9421 - val_loss: 0.1022 - val_accuracy: 0.9705
Epoch 7/40
907/907 [==============================] - 33s 36ms/step - loss: 0.1664 - accuracy: 0.9479 - val_loss: 0.1443 - val_accuracy: 0.9585
Epoch 8/40
907/907 [==============================] - 32s 36ms/step - loss: 0.1526 - accuracy: 0.9513 - val_loss: 0.1100 - val_accuracy: 0.9695
Epoch 9/40
907/907 [==============================] - 33s 36ms/step - loss: 0.1386 - accuracy: 0.9567 - val_loss: 0.0996 - val_accuracy: 0.9745
Epoch 10/40
907/907 [==============================] - 32s 35ms/step - loss: 0.1295 - accuracy: 0.9596 - val_loss: 0.1030 - val_accuracy: 0.9715
Epoch 11/40
907/907 [==============================] - 32s 35ms/step - loss: 0.1285 - accuracy: 0.9598 - val_loss: 0.0840 - val_accuracy: 0.9800
Epoch 12/40
907/907 [==============================] - 33s 37ms/step - loss: 0.1172 - accuracy: 0.9625 - val_loss: 0.0809 - val_accuracy: 0.9795
Epoch 13/40
907/907 [==============================] - 32s 35ms/step - loss: 0.1141 - accuracy: 0.9630 - val_loss: 0.1030 - val_accuracy: 0.9750
Epoch 14/40
907/907 [==============================] - 32s 35ms/step - loss: 0.1058 - accuracy: 0.9662 - val_loss: 0.1384 - val_accuracy: 0.9570
Epoch 15/40
907/907 [==============================] - 31s 34ms/step - loss: 0.1018 - accuracy: 0.9673 - val_loss: 0.0761 - val_accuracy: 0.9810
Epoch 16/40
907/907 [==============================] - 31s 34ms/step - loss: 0.0969 - accuracy: 0.9693 - val_loss: 0.0709 - val_accuracy: 0.9820
Epoch 17/40
907/907 [==============================] - 31s 34ms/step - loss: 0.0912 - accuracy: 0.9706 - val_loss: 0.0821 - val_accuracy: 0.9780
Epoch 18/40
907/907 [==============================] - 32s 36ms/step - loss: 0.0867 - accuracy: 0.9726 - val_loss: 0.1114 - val_accuracy: 0.9700
Epoch 19/40
907/907 [==============================] - 31s 35ms/step - loss: 0.0850 - accuracy: 0.9723 - val_loss: 0.0898 - val_accuracy: 0.9735
Epoch 20/40
907/907 [==============================] - 32s 35ms/step - loss: 0.0834 - accuracy: 0.9734 - val_loss: 0.0778 - val_accuracy: 0.9800
Epoch 21/40
907/907 [==============================] - 32s 35ms/step - loss: 0.0810 - accuracy: 0.9733 - val_loss: 0.0830 - val_accuracy: 0.9770
Epoch 22/40
907/907 [==============================] - 32s 35ms/step - loss: 0.0757 - accuracy: 0.9761 - val_loss: 0.0829 - val_accuracy: 0.9765
Epoch 23/40
907/907 [==============================] - 34s 37ms/step - loss: 0.0739 - accuracy: 0.9764 - val_loss: 0.0793 - val_accuracy: 0.9810
Epoch 24/40
907/907 [==============================] - 32s 35ms/step - loss: 0.0727 - accuracy: 0.9763 - val_loss: 0.0751 - val_accuracy: 0.9820
Epoch 25/40
907/907 [==============================] - 32s 36ms/step - loss: 0.0710 - accuracy: 0.9767 - val_loss: 0.0609 - val_accuracy: 0.9860
Epoch 26/40
907/907 [==============================] - 33s 36ms/step - loss: 0.0658 - accuracy: 0.9789 - val_loss: 0.0744 - val_accuracy: 0.9840
Epoch 27/40
907/907 [==============================] - 32s 36ms/step - loss: 0.0656 - accuracy: 0.9792 - val_loss: 0.0654 - val_accuracy: 0.9850
Epoch 28/40
907/907 [==============================] - 34s 38ms/step - loss: 0.0662 - accuracy: 0.9782 - val_loss: 0.0638 - val_accuracy: 0.9875
Epoch 29/40
907/907 [==============================] - 33s 37ms/step - loss: 0.0619 - accuracy: 0.9796 - val_loss: 0.0883 - val_accuracy: 0.9790
Epoch 30/40
907/907 [==============================] - 34s 37ms/step - loss: 0.0576 - accuracy: 0.9813 - val_loss: 0.0609 - val_accuracy: 0.9870
Epoch 31/40
907/907 [==============================] - 34s 37ms/step - loss: 0.0603 - accuracy: 0.9801 - val_loss: 0.0641 - val_accuracy: 0.9855
Epoch 32/40
907/907 [==============================] - 34s 37ms/step - loss: 0.0597 - accuracy: 0.9810 - val_loss: 0.0704 - val_accuracy: 0.9835
Epoch 33/40
907/907 [==============================] - 35s 39ms/step - loss: 0.0548 - accuracy: 0.9825 - val_loss: 0.0582 - val_accuracy: 0.9865
Epoch 34/40
907/907 [==============================] - 34s 37ms/step - loss: 0.0539 - accuracy: 0.9822 - val_loss: 0.0545 - val_accuracy: 0.9880
Epoch 35/40
907/907 [==============================] - 34s 37ms/step - loss: 0.0515 - accuracy: 0.9831 - val_loss: 0.0468 - val_accuracy: 0.9890
Epoch 36/40
907/907 [==============================] - 34s 37ms/step - loss: 0.0525 - accuracy: 0.9828 - val_loss: 0.0563 - val_accuracy: 0.9870
Epoch 37/40
907/907 [==============================] - 33s 37ms/step - loss: 0.0521 - accuracy: 0.9830 - val_loss: 0.0627 - val_accuracy: 0.9855
Epoch 38/40
907/907 [==============================] - 35s 38ms/step - loss: 0.0503 - accuracy: 0.9838 - val_loss: 0.0670 - val_accuracy: 0.9830
Epoch 39/40
907/907 [==============================] - 34s 37ms/step - loss: 0.0476 - accuracy: 0.9844 - val_loss: 0.0538 - val_accuracy: 0.9865
Epoch 40/40
907/907 [==============================] - 34s 38ms/step - loss: 0.0484 - accuracy: 0.9842 - val_loss: 0.0567 - val_accuracy: 0.9865
```
Evaluating the Model's accuracy

```
model.evaluate(x_test,y_test)
-----------------------------------------------------------------------------------------
313/313 [==============================] - 3s 10ms/step - loss: 0.0571 - accuracy: 0.9826
[0.05708719417452812, 0.9825999736785889]
```
Accuracy PLot

```
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].legend(['Train', 'Val'])
```
![download](https://user-images.githubusercontent.com/111745916/194023950-f8365a5e-be95-47d2-8a96-ea9f27a3c993.png)

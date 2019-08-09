import os, time
import tensorflow as tf

# from tensorflow.python.client import device_lib
#
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
#
# print(tf.test.gpu_device_name())
# print(get_available_gpus())
# asdf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

begin = time.time()
model.fit(x_train, y_train, epochs=5, batch_size=1000)
print(time.time() - begin)
# model.evaluate(x_test, y_test)

"15.24682903289795"
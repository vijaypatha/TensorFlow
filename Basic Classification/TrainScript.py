#!/usr/bin/env python
# coding: utf-8

# In[ ]:


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[ ]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[ ]:


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(train_images, train_labels, epochs=15)


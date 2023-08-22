#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.losses import mae


# In[4]:


df = pd.read_csv( r"C:\Users\kashi\creditcard.csv")


# In[5]:


df.head(10)


# In[6]:


raw_data = df.values
print("Type: ", type(raw_data))
print("Shape: ", raw_data.shape)


# In[7]:


labels = raw_data[:, -1]
print(type(labels), labels.shape)


# In[8]:


data = raw_data[:, 0:-1]
print(type(data), data.shape)


# In[9]:


train_data, test_data, train_labels, test_labels = train_test_split(                                                                     data, labels, test_size=0.2, random_state=47)
print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)


# In[10]:


scaler = MinMaxScaler(feature_range=(0,1))
train_data_scaled = scaler.fit_transform(train_data)
print(train_data_scaled.min(), train_data_scaled.max())


# In[11]:


test_data_scaled = scaler.transform(test_data)
test_data_scaled.min(), test_data_scaled.max()


# In[12]:


normal_train_data_scaled = train_data_scaled[train_labels == 1]
normal_test_data_scaled = test_data_scaled[test_labels == 1]
anomolous_train_data_scaled = train_data_scaled[train_labels == 0]
anomolous_test_data_scaled = test_data_scaled[test_labels == 0]
normal_train_data_scaled.shape, anomolous_train_data_scaled.shape


# In[14]:


normal_test_data_scaled.shape, anomolous_test_data_scaled.shape


# In[15]:


plt.grid()
plt.plot(np.arange(30), normal_train_data_scaled[0])
plt.title('Normal ECG')
plt.show()


# In[16]:


plt.grid()
plt.plot(np.arange(30), anomolous_train_data_scaled[0])
plt.title('Anomolous ECG')
plt.show()


# In[17]:


# Creating the model
autoencoder = Sequential([
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(30, activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mae')


# In[18]:


history = autoencoder.fit(normal_train_data_scaled, normal_train_data_scaled,                          epochs=100, batch_size=512,                           validation_data=(normal_test_data_scaled, normal_test_data_scaled),                           shuffle=True)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()


# In[19]:


# Encoer layer Extraction 
encoder = Sequential()
for layer in autoencoder.layers[0:3]:
    encoder.add(layer)
encoder.compile(optimizer='adam', loss='mae')
# Extract the decoder layers from the autoencoder model
decoder = Sequential()
for layer in autoencoder.layers[3:]:
    decoder.add(layer)
decoder.compile(optimizer='adam', loss='mae')


# In[21]:


encoded_data = encoder(normal_test_data_scaled).numpy()
decoded_data = decoder(encoded_data).numpy()

plt.plot(normal_test_data_scaled[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(30), normal_test_data_scaled[0], decoded_data[0],                  color='lightcoral')
plt.legend(labels=['Input', 'Reconstruction', 'Error'])
plt.show()


# In[22]:


encoded_data = encoder(anomolous_test_data_scaled).numpy()
decoded_data = decoder(encoded_data).numpy()

plt.plot(anomolous_test_data_scaled[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(30), anomolous_test_data_scaled[0], decoded_data[0], color='lightcoral')
plt.legend(labels=['Input', 'Reconstruction', 'Error'])
plt.show()


# In[23]:


normal_reconstruction = autoencoder.predict(normal_test_data_scaled)
train_loss = mae(normal_reconstruction, normal_test_data_scaled)

plt.hist(train_loss[None, :], bins=50)
plt.xlabel("Train Loss")
plt.ylabel("Number of examples")
plt.show()

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)


# In[24]:


mean = np.mean(train_loss)
std = np.std(train_loss)
print(mean, std)

anomolous_reconstruction = autoencoder.predict(anomolous_test_data_scaled)
train_loss = mae(anomolous_reconstruction, anomolous_test_data_scaled)
plt.hist(train_loss[None, :], bins=50)
plt.xlabel("Train Loss")
plt.ylabel("Number of examples")
plt.show()


# In[25]:


mean = np.mean(train_loss)
std = np.std(train_loss)
print(mean, std)


def predict(model, data, treshold):
    reconstruction = model(data)
    loss = mae(reconstruction, data)
    return tf.math.less(loss, mean+std)


# In[26]:


from sklearn.metrics import accuracy_score


preds = predict(autoencoder, normal_test_data_scaled, threshold)

normal_test_labels = test_labels[test_labels == 1]

accuracy_score(preds, normal_test_labels)


# In[ ]:





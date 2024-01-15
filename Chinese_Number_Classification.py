# Ryan Skidmore
# ITP 259 Fall 2023
# Final Project

import tensorflow as tf
import seaborn as sb
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

# Problem 1
font_path = "SimHei.ttf"
chinese_font = FontProperties(fname=font_path)
data = pd.read_csv("chineseMNIST.csv")
# Problem 2
plt.figure(1)
sb.countplot(data=data, x="label")
plt.xticks(rotation=45)
plt.show()
# Problem 3
plt.figure(figsize=[10, 10])
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    random_select = np.random.randint(0, len(data))
    image = data.iloc[random_select, :4096].values.astype(float)
    plt.imshow(image.reshape(64, 64), cmap="gray")
    plt.title(data.loc[random_select, "character"], fontproperties=chinese_font)
    plt.xlabel(data.loc[random_select, "label"])
    plt.tight_layout()
plt.show()
# Problem 4
target_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8:
    '8', 9: '9', 10: '10', 100: '11', 1000: '12',
               10000: '13', 100000000: '14'}
reverse_dict = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十",
                11: "百", 12: "千",
                13: "万", 14: "亿"}
data = data.replace({"label": target_dict})
print(data["label"])

target = data["label"].astype(int)
chinese_chars = data["character"]
features = data.drop(["label", "character"], axis=1)

# Problem 5
x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.3, random_state=2023, stratify=target)
x_train_images = x_train.values.reshape(-1, 64, 64, 1) / 255
x_test_images = x_test.values.reshape(-1, 64, 64, 1) / 255
print(x_train_images.shape)
print(y_train.shape)
print(x_test_images.shape)
print(y_test.shape)

# Problem 6
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=9, padding="same", activation="relu", input_shape=(64, 64, 1)))
model.add(MaxPool2D())
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", input_shape=(64, 64, 1)))
model.add(MaxPool2D())
model.add(Conv2D(filters=32, kernel_size=5, padding="same", activation="relu", input_shape=(64, 64, 1)))
model.add(MaxPool2D())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(units=15, activation="softmax"))
# Problem 7
model.summary()
# Problem 8
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Problem 9
early_stopping = EarlyStopping(monitor="accuracy", patience=2, verbose= True, mode = "auto")
h = model.fit(x_train_images, y_train, epochs=10, validation_data=(x_test_images, y_test), callbacks=[early_stopping],
              batch_size=64, verbose=True)
# Problem 10
pd.DataFrame(h.history).plot()
plt.show()
# Problem 11
y_pred = model.predict(x_test_images)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
plt.figure(figsize=[10, 10])
for i in range(30):
    plt.subplot(6, 6, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test_images[i], cmap="gray")
    plt.title("True:" + str(reverse_dict[y_test.iloc[i]]) + "\nPredict:" + str(reverse_dict[y_pred[i]]),
              fontproperties=chinese_font)
    plt.tight_layout()
plt.show()
# Problem 12
failed_indices = []
idx = 0

for i in y_test:
    if i != y_pred[idx]:
        failed_indices.append(idx)
    idx = idx + 1

plt.figure(figsize=[10, 10])
for i in range(30):
    plt.subplot(6, 6, i + 1)
    random_select = np.random.randint(0, len(failed_indices))
    failed_index = failed_indices[random_select]
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test_images[failed_index], cmap="gray")
    pred_failed = reverse_dict[y_pred[failed_index]]
    actual_failed = reverse_dict[y_test.iloc[failed_index]]
    plt.title("Predicted:" + str(pred_failed) + "\nActual:" + str(actual_failed), fontproperties=chinese_font)
    plt.tight_layout()
plt.show()

"""
flowers dataset kullanılacak:
rgb: 224*224
CNN ile siniflandirma  modeli olusturma ve problemi cozme.

"""
# import libraries
from tensorflow_datasets import load # veri seti yukleme
from tensorflow.data import AUTOTUNE # veri seti optimizasyonu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, # 2D convolutional layer
    MaxPooling2D, # max pooling layer
    Flatten, # cok boyutlu veriyi tek boyutlu hale getirme
    Dense, # tam baglantılı katman
    Dropout # rastgele noronları kapatma ve overfittingi engelleme
)
from tensorflow.keras.optimizers import Adam # optimizer
from tensorflow.keras.callbacks import (
    EarlyStopping, # erken durdurma
    ReduceLROnPlateau, #ogrenme oranını azaltma
    ModelCheckpoint # model kaydetme
)

import tensorflow as tf
import matplotlib.pyplot as plt # gorsellestirme

# veri seti yukleme
(ds_train, ds_val), ds_info = load(
    "tf_flowers", # veri seti ismi
    split = ["train[:80%]", # veri setinin %80 i egitim için
             "train[80%:]"], # veri setinin %20 si test için 
    as_supervised=True, # veri setinin gorsel-etiket veri ciftinin olması
    with_info=True       # veri seti hakkında bilgi alma  
)
print(ds_info.features) # veri seti hakkında bilgi yazdırma
print("number of classes:", ds_info.features["label"].num_classes)


# ornek veri gorsellestirme
# egitim setinden rastgele 3 resim ve etiket alma
fig = plt.figure(figsize=(10,5))
for i, (image,label) in enumerate(ds_train.take(3)):
    ax = fig.add_subplot(1, 3, i+1) # 1.satır, 3.sutun, i+1.resim
    ax.imshow(image.numpy().astype("uint8")) # resmi gorsellestirme
    ax.set_title(f"etiket: {label.numpy()}") # etiket baslık olarak yazdırma
    ax.axis("off") # eksenleri kapatma

plt.tight_layout()
plt.show() # grafigi gosterme


# data augmentation + preprocessing (veri seti veri gorsel cogaltma)
IMG_SIZE = (180, 180)
def preprocess_train(image, label):
    """
    resize, randomflip, brightness, contrast, crop normalize
    """
    image = tf.image.resize(image, IMG_SIZE) # boyutlandırma
    image = tf.image.random_flip_left_right(image) # yatay olarak rastgele cevirme
    image = tf.image.random_brightness(image, max_delta=0.1) # rastgele parlaklık
    image = tf.image.random_contrast(image , lower=0.9, upper=1.2) # rastgele contrast
    image = tf.image.random_crop(image, size=(160, 160, 3)) # rastgele crop
    image = tf.image.resize(image, IMG_SIZE) # tekrar boyutlandırma
    image = tf.cast(image, tf.float32)/255.0 # normalize etme
    return image, label
def preprocess_val(image, label):
    """
    resize, normalize
    """
    image = tf.image.resize(image, IMG_SIZE) # boyutlandırma
    image = tf.cast(image, tf.float32)/255.0 # normalize etme
    return image, label

# veri setini hazırlamak
ds_train = (
    ds_train
    .map(preprocess_train, num_parallel_calls=AUTOTUNE) # on isleme ve augmentasyon
    .shuffle(1000) # karistirma
    .batch(32) # batch boyutu
    .prefetch(AUTOTUNE) # veri setini onceden hazırlamak
)
ds_val = (
    ds_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE) # on isleme
    .batch(32) # batch boyutu
    .prefetch(AUTOTUNE) # veri setini onceden hazırlamak
)

# CNN modelini olusturma
model = Sequential([
    # Feature Extraction Layers
    Conv2D(32, (3,3), activation = "relu", input_shape = (*IMG_SIZE, 3)), # 32 filtre,3*3 kernel, relu aktivasyon, 3 kanal(RGB)
    MaxPooling2D((2,2)), # 2*2 max pooling

    Conv2D(64, (3,3), activation = "relu"), # 64 filtre, 3*3 kernel, relu aktivasyon
    MaxPooling2D((2,2)), # 2*2 max pooling

    Conv2D(128, (3,3), activation = "relu"), # 128 filtre, 3*3 kernel, relu aktivasyon
    MaxPooling2D((2,2)), # 2*2 max pooling

    # classification layers
    Flatten(), # cok boyutlu veriyi vektore cevirir.
    Dense(128, activation = "relu"),
    Dropout(0.5), # overfittingi engellemek için dropout
    Dense(ds_info.features["label"].num_classes, activation = "softmax") # cıkıs katmanı, softmax aktivasyon

])

# callbacks
callbacks = [
    # eger val loss 3 epoch boyunca ıyıleşmezse egitimi durdur ve en ıyı agırlıkları yukle
    EarlyStopping(monitor="val_loss", patience = 3, restore_best_weights=True), # erken durdurma

    # val loss 2 epoch boyunca iyileşmezse learning rate 0.2 carpanı ile azalt
    ReduceLROnPlateau(monitor = "val_loss", factor = 0.2, patience = 2, verbose = 1, min_lr = 1e-9 ), # ogrenme oranını azaltma

    # her epoch sonunda eger model daha iyi ise kaydolur
    ModelCheckpoint("best_model.h5", save_best_only=True) # model kaydetme en ıyı modeli kaydet
]   

# derleme
model.compile(
    optimizer = Adam(learning_rate = 0.001), # adam optimizer ogrenme oranını 0.001 ayarla
    loss = "sparse_categorical_crossentropy", # kayıp fonksıyonu etiketler tam sayı oldugu için sparse kullanılır
    metrics = ["accuracy"] # metrik olarak dogruluk kullan

)
print(model.summary()) # model ozeti

# training 
history = model.fit(
    ds_train, # egitim veri seti
    validation_data = ds_val, # validasyon veri seti
    epochs = 10,
    callbacks = callbacks, 
    verbose = 1 # egitim
)
# model evulation
plt.figure(figsize=(12,5))

# dogruluk grafigi
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label = "egitim dogrulugu")
plt.plot(history.history["val_accuracy"], label= "validasyon dogrulugu")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("model accuracy")
plt.legend()

# loss plot
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label = "egitim kaybi")
plt.plot(history.history["val_loss"], label = "validasyon kaybı")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("model loss")
plt.legend()

plt.tight_layout()
plt.show()

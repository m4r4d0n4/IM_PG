import argparse
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import os
import tensorflow as tf

## MODEL0 VGG16
class Vgg16Model():

    def __init__(self, input_shape=(224, 224, 3), pretrained=False):
        self.input_shape = input_shape
        self.pretrained = pretrained
        self.model = self._build_model(pretrained)

    def _build_model(self, pretrained):
        if pretrained:
            self._base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

            # Congelar las capas del modelo base
            for layer in self._base_model.layers:
                layer.trainable = False

            input = self._base_model.input
            output = self._base_model.output
        else:
            # Dimensiones de las imagenes
            input = Input(self.input_shape) 

            ## Capas convolucionales
            # Bloque 1
            conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(input)
            conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
            pool1  = MaxPooling2D((2, 2))(conv2)

            # Bloque 2
            conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
            conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
            pool2  = MaxPooling2D((2, 2))(conv4)

            # Bloque 3
            conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
            conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
            conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
            pool3  = MaxPooling2D((2, 2))(conv7)

            # Bloque 4
            conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
            conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
            conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
            pool4  = MaxPooling2D((2, 2))(conv10)

            # Bloque 5
            conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
            conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
            conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
            output  = MaxPooling2D((2, 2))(conv13)

        ## Capas
        flat   = Flatten()(output)
        dense1 = Dense(4096, activation="relu")(flat)
        drop1  = Dropout(0.5)(dense1)
        dense2 = Dense(4096, activation="relu")(drop1)
        drop2  = Dropout(0.5)(dense2)
        output = Dense(2, activation="softmax")(drop2)

        vgg_16_model = Model(inputs=input, outputs=output)
        vgg_16_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        return vgg_16_model
    

    def train(self, train_generator, validation_generator):
        
        # Guarda el modelo con mejor precision en la validacion
        mcp = ModelCheckpoint('ISIC-images/modelVVG.h5', verbose=1)

        # Parametros 
        train_steps = train_generator.n // train_generator.batch_size
        validation_steps = validation_generator.n // validation_generator.batch_size
        epochs = 200

        # Entrenamiento del modelo
        return self.model.fit(train_generator, steps_per_epoch=train_steps, epochs=epochs, validation_data=validation_generator, validation_steps=validation_steps, verbose=1, callbacks=[mcp])
  
    def predict(self, data):
        return self.model.predict(data)





## PROCESAMIENTO DE LOS DATOS
def load_data():
    # Crear el DataFrame necesario para ImageDataGenerator
    dataset_path = './ISIC-images/' 

    # Cargar el fichero con las etiquetas
    data = pd.read_csv(dataset_path + 'metadata.csv')
    
    # Crear DataFrame con dos columnas -> ruta de la imagen | etiqueta (0:nevus, 1:melanoma)
    class_map = {'nevus': "nevus", 'melanoma': "melanoma"}
    new_data = pd.DataFrame({
        'x_col': data['isic_id'].apply(lambda x: os.path.join(dataset_path, x + '.jpg')),
        'y_col': data['diagnosis'].apply(lambda x: class_map[x] if x in class_map else 3)
    })
    seed = 142
    melanoma_data = new_data[new_data['y_col'] == "melanoma"].sample(n=3000, random_state=seed)
    benign_data = new_data[new_data['y_col'] == "nevus"].sample(n=3000, random_state=seed)

    # Combinar y mezclar los datos seleccionados
    new_data = pd.concat([melanoma_data, benign_data]).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Separacion Train Test -> 90% Train and 10% Test
    train_data, test_data = train_test_split(new_data, test_size=0.10, random_state=42, stratify=new_data['y_col'])

    # Crear el generador de datos de imagen en tiempo real -> Aplicar la normalización + Train Test Split (20% Validación y 80% Train)
    datagen = ImageDataGenerator(rescale=1./255., validation_split=0.20)

    batch_size = 64 

    # Generador de datos de entrenamiento
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_data,
        x_col='x_col',
        y_col='y_col',
        target_size=(224,224),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Generador de datos de validacion
    validation_generator = datagen.flow_from_dataframe(
        dataframe=train_data,
        x_col='x_col',
        y_col='y_col',
        target_size=(224,224),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    # Generador de datos de test
    test_generator = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_dataframe(
        dataframe=test_data,
        x_col='x_col',
        y_col='y_col',
        target_size=(224,224),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    return train_generator, validation_generator, test_generator


def main():

    # Cargar los datos 
    train, validation, test = load_data()
    
    print(test.labels)

    # Entrenamiento con transfer learning
    # model = Vgg16Model(pretrained=True)

    # Entrenamiento desde 0
    model = Vgg16Model(pretrained=False)

    history = model.train(train, validation)

    # Cargar el modelo
    # model = load_model('./ISIC-images/modelVVG.h5')
    
    ## ACCURACY ENTRENAMIENTO
    '''
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Perdida
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("ISIC-images/fig1.png")
    plt.show()
    '''

    # TEST SCORE
    prediction = model.predict(test)
    predicted_classes = np.argmax(prediction, axis=1)
    print("TEST")
    print(predicted_classes)
    test_labels = np.array([test.labels])
    print(test_labels)
    y_pred = predicted_classes
    y_true = test_labels
    # Verdaderos Positivos (TP): Ambos son 1
    TP = np.sum((y_true == 1) & (y_pred == 1))

    # Verdaderos Negativos (TN): Ambos son 0
    TN = np.sum((y_true == 0) & (y_pred == 0))

    # Falsos Positivos (FP): y_true es 0 pero y_pred es 1
    FP = np.sum((y_true == 0) & (y_pred == 1))

    # Falsos Negativos (FN): y_true es 1 pero y_pred es 0
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')


if __name__ == "__main__":
    main()

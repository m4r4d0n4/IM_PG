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




## MODEL VGG16
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
            # Image dimensions
            input = Input(self.input_shape) 

            ## Convolutional Layers
            # Block 1
            conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(input)
            conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
            pool1  = MaxPooling2D((2, 2))(conv2)

            # Block 2
            conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
            conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
            pool2  = MaxPooling2D((2, 2))(conv4)

            # Block 3
            conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
            conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
            conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
            pool3  = MaxPooling2D((2, 2))(conv7)

            # Block 4
            conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
            conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
            conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
            pool4  = MaxPooling2D((2, 2))(conv10)

            # Block 5
            conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
            conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
            conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
            output  = MaxPooling2D((2, 2))(conv13)

        ## Dense Layers
        flat   = Flatten()(output)
        dense1 = Dense(4096, activation="relu")(flat)
        drop1  = Dropout(0.5)(dense1)
        dense2 = Dense(4096, activation="relu")(drop1)
        drop2  = Dropout(0.5)(dense2)
        output = Dense(1, activation="sigmoid")(drop2)

        vgg_16_model = Model(inputs=input, outputs=output)
        vgg_16_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        return vgg_16_model
    

    def train(self, train_generator, validation_generator):
        # Detiene el entrenamiento si la precision de la validacion no mejora en despues de 2 epocas
        early_stop = EarlyStopping(monitor="val_loss", patience=2, verbose=1)

        # Guarda el modelo con mejor precision en la validacion
        mcp = ModelCheckpoint('modelVVG.h5', verbose=1)

        # Parametros !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        train_steps = train_generator.n // train_generator.batch_size
        validation_steps = validation_generator.n // validation_generator.batch_size
        epochs = 30

        # Entrenamiento del modelo
        return self.model.fit(train_generator, steps_per_epoch=train_steps, epochs=epochs, validation_data=validation_generator, validation_steps=validation_steps, verbose=1, callbacks=[mcp,early_stop])

    
    def predict(self, data):
        return self.model.predict(data)





## DATA PROCESSING
def load_data(data_good, data_bad):
    # Combine the two DataFrames
    combined_data = pd.concat([data_good, data_bad], ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_data.to_csv('dataset_nuria/metadata.csv', index=False)

    # Create the DataFrame necessary for ImageDataGenerator
    dataset_path = 'dataset_nuria/' # Cambiar ruta del archivo CSV !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Load the file with the labels
    data = pd.read_csv(dataset_path + 'metadata.csv')
    
    # Create the DataFrame with two columns -> image path | label (0:benign, 1:melanoma)
    new_data = pd.DataFrame({'x_col': dataset_path + data['isic_id'] + '.jpg',
                         'y_col': data['benign_malignant'].apply(lambda x: 'benign' if x == 'benign' else 'melanoma')}) 

    # Tarin Test Separation -> 90% Train and 10% Test
    train_data, test_data = train_test_split(new_data, test_size=0.10, random_state=42, stratify=new_data['y_col'])

    # Create the real-time image data generator -> Apply normalization + Train Test Split (20% Validation and 80% Train)
    datagen = ImageDataGenerator(rescale=1./255., validation_split=0.20)

    batch_size = 10 # Lo ponen a 128 pero con 100 imagenes no puedo ponerlo a tanto!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Training data generator
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

    # Validation data generator
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


    # No se si hace falta para el dataset final -> tengo dos archivos csv porque me he descargado por separado las img benignas y las malignas -> concateno los csv !!!!!!!!!!!!!!!!!!!!!!!!!
    # Load data from the two CSV files
    data_good = pd.read_csv('dataset_nuria/metadata_1.csv')
    data_bad = pd.read_csv('dataset_nuria/metadata_2.csv')

    train, validation, test = load_data(data_good, data_bad)
    
    print(test.labels)

    # For transfer learning
    model = Vgg16Model(pretrained=True)

    # For normal training
    # model = Vgg16Model(pretrained=False)

    history = model.train(train, validation)
    
    ## TRAIN ACCURACY
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
    plt.show()


    # TEST SCORE
    prediction = model.predict(test).argmax(axis=1)
    test_labels = test.labels
    test_confusion = confusion_matrix(test_labels, prediction)
    test_accuracy = accuracy_score(test_labels, prediction, normalize=True, sample_weight=None)
    print(test_confusion)
    print(test_accuracy)

    print(classification_report(test.classes, prediction, zero_division=0))


if __name__ == "__main__":
    main()

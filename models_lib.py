from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalMaxPool2D, Input, BatchNormalization
from keras.regularizers import l2, l1
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Nadam, Adamax, Ftrl
from keras.applications.vgg16 import VGG16
from keras.metrics import Precision, Recall, F1Score, AUC
from keras import layers
from keras.utils import plot_model


precision_per_class = [Precision(class_id=i, name=f'precision_class{i}') for i in range(10)]
recall_per_class = [Recall(class_id=i, name=f'recall_class{i}') for i in range(10)]

all_metrics = ['accuracy'] + precision_per_class + recall_per_class


class MyVGGModel():
    def build(self):
        regul_coeff = 0.0001
        
        inputs = Input(shape=(32, 32, 3))
        
        x = Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(rate=0.1)(x)

        x = Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(rate=0.2)(x)

        x = Conv2D(256, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(rate=0.3)(x)

        x = Conv2D(512, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(rate=0.4)(x)

        x = Conv2D(512, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((1, 1))(x)
        x = Dropout(rate=0.5)(x)

        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        outputs = Dense(10, activation="softmax")(x)
        
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=Adam(learning_rate=0.0005),
                      loss='categorical_crossentropy',
                      metrics=all_metrics)

        return model
    

class MyModel():
    def build(self):
        regul_coeff = 0.0001
        
        inputs = Input(shape=(32, 32, 3))
        
        x = Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        block_1_output = MaxPooling2D((2, 2))(x)
        x = Dropout(rate=0.1)(x)

        x = Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(block_1_output)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        block_2_output = MaxPooling2D((2, 2))(x)
        x = Dropout(rate=0.2)(x)

        x = Conv2D(256, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(block_2_output)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        block_3_output = MaxPooling2D((2, 2))(x)
        x = Dropout(rate=0.3)(x)

        x = Conv2D(512, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(block_3_output)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((1, 1))(x)
        x = Dropout(rate=0.4)(x)
        block_4_output = layers.add([x, block_3_output])

        x = Conv2D(512, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(block_4_output)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation="relu", kernel_regularizer=l2(regul_coeff), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(rate=0.4)(x)

        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        outputs = Dense(10, activation="softmax")(x)
        
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=Adam(learning_rate=0.0005),
                      loss='categorical_crossentropy',
                      metrics=all_metrics)

        return model
    
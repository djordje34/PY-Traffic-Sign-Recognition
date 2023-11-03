from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dense,\
Flatten, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l1, l2
from PIL import ImageFont
import visualkeras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import seaborn as sns
from typing import Self
from backend.collector import Collector

class Model():
    
    def __init__(self,target_size:tuple[int,int]=(32,32)) -> None:
        print("Initializing model constraints...")
        self.root = "../Dataset"
        self.train, self.test_dataset = Collector.aggregate(self.root, target_size=target_size)
        num_samples = sum(1 for _ in self.train)
        train_size = int(0.8 * num_samples)
        self.train_dataset = self.train.take(train_size)
        self.validation_dataset = self.train.skip(train_size)
        self.train_dataset = self.train_dataset.batch(32)
        self.validation_dataset = self.validation_dataset.batch(32)
        self.test_dataset = self.test_dataset.batch(32)
        self.name=f"model_{target_size[0]}x{target_size[1]}"
        self.model = Sequential(name=self.name)
        self.model.add(Input(shape=(target_size[0],target_size[1],3), name="input_layer"))
        print("Done!")
    
    def load_model(self)->Self:
        try:
            self.model = tf.keras.models.load_model(f"models/{self.name}.keras")
            print("Model succesfully loaded.")
        except:
            print("Unexistant model.")
        return self
    def plot(self):
        plot_model(self.model, to_file=f"plots/{self.model.name}_g.png", show_shapes=True, show_layer_names=True)
        font = ImageFont.truetype("arial.ttf", 12)
        visualkeras.layered_view(self.model, legend=True, font=font,to_file=f"plots/{self.model.name}_a.png")
    
    def inputnet_conv(self,filters:int=32,kernel_size:int=7, padding:str='valid',pool_size:int=2,\
                        dropout_rate:float=0.25, l1_reg:float=0.01, l2_reg:float=0.01)->Self:
        self.model.add(Conv2D(filters, kernel_size=(kernel_size, kernel_size), padding=padding, activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l1(l1_reg)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=pool_size))
        self.model.add(Dropout(dropout_rate))
        return self
    
    def midnet_conv(self,filters:int=32,kernel_size:int=3, padding:str='valid',pool_size:int=2,\
                    dropout_rate:float=0.25, l1_reg:float=0.01, l2_reg:float=0.01)->Self:
        self.model.add(Conv2D(filters, kernel_size=(kernel_size, kernel_size), padding=padding, activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l1(l1_reg)))
        self.model.add(Conv2D(filters, kernel_size=(kernel_size, kernel_size), padding=padding, activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l1(l1_reg)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=pool_size))
        self.model.add(Dropout(dropout_rate))
        return self
    
    def midnet_flatten(self)->Self:
        self.model.add(Flatten())
        return self
    
    def fc_block(self, units:int,activation:str='relu',dropout_rate:float=0.3,\
                l1_reg:float=0.01, l2_reg:float=0.01)->Self:
        self.model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg), bias_regularizer=l1(l1_reg)))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation))
        self.model.add(Dropout(dropout_rate))
        return self
    
    def outnet_block(self)->Self:
        self.model.add(Dense(62,activation='softmax'))
        return self
        
    def compile(self, optimizer:str = 'adam', loss:str = 'categorical_crossentropy',\
                metrics: list[str] = ['accuracy','categorical_crossentropy'], learning_rate:float = 0.001)->Self:                                    
        optimizer = optimizers.get(optimizer)
        optimizer.learning_rate.assign(learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self
    
    def fit(self, batch_size: int = 32, epochs: int = 100) -> any:
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True 
        )
        
        reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

        
        
        history = self.model.fit(
        self.train_dataset,
        validation_data=self.validation_dataset,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr_callback]
    )
        train_accuracy, val_accuracy = history.history['accuracy'], history.history['val_accuracy']
        learning_rate, = history.history['lr']
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

        ax[0].set_title('Тачност по епохама')
        ax[0].plot(train_accuracy, 'o-', label='Тренинг - тачност')
        ax[0].plot(val_accuracy, 'o-', label='Валидација - тачност')
        ax[0].set_xlabel('Епоха')
        ax[0].set_ylabel('Тачност')
        ax[0].legend(loc='best')

        ax[1].set_title('Стопа учења по епохама')
        ax[1].plot(learning_rate, 'o-', label='Стопа учења')
        ax[1].set_xlabel('Епоха')
        ax[1].set_ylabel('Губитак')
        ax[1].legend(loc='best')

        plt.tight_layout()
        plt.savefig(f'plots/{self.name}_report.png')
        
        return history
    
    def predict(self)->np.ndarray:
        predictions = self.model.predict(self.test_dataset)
        self.test_dataset = self.test_dataset.unbatch()
        images, labels = tuple(zip(*self.test_dataset))
        decoded_labels = tf.argmax(labels, axis=-1)
        decoded_pred = tf.argmax(predictions, axis=-1)
        orig = np.array(decoded_labels)
        confusion = confusion_matrix(decoded_labels, decoded_pred,labels=range(0,62))
        self.actual_out = orig
        plt.figure(figsize=(8, 6)) 
        sns.set(font_scale=0.6) 
        sns.heatmap(confusion, annot=True, fmt='d', cmap='twilight', cbar=True)
        plt.xlabel("Претпостављено")
        plt.ylabel("Стварно")
        plt.savefig(f'plots/confm_{self.name}.png')
        report = classification_report(decoded_labels, decoded_pred)
        with open(f'plots/classification_report_{self.name}.txt', 'w') as report_file:
            report_file.write(report)
        return np.array(decoded_pred)
    
    def evaluate(self)->np.ndarray:
        test_loss, test_accuracy, _ = self.model.evaluate(self.test_dataset)
        plt.figure()
        labels, values = ['Тест - губитак', 'Тест - тачност'], [test_loss, test_accuracy]
        plt.bar(labels, values, color=['red','green'])
        plt.title('Тест - губитак и тачност')
        plt.savefig(f"plots/{self.name}_testacc.png")
        
    def save(self)->None:
        self.model.save(f"models/{self.model.name}.keras")
        
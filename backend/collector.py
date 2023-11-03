import os
import numpy as np
from PIL import Image
import tensorflow as tf


class Collector():
    @staticmethod
    def aggregate(root: str = None, target_size: tuple[int, int] = (32, 32)) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        train_dir, test_dir = [os.path.join(os.path.dirname(__file__), root, x) for x in ['Training', 'Testing']]
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=target_size,
            batch_size=None,
            label_mode='categorical',
            seed=1337,
            crop_to_aspect_ratio=True
        )
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=target_size,
            shuffle=False,
            batch_size=None,
            label_mode='categorical',
            seed=1337,
            crop_to_aspect_ratio=True
        )
        train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
        test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y))
        return train_dataset, test_dataset
    
    @staticmethod
    def read_img(subroot:str=None)->tuple[np.ndarray,np.ndarray]:
        print(f"Reading data from {subroot}")
        data = []
        labels = []
        for class_name in os.listdir(subroot):
            print(f"Current directory: {class_name}")
            if not os.path.isdir(os.path.join(subroot, class_name)):
                continue 
            class_dir = os.path.join(subroot, class_name)
            for filename in os.listdir(class_dir):
                if not filename.endswith('.ppm'):
                    continue
                filepath = os.path.join(class_dir,filename)
                label = class_name.lstrip('0')
                data.append((np.array(Image.open(filepath)).astype('float32')))
                labels.append(label.zfill(1))
        return data, labels
    
    @staticmethod
    def aggregate_orig(root:str=None) -> tuple[np.ndarray, np.ndarray]:
        trainroot, testroot = (os.path.join(os.path.dirname(__file__), root, x) for x in ['Training','Testing'])
        train, test = [],[]
        train, _ = Collector.read_img(trainroot)
        test, _ = Collector.read_img(testroot)
        return train,test
    
    @staticmethod
    def aggregate_labels(root:str=None) -> tuple[np.ndarray, np.ndarray]:
        trainroot, testroot = (os.path.join(os.path.dirname(__file__), root, x) for x in ['Training','Testing'])
        _, train_labels = Collector.read_img(trainroot)
        _, test_labels = Collector.read_img(testroot)
        return train_labels,test_labels
    
    @staticmethod
    def read_labels(root:str=None)->tuple[np.ndarray, np.ndarray]:
        trainroot, testroot = (os.path.join(os.path.dirname(__file__), root, x) for x in ['Training','Testing'])
        _, train_labels = Collector.read_img(trainroot)
        _, test_labels= Collector.read_img(testroot)
        return train_labels,test_labels
    ########################################################
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from PIL import Image

class Preprocessor():
    
    @staticmethod
    def resize_image(image:np.ndarray, target_size:tuple[int, int])->np.ndarray:
        original_height, original_width, _ = image.shape
        target_height, target_width = target_size
        original_aspect_ratio = original_width / original_height
        target_aspect_ratio = target_width / target_height

        if original_width <= target_width and original_height <= target_height:
            new_width = target_width
            new_height = target_height
        elif original_aspect_ratio > target_aspect_ratio:
            new_width = target_width
            new_height = int(new_width / original_aspect_ratio)
        else:
            new_height = target_height
            new_width = int(new_height * original_aspect_ratio)

        pillow_image = Image.fromarray(image)
        resized_image = pillow_image.resize((new_width, new_height), Image.LANCZOS)
        resized_image = np.array(resized_image)
        horizontal_offset = (target_width - new_width) // 2
        vertical_offset = (target_height - new_height) // 2
        padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded_image[vertical_offset:vertical_offset + new_height, horizontal_offset:horizontal_offset + new_width, :] = resized_image

        return padded_image

    @staticmethod
    def resize_dataset(data_array:np.ndarray, target_size:tuple[int,int])->None:
        print("Resizing dataset...")
        for data in data_array:
            data['image'] = Preprocessor.resize_image(data['image'], target_size)
        print("Done!")
        
        
    @staticmethod
    def partition_dataset(data_array:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        X_train, X_val, y_train, y_val = train_test_split(data_array['image'],
                                            data_array['label'], test_size=0.2,
                                            stratify=data_array['label'], random_state=42)
    
        
        return X_train, y_train, X_val, y_val
    
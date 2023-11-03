from PIL import Image
import os

root_dir = "../Dataset"
train_dir, test_dir = [os.path.join(os.path.dirname(__file__), root_dir, x) for x in ['Training', 'Testing']]

def convert_and_save_images_in_folder(folder_path):
    for subdir, dirs, files in os.walk(folder_path):
        print(subdir,dirs,files)
        for file in files:
            if file.endswith(".ppm"):
                ppm_path = os.path.join(subdir, file)
                image = Image.open(ppm_path)
                png_file = os.path.splitext(file)[0] + ".png"
                png_path = os.path.join(subdir, png_file)
                image.save(png_path)


convert_and_save_images_in_folder(train_dir)
convert_and_save_images_in_folder(test_dir)
print("PPM to PNG conversion complete.")
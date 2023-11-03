import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from collector import Collector


class Analyzer():
    def __init__(self) -> None:
        self.cmap = plt.get_cmap('twilight')
        self.root = "../Dataset"
        self.train, self.test = Collector.aggregate_orig(self.root)
        self.train_labels,self.test_labels = Collector.read_labels(self.root)
        
    def class_distribution_plot(self, data_array: np.ndarray) -> None:
        class_counts = {}
        for labels in list(map(int, data_array)):
            if labels not in class_counts:
                class_counts[labels] = 1
            class_counts[labels] += 1
        sorted_class_counts = sorted(class_counts.items(), key=lambda item: item[0])
        classes, counts = zip(*sorted_class_counts)
        plt.figure(figsize=(15, 5))
        plt.bar(classes, counts, color=self.cmap(300))
        plt.xlabel('Класа')
        plt.ylabel('Број слика')
        plt.title('Расподела слика по класама')
        plt.xticks(ticks=classes, labels=classes, fontsize=7)
        plt.savefig('plots/class_distribution_plot.png')
        plt.close()

        
    def size_distribution_plot(self) -> None:
        base = 25000
        _, data_array = Collector.aggregate_orig(self.root)
        size_classes = [[] for _ in range(10)]
        def group_images_by_size(data_array):
            for i, image in enumerate(data_array):
                size = image.shape[0] * image.shape[1]
                for j in range(10):
                    if size < (j + 1) * base:
                        size_classes[j].append(size)
                        break
                else:
                    size_classes[9].append(size)
        group_images_by_size(data_array)
        plt.figure(figsize=(10, 5))
        for class_num in range(10):
            plt.subplot(2, 5, class_num + 1)
            sizes_in_class = size_classes[class_num]
            plt.hist(sizes_in_class, bins=20, edgecolor='k', color=self.cmap(300))
            plt.xticks(fontsize=7)
            plt.xlabel('Број пиксела слике')
            plt.ylabel('Број слика')
            if class_num != 9:
                plt.title(f"Од {class_num * base} до {(class_num + 1) * base} пиксела", fontsize=7)
            else:
                plt.title(f"Веће од {class_num * base} пиксела", fontsize=7)
        plt.tight_layout()
        plt.savefig('plots/size_distribution_plot_test.png')
        plt.close()

    def image_show(self, data_array: tf.data.Dataset, seed: int = 3, num: int = 5) -> None:
        random.seed(seed)
        num_total = len(data_array)
        indices = random.sample(range(num_total), num)
        plt.figure(figsize=(12, 4))
        for i, idx in enumerate(indices):
            image_data = tf.image.resize(data_array.element_spec[0].shape, (32, 32))
            label = int(data_array.element_spec[1].shape)
            plt.subplot(1, num, i + 1)
            plt.imshow(image_data.numpy().astype(np.uint8))
            plt.title(f'Класа: {label}')
        plt.tight_layout()
        plt.savefig('plots/image_show.png')
        plt.close()

    def pixel_heatmap(self, seed:int=3)->None:
        random.seed(seed)
        idx = random.randint(0,len(self.train)-1)
        image_data = self.train[idx]
        label = self.train_labels[idx]
        ch_r,ch_g,ch_b = image_data[:,:,0],image_data[:,:,1],image_data[:,:,2]
        plt.figure(figsize=(16, 4))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.1)
        plt.subplot(gs[0])
        sns.heatmap(ch_r, cmap='Reds', square=True, annot=False)
        plt.title(f"Матрица корелације за црвену боју",fontsize=10)
        plt.subplot(gs[1])
        sns.heatmap(ch_g, cmap='Greens', square=True, annot=False)
        plt.title(f"Матрица корелације за зелену боју",fontsize=10)
        plt.subplot(gs[2])
        sns.heatmap(ch_b, cmap='Blues', square=True, annot=False)
        plt.title(f"Матрица корелације за плаву боју",fontsize=10)
        plt.suptitle(f"Матрица корелације за примерак из класе {label}")
        plt.savefig('plots/pixel_heatmap.png')
        plt.close()


    def min_max_size(self,target_size:tuple[int,int])->None:
        images = Collector.aggregate_orig(self.root)[0]
        sizes = np.array([data.shape[0] * data.shape[1] for data in images])
        i_min, i_max = np.argmin(sizes), np.argmax(sizes)
        smallest_image = images[i_min] / 255.0
        largest_image = images[i_max] / 255.0
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.imshow(smallest_image)
        plt.title("Најмања слика")
        plt.xlabel(f"{smallest_image.shape[1]}")
        plt.ylabel(f"{smallest_image.shape[0]}")
        plt.subplot(122)
        plt.imshow(largest_image)
        plt.title("Највећа слика")
        plt.xlabel(f"{largest_image.shape[1]}")
        plt.ylabel(f"{largest_image.shape[0]}")
        plt.savefig('plots/size_comparison.png')
        smallest_image_resized = tf.image.resize(smallest_image, target_size)
        largest_image_resized = tf.image.resize(largest_image, target_size)
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.imshow(smallest_image_resized)
        plt.title("Најмања слика са промењеним димензијама")
        plt.xlabel(f"{smallest_image_resized.shape[1]}")
        plt.ylabel(f"{smallest_image_resized.shape[0]}")
        plt.subplot(122)
        plt.imshow(largest_image_resized)
        plt.title("Највећа слика са промењеним димензијама")
        plt.xlabel(f"{largest_image_resized.shape[1]}")
        plt.ylabel(f"{largest_image_resized.shape[0]}")
        plt.savefig('plots/resized_comparison.png')
        plt.close()
        
a = Analyzer()
#a.class_distribution_plot(a.train_labels)
#a.size_distribution_plot()
#a.pixel_heatmap(a.train)
a.min_max_size(target_size=(32,32))
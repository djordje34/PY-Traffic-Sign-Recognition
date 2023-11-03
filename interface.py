from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from backend.model import Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    models = []
    names=['model_32x32','model_48x48','model_64x64','model_72x72','model_128x128']
    sizes=[32,48,64,72,128]
    for i,val in enumerate(names):  
        print(f"Initializing {val} for dataset w/ size ({sizes[i]},{sizes[i]})")
        models.append(Model(target_size=(sizes[i],sizes[i])).load_model())
    all_predictions = [model.predict() for model in models]
    ensemble_predictions = []
    for i in range(len(models[0].actual_out)):
        input_predictions = [predictions[i] for predictions in all_predictions]
        most_common_prediction = Counter(input_predictions).most_common(1)[0][0]
        ensemble_predictions.append(most_common_prediction)
    accuracy = accuracy_score(models[0].actual_out,ensemble_predictions)
    report = classification_report(models[0].actual_out,ensemble_predictions)
    confusion = confusion_matrix(models[0].actual_out,ensemble_predictions,labels=range(0,62))
    plt.figure(figsize=(8, 6)) 
    sns.set(font_scale=0.6) 
    sns.heatmap(confusion, annot=True, fmt='d', cmap='twilight', cbar=True)
    plt.xlabel("Претпостављено")
    plt.ylabel("Стварно")
    plt.savefig(f'plots/confm_ensemble.png')
    with open(f'plots/classification_report_ensemble.txt', 'w') as report_file:
        report_file.write(report)
    


def main2():
    wrapper = Model(target_size=(128,128))
    wrapper\
    .midnet_conv()\
    .midnet_conv(filters=64)\
    .midnet_flatten()\
    .fc_block(512)\
    .outnet_block()\
    .compile()\
    .plot()
    wrapper.fit(epochs=100)
    wrapper.save()
    wrapper.evaluate()
    wrapper.predict()
    
def main3():
    mod = Model(target_size=(32,32)).load_model()
    pred = mod.predict()
    
if __name__ == "__main__":
    main()
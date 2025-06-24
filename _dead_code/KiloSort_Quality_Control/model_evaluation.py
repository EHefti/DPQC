import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import spikeinterface.curation as sc
import spikeinterface.full as si


def plot_model_evaluation(analyzer, model_folder, manual_labels):
    
    print("---Model Evaluation---")
    print("")
    model, model_info = sc.load_model(
        model_folder=model_folder,
        trusted=['numpy.dtype']
    )

    labels_and_probabilities = si.auto_label_units(
        sorting_analyzer=analyzer,
        model_folder=model_folder,
        trust_model=True
    )

    avg_confidence = mean(labels_and_probabilities["probability"])
    print(labels_and_probabilities.head())
    print('...')
    print(f'The average confidence of the model is {avg_confidence:.3f}.')
    print("")
    print("---------------")
    print("")

    predictions = labels_and_probabilities['prediction'].tolist()
    
    class_labels_ordered = sorted(list(model_info['label_conversion'].values()))

    if len(manual_labels) != len(predictions):
        print("Warning: Length of manual_labels and model predictions do not match. Cannot generate confusion matrix.")
        return

    conf_matrix = confusion_matrix(manual_labels, predictions, labels=class_labels_ordered)

    balanced_accuracy = balanced_accuracy_score(manual_labels, predictions)

    plt.figure(figsize=(7, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black",
                     fontsize=15, weight='bold')
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Manual Label', fontsize=12)
    
    plt.xticks(ticks=np.arange(len(class_labels_ordered)), labels=class_labels_ordered, rotation=45, ha='right', fontsize=10)
    plt.yticks(ticks=np.arange(len(class_labels_ordered)), labels=class_labels_ordered, fontsize=10)
    
    plt.title('Predicted vs Manual Label', fontsize=14, pad=20)
    plt.suptitle(f"Balanced Accuracy: {balanced_accuracy:.3f}", fontsize=16, weight='bold', y=0.98)
    plt.colorbar(label='Count', shrink=0.7)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
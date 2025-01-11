import seaborn as sns
import matplotlib.pyplot as plt
from confu import *
from save_load import *
def plot1():
    # Given values and methods
    x = ['accuracy', 'precision', 'sensitivity', 'specificity', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr']
    y = multi_confu_matrix(load("y_test"),load("y_pred"))
    y= [i*100 for i in y]

    # Increase DPI by setting the figure size and DPI
    plt.figure(figsize=(8, 3), dpi=200)


    sns.barplot(x=y, y=x, palette='flare')
    #sns.color_palette("Spectral", as_cmap=True)
    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Metrics')

    # Show the plot
    plt.tight_layout()
    plt.savefig("Results/mat.png",dpi=400)
    plt.show()

    import pandas as pd
    metrics_df = pd.DataFrame({'Metric': x, 'Value': y})
    metrics_df.to_csv("Results/res.csv")
    # Display the DataFrame
    print(metrics_df)

def plot2():
    mat = confusion_matrix(load("y_test"), load("y_pred"))

    # Plot confusion matrix
    plt.figure(figsize=(8, 4))
    sns.heatmap(mat, annot=True, cmap="crest", fmt="d", xticklabels=['champaka', 'chitrak', 'Common_Lanthana', 'Hibiscus', 'honeysuckle', 'indian-mallow', 'Jatropha', 'malabar_melastome', 'Marigold','Rose','shankupushpam', 'spider_lily', 'sunflower'], yticklabels=['champaka', 'chitrak', 'Common_Lanthana', 'Hibiscus', 'honeysuckle', 'indian-mallow', 'Jatropha', 'malabar_melastome', 'Marigold','Rose','shankupushpam', 'spider_lily', 'sunflower'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    plt.savefig("Results/confu.png", dpi=400)
    plt.show()



#plot1()
plot2()
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

data_path='/data/zhuoer/AG_news'

train_df = pd.read_csv(data_path + '/train_5000_head.csv')  #
# train_texts = np.array([v for v in train_df['review']])  #
train_labels = np.array([v-1 for v in train_df['label']]).tolist()  #

train_df_preds = pd.read_csv(data_path + '/train_asym_0.4_5000.csv',header=None)  #
train_preds = np.array([v for v in train_df_preds[0]]).tolist() 

# # Calculate confusion matrix
# cm = confusion_matrix(train_labels, train_preds,normalize='true')
# print("Confusion Matrix:")
# print(cm)

ConfusionMatrixDisplay.from_predictions(
y_true=train_labels,
y_pred=train_preds,
normalize='true',
display_labels=['World','Sports','Business','Sci/Tech'],
cmap=plt.cm.Blues
)

plt.figure(figsize=(8, 6))
plt.show()
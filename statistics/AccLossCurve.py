import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel(r'ACC_VAL_S1.xlsx', sheet_name=0, header=None)
train_data = np.array(data)  # np.ndarray()
excel_list = train_data.tolist()  # list
excel_listV1 = list(map(list, zip(*excel_list)))
acc = excel_listV1[0]
val_acc = excel_listV1[1]
Loss = excel_listV1[2]
val_loss = excel_listV1[3]
epochs = range(0, 100)

plt.plot(epochs, acc, linestyle='-', color='b', label='Training accuracy')
plt.plot(epochs, val_acc, linestyle='--', color='r', label='validation accuracy')
plt.title('(b) Training and Validation Accuracy of Strategy1')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(alpha=1, linestyle='-.')
plt.savefig("(b)accuracyof Strategy1.jpg")
plt.figure()
plt.show()

data = pd.read_excel(r'ACC_VAL_S2.xlsx', sheet_name=0, header=None)
train_data = np.array(data)  # np.ndarray()
excel_list = train_data.tolist()  # list
excel_listV1 = list(map(list, zip(*excel_list)))
accV2 = excel_listV1[0]
val_accV2 = excel_listV1[1]
LossV2 = excel_listV1[2]
val_lossV2 = excel_listV1[3]
epochs = range(0, 100)
plt.plot(epochs, accV2, linestyle='-', color='b', label='Training accuracy')
plt.plot(epochs, val_accV2, linestyle='--', color='r', label='validation accuracy')
plt.title('(c) Training and Validation Accuracy of Strategy2')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(alpha=1, linestyle='-.')
plt.savefig("(c)accuracy of Strategy2.jpg")
plt.figure()
plt.show()

data = pd.read_excel(r'ACC_VAL_S0.xlsx', sheet_name=0, header=None)
train_data = np.array(data)  # np.ndarray()
excel_list = train_data.tolist()  # list
excel_listV1 = list(map(list, zip(*excel_list)))
accV3 = excel_listV1[0]
val_accV3 = excel_listV1[1]
LossV3 = excel_listV1[2]
val_lossV3 = excel_listV1[3]
epochs = range(0, 100)

plt.plot(epochs, accV3, linestyle='-', color='b', label='Training accuracy')
plt.plot(epochs, val_accV3, linestyle='--', color='r', label='validation accuracy')
plt.title('(a) Training and Validation Accuracy of Strategy0')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(alpha=1, linestyle='-.')
plt.savefig("(a)accuracy of Strategy0.jpg")
plt.figure()
plt.show()


'''
plt.plot(epochs, lossV3, linestyle='-', color='b', label='Training Loss')
plt.plot(epochs, val_lossV3, linestyle='--', color='r', label='validation Loss')
plt.title('(a) Training and Validation Loss of Strategy0')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(alpha=1, linestyle='-.')
plt.savefig("(a)loss of Strategy0.jpg")
plt.figure()
plt.show()



plt.plot(epochs, loss, linestyle='-', color='b', label='Training loss')
plt.plot(epochs, val_loss, linestyle='--', color='r', label='validation loss')
plt.title('(b) Training and Validation Loss of Strategy1')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(alpha=1, linestyle='--')
plt.savefig("(b)loss of Strategy1.jpg")
plt.show()


plt.plot(epochs, lossV2, linestyle='-', color='b', label='Training loss')
plt.plot(epochs, val_lossV2, linestyle='--', color='r', label='validation loss')
plt.title('(c) Training and Validation Loss of Strategy2')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(alpha=1, linestyle='--')
plt.savefig("(c)loss of Strategy2.jpg")
plt.show()
'''


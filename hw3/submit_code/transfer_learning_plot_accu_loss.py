
# coding: utf-8

# In[9]:

# transfer learning
# plot accuracy and loss

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

acc_history = np.load('p2_acc_history.npy') * 100
loss_history = np.load('p2_loss_history.npy')
# print(acc_history)
# print(loss_history)

plt.figure()
plt.plot(acc_history)
plt.legend(('train', 'test'))
plt.title('Accuracy vs epochs')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epochs')
plt.show()

plt.figure()
plt.plot(loss_history)
plt.legend(('train', 'test'))
plt.title('Loss vs epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()


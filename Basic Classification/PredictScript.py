#!/usr/bin/env python
# coding: utf-8

# In[ ]:


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


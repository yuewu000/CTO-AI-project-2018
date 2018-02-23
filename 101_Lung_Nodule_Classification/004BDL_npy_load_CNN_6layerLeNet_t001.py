
# coding: utf-8

# In[77]:


import matplotlib
matplotlib.use('Agg')
get_ipython().magic(u'pylab inline')

import os
import math
import pandas
import datetime as dt
from time import time
import numpy as np
import pandas as pd

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.dataset.transformer import *

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from pyspark import SparkContext

sc.stop
sc=SparkContext.getOrCreate(conf=create_spark_conf().setMaster("local[4]").set("spark.driver.memory","64g"))

init_engine()


# In[78]:


import os

cwd = os.getcwd()
print cwd

os.chdir("/notebooks")


cwd = os.getcwd()
print cwd
print os.listdir(os.path.join(cwd, 'data'))


# In[79]:


import numpy as np
import time


# In[80]:


much_data = np.load('./data/muchdata-32-32-32.npy', encoding = 'latin1')


# In[81]:


print len(much_data)
print len(much_data[0][0])
print len(much_data[0][1])


# In[82]:


train_data = much_data[:16]
validation_data = much_data[16:24]


# In[86]:


# creat the RDD for model training now
prep_train_rdd = sc.parallelize(train_data,4)
prep_test_rdd = sc.parallelize(validation_data,4)

print prep_train_rdd.take(1)[0][0].transpose().shape

train_rdd = prep_train_rdd.map(lambda s : Sample.from_ndarray(np.array(s[0]).reshape(1,32,32,32), np.array(s[1][1]+1)))
test_rdd = prep_test_rdd.map(lambda s : Sample.from_ndarray(np.array(s[0]).reshape(1,32,32,32), np.array(s[1][1]+1)))


# In[56]:


train_rdd.take(1)


# In[ ]:


# Create a model simpler than LeNet model to test the data pipeline and layer setup
# here the input number to the full connected Linear layer is
# based on the 3D Max pool
# 6*(32/2)*(32/2)*(32/2) = 24576

def build_model(class_num):
    model = Sequential()
    model.add(Reshape([1,32,32,32]))
    model.add(VolumetricConvolution(3, 6, 5, 5, 5).set_name('3Dconv1'))
    model.add(Tanh())
    model.add(Echo())
    model.add(VolumetricMaxPooling(2, 2, 2,                                    1, 1, 1,                                    0, 0, 0).set_name('3Dpool1'))
    model.add(Echo())
    model.add(Reshape([6 * 16 * 16 * 16]))
    model.add(Linear(6 * 16 * 16 * 16, 128).set_name('fc1'))
    model.add(Tanh())
    model.add(Echo())
    model.add(Linear(128, class_num).set_name('score'))
    model.add(LogSoftMax())
    return model
lenet_model = build_model(2)


# In[57]:


# Create a model simpler than LeNet model to test the data pipeline

def build_model(class_num):
    model = Sequential()
    model.add(Reshape([1,32,32,32]))
    model.add(VolumetricConvolution(3, 6, 5, 5, 5).set_name('3Dconv1'))
    model.add(Tanh())
    model.add(VolumetricMaxPooling(2, 2, 2,                                    1, 1, 1,                                    0, 0, 0).set_name('3Dpool1'))
    model.add(Reshape([81 *18 *18 * 18]))
    model.add(Linear(81 * 18 * 18 * 18, 16).set_name('fc1'))
    model.add(Tanh())
    model.add(Linear(16, class_num).set_name('score'))
    model.add(LogSoftMax())
    return model
lenet_model = build_model(2)


# In[64]:


# Create an Optimizer

optimizer = Optimizer(
    model=lenet_model,
    training_rdd=train_rdd,
    criterion=ClassNLLCriterion(),
    optim_method=SGD(learningrate=0.4, learningrate_decay=0.0002),
    end_trigger=MaxEpoch(3),
    batch_size=8)

# Set the validation logic
optimizer.set_validation(
    batch_size=8,
    val_rdd=test_rdd,
    trigger=EveryEpoch(),
    val_method=[Top1Accuracy()]
)

app_name='lenet-'+dt.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary = TrainSummary(log_dir='/tmp/bigdl_summaries',
                                     app_name=app_name)
train_summary.set_summary_trigger("Parameters", SeveralIteration(50))
val_summary = ValidationSummary(log_dir='/tmp/bigdl_summaries',
                                        app_name=app_name)
optimizer.set_train_summary(train_summary)
optimizer.set_val_summary(val_summary)
print "saving logs to ",app_name


# In[65]:


get_ipython().run_cell_magic(u'time', u'', u'start = time()\n# Boot training process\ntrained_model = optimizer.optimize()\nprint "Optimization Done in {} seconds.".format(time()-start)')


# In[ ]:


# original net: Create a LeNet model
"""
def build_model(class_num):
    model = Sequential()
    model.add(Reshape([1,32,32,32]))
    model.add(VolumetricConvolution(3, 6, 5, 5, 5).set_name('3Dconv1'))
    model.add(Tanh())
    model.add(VolumetricMaxPooling(2, 2, 2, \
                                   1, 1, 1, \
                                   0, 0, 0).set_name('3Dpool1'))
    model.add(Tanh())
    model.add(VolumetricConvolution(6, 12, 5, 5, 5).set_name('3Dconv2'))
    model.add(VolumetricMaxPooling(2, 2, 2, \
                                   1, 1, 1, \
                                   0, 0, 0).set_name('3Dpool2'))
    model.add(Reshape([12 * 22 * 22 * 22]))
    model.add(Linear(12 * 22 * 22 * 22, 16).set_name('fc1'))
    model.add(Tanh())
    model.add(Linear(16, class_num).set_name('score'))
    model.add(LogSoftMax())
    return model
lenet_model = build_model(2)
"""


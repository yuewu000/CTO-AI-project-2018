
# coding: utf-8

import tensorflow as tf
import numpy as np
import time

IMG_SIZE_PX = 100
SLICE_COUNT = 40

n_classes = 2
batch_size = 10

hm_epochs = 50

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

keep_rates = [0.8, 0.7, 0.6, 0.5]

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')
def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
def avgpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.avg_pool3d (x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x, USE_DROPOUT = True):
    #                # 3 x 3 x 3 patches, 1 channel, 64 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,64])),
               #       3 x 3 x 3 patches, 64 channels, 128 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,64,128])),
               #                          128 channels,  256 features
               'W_conv3a':tf.Variable(tf.random_normal([3,3,3,128,256])),
               #       3 x 3 x 3 patches, 256 channels, 256 features to compute.
               'W_conv3b':tf.Variable(tf.random_normal([3,3,3,256,256])),
                              #          256 channles, 512 features
               'W_conv4a':tf.Variable(tf.random_normal([3,3,3,256,512])),
               #       3 x 3 x 3 patches, 512 channels, 512 features to compute.
               'W_conv4b':tf.Variable(tf.random_normal([3,3,3,512,512])),
                #          512 channles, 1024 features
               'W_conv5a':tf.Variable(tf.random_normal([3,3,3,512,1024])),
               #       3 x 3 x 3 patches, 1024 channels, 1024 features to compute.
               'W_conv5b':tf.Variable(tf.random_normal([3,3,3,1024,1024])),
               'W_fc':tf.Variable(tf.random_normal([4096,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([64])),
               'b_conv2':tf.Variable(tf.random_normal([128])),
               'b_conv3a':tf.Variable(tf.random_normal([256])),
               'b_conv3b':tf.Variable(tf.random_normal([256])),
               'b_conv4a':tf.Variable(tf.random_normal([512])),
               'b_conv4b':tf.Variable(tf.random_normal([512])),
               'b_conv5a':tf.Variable(tf.random_normal([1024])),
               'b_conv5b':tf.Variable(tf.random_normal([1024])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}
    """
    keep_rates = {'kr_conv2':tf.placeholder(tf.float32),
                  'kr_conv3':tf.placeholder(tf.float32),
                  'kr_conv4':tf.placeholder(tf.float32),                
                }
    """	

    #                            image X      image Y        image Z
    print (keep_rates)
    
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])
    conv1 = avgpool3d(x)
    conv1 = tf.nn.relu(conv3d(conv1, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)
    #conv1 = tf.nn.dropout(conv1, keep_rates['kr_conv1'])
    
    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)
    if USE_DROPOUT:
        conv2 = tf.nn.dropout(conv2, keep_rates[1])
    
    conv3 = tf.nn.relu(conv3d(conv2, weights['W_conv3a']) + biases['b_conv3a'])
    conv3 = tf.nn.relu(conv3d(conv3, weights['W_conv3b']) + biases['b_conv3b'])
    conv3 = maxpool3d(conv3)
    if USE_DROPOUT:
        conv3 = tf.nn.dropout(conv3, keep_rates[2])
    
    conv4 = tf.nn.relu(conv3d(conv3, weights['W_conv4a']) + biases['b_conv4a'])
    conv4 = tf.nn.relu(conv3d(conv4, weights['W_conv4b']) + biases['b_conv4b'])
    conv4 = maxpool3d(conv4)
    if USE_DROPOUT:
        #conv4 = tf.nn.dropout(conv4, keep_rates['kr_conv4'])
        conv4 = tf.nn.dropout(conv4, keep_rates[3])
    
    conv5 = tf.nn.relu(conv3d(conv4, weights['W_conv5a']) + biases['b_conv5a'])
    conv5 = tf.nn.relu(conv3d(conv5, weights['W_conv5b']) + biases['b_conv5b'])
    conv5 = maxpool3d(conv5)
    if USE_DROPOUT:
        #conv4 = tf.nn.dropout(conv4, keep_rates['kr_conv4'])
        conv5 = tf.nn.dropout(conv5, keep_rates[3])
	    
 
    fc = tf.reshape(conv5,[-1, 4096])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rates[0])

    output = tf.matmul(fc, weights['out'])+biases['out']
    
    print ('the current prediction is:', output )
    
    return output

much_data = np.load('./data/muchdata-' + str(IMG_SIZE_PX) + '-' + str(IMG_SIZE_PX) + '-' + str(SLICE_COUNT) + '.npy')
# If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
much_data.shape

train_data = much_data[:1000]
validation_data = much_data[1000:1300]


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    
    #change the cross entropy function to UDF to avoid NAN
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    cross_entropy = -tf.reduce_sum(y*tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    cost = tf.reduce_mean( cross_entropy )

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cost)
    
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    # define tf saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        successful_runs = 0
        total_runs = 0
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            #predp = []
            time_start = time.time()
            for data in train_data:
                total_runs += 1
                try:
                    #epoch_loss = 0
                    X = data[0]
                    Y = data[1]
                    #print(Y)
                    """
                    _, c, pred = sess.run([optimizer, cost, prediction], feed_dict={x: X, y: Y, 
                        keep_rates['kr_conv2']:[RATES_KEEP[0]], keep_rates['kr_conv3']:[RATES_KEEP[1]], keep_rates['kr_conv3']:[RATES_KEEP[2]]})
                    """
                    
                    _, c = sess.run([optimizer, cost, ], feed_dict={x: X, y: Y})
                    #outputNow = sess.run([prediction], feed_dict={x: X, y: Y})
                    
                    #predp.append(pred)
                    epoch_loss += c
                    successful_run += 1
                except Exception as e:
                    pass
                    #print(str(e))
                #print ("current prediction is {} of length {}".format(outputNow, len(outputNow)))
            
            print ("run time for epoch {} is {} seconds".format(epoch+1, time.time()-time_start))
            
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            print('Accuracy:',sess.run(accuracy, feed_dict = {x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
            #print('Accuracy:',accuracy.eval(feed_dict = {x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
            
        print('Done. Finishing accuracy:')
        print('Accuracy:',sess.run(accuracy, feed_dict = {x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        
        print('fitment percent:',successful_runs/total_runs)
        save_path = saver.save(sess, "./model/model_18layer_100x100x40_t1e{}.tfm".format(hm_epochs))
        print("Model saved in file: %s" % save_path)

train_neural_network(x)



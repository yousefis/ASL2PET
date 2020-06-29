# import tensorflow as tf
#
# tf.reset_default_graph()
#
# a = tf.Variable(10.0)
# b = tf.Variable(10.0)
# switch = tf.placeholder(tf.bool)
# res = tf.cond(switch, lambda: (2.0* a), lambda: tf.square(b))
# opt = tf.train.GradientDescentOptimizer(0.05)
# grads = opt.compute_gradients(res)
# train = opt.apply_gradients(grads)
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(grads, {switch: True}))
# ##################################################
# ## {}
# ##################################################
# ## {License_info}
# ##################################################
# ## Author: {Sahar Yousefi}
# ## Copyright: Copyright {2020}, {LUMC}
# ## Credits: [Sahar Yousefi]
# ## License: {GPL}
# ## Version: 1.0.0
# ## Mmaintainer: {Sahar Yousefi}
# ## Email: {s.yousefi.radi[at]lumc.nl}
# ## Status: {Research}
# ##################################################
import numpy as np
import  SimpleITK as sitk
from functions.reader.data_reader_hybrid import _read_data
if __name__=="__main__":
    # data_path="/exports/lkeb-hpc/syousefi/Data/asl_pet/"
    '''read path of the images for train, test, and validation'''
    data_path_AMUC = "/exports/lkeb-hpc/syousefi/Data/asl_pet/"
    data_path_LUMC = "/exports/lkeb-hpc/syousefi/Data/asl_pet2/"
    _rd = _read_data(data_path_AMUC,data_path_LUMC)
    train_data, validation_data, test_data=_rd.read_data_path(35)
    m_t1=.00000000000000000000001
    M_t1=1
    for d in train_data:
        print(d['asl'])
        t1=sitk.GetArrayFromImage(sitk.ReadImage(d['asl']))
        mt1= np.min(t1)
        Mt1= np.max(t1)
        if mt1<m_t1:
            m_t1=mt1
        if Mt1>M_t1:
            M_t1=Mt1
        print('['+str(mt1)+' , '+str(Mt1)+']')
    print('['+str(m_t1)+' , '+str(M_t1)+']')


# import tensorflow as tf
# #
# x = tf.constant(5.0)
# with tf.GradientTape() as tape:
#     tape.watch(x)
#     y = x**3
# with tf.GradientTape() as tape:
#     tape.watch(x)
#     y += x**1
# sess=tf.Session()
# res=sess.run(tape.gradient(y, x))
# print(res) # -> 75.0


# x = tf.constant(5.0)
# with tf.GradientTape() as t:
#     t.watch(x)
#     loss = x**3
# with tf.GradientTape() as t:
#     loss += x**5
# # t.gradient(loss, x)  # Only differentiates other_loss_fn, not loss_fn
# sess=tf.Session()
# res=sess.run(t.gradient(loss, x))
# print(res)

# # The following is equivalent to the above
# with tf.GradientTape() as t:
#   loss = loss_fn()
#   t.reset()
#   loss += other_loss_fn()
# t.gradient(loss, ...)  # Only differentiates other_loss_fn, not loss_fn
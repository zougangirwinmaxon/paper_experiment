import tensorflow as tf
import numpy as np
np.random.seed(1)
tf.set_random_seed(1)  # reproducible
#定义超参数设置
MAX_EPISODES = 500
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic

GAMMA = 0.5  # optimal reward discount######################0.01 0.01 0.001 0.001 0.5 0.5

TAU = 0.01  # soft replacement 软更新参数一般取0.001 ###########
VAR_MIN = 0.1#最小的噪声方差
MEMORY_CAPACITY =  1500#300     #1500               #########1500 1000 1000 1500 1000 1500
BATCH_SIZE =64 #32   #128
OUTPUT_GRAPH = False

#定义ddpg智能体
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # memory里存放当前和下一个state，动作和奖励
        self.pointer = 0
        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')  # 输入
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.A = tf.placeholder(tf.float32, [None, a_dim], 'a_input')
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_eval = self._build_c(self.S, self.A, scope='eval', trainable=True, reuse=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)

        #定义两个目标网络的参数更新
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        #定义训练评估网络
        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q_eval)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        #定义训练策略网络
        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        #初始化所有全局变量
        self.sess.run(tf.global_variables_initializer())
        # 初始化Saver对象
        self.saver = tf.train.Saver()
        # #定义记录训练图
        # if OUTPUT_GRAPH:
        #     tf.summary.FileWriter("logs/", self.sess.graph)

    #定义选择的动作
    def choose_action(self, s):
        temp = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        # print(temp[0])
        return temp[0]

    #定义学习过程
    def learn(self):
        self.sess.run(self.soft_replace)#软更新策略目标网路和评估目标网络
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.A: ba, self.R: br, self.S_: bs_})

    #定义经验回放池
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY #用新的经验替换旧的经验
        self.memory[index, :] = transition
        self.pointer += 1

    #定义动作网络结构
    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[1], name='scaled_a')

    #定义评估网络结构
    def _build_c(self, s, a, scope, trainable, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            n_l1 = 400
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

# 保存模型的函数
    def save_model(self, path='ddpg_model/model.ckpt'):
        self.saver.save(self.sess, path)
        print("Model saved in path: %s" % path)

    # 加载模型的函数
    def load_model(self, path='ddpg_model/model.ckpt'):
        self.saver.restore(self.sess, path)
        print("Model restored from path: %s" % path)
import numpy as np
from env import UAVEnv
np.random.seed(1)
np.set_printoptions(suppress=True)  # 禁止科学计数法
#定义标准化状态空间
class StateNormalization(object):
    def __init__(self):
        env = UAVEnv()
        M = env.M
        """任务大小范围：2097153, 2621440
            终端电量范围：0，2
            无人机电量范围：0，1
            终端位置长宽范围：0，10000
            无人机位置长宽范围：0，10000
            遮挡标记范围：0，1
            截止时间范围：0，0.5
        """
        self.high_state=np.append(np.ones(M) * 2621440,np.ones(M) * 1)#任务最大值，终端电量
        self.high_state=np.append(self.high_state,8)#无人机最大电池容量
        self.high_state=np.append(self.high_state,np.ones(M) * env.ground_length)#终端位置的最大长度
        self.high_state=np.append(self.high_state,np.ones(M) * env.ground_width)#终端位置的最大宽度
        self.high_state=np.append(self.high_state,env.ground_length)#无人机位置的最大长度
        self.high_state=np.append(self.high_state, env.ground_width)#无人机位置的最大宽度
        self.high_state = np.append(self.high_state, np.ones(M))#遮挡标记
        self.high_state = np.append(self.high_state, np.ones(M)*3)#任务截止时间

        self.low_state = np.append(np.ones(M)*2097153, np.ones(M)*0)  # 任务最小值和终端电量，假设最小值为0
        self.low_state = np.append(self.low_state, 0)  # 无人机最小电池容量，假设最小值为0
        self.low_state = np.append(self.low_state, np.zeros(M))  # 终端位置的最小长度
        self.low_state = np.append(self.low_state, np.zeros(M))  # 终端位置的最小宽度
        self.low_state = np.append(self.low_state, 0)  # 无人机位置的最小长度，假设最小值为0
        self.low_state = np.append(self.low_state, 0)  # 无人机位置的最小宽度，假设最小值为0
        self.low_state = np.append(self.low_state, np.zeros(M))  # 遮挡标记，假设最小值为0
        self.low_state = np.append(self.low_state, np.ones(M)*1)  # 任务截止时间的最小值，假设最小值为0

    def state_normal(self, state):
        denom = self.high_state - self.low_state
        normalized = (state - self.low_state) / (denom + 1e-8)
        return np.clip(normalized, 0.0, 1.0)

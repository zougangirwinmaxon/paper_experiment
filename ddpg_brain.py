import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(1)
torch.manual_seed(1)

# 定义超参数设置
MAX_EPISODES = 500
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.5  # optimal reward discount
TAU = 0.01  # soft replacement
VAR_MIN = 0.1  # 最小的噪声方差
MEMORY_CAPACITY = 1500
BATCH_SIZE = 64


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 400),
            nn.ReLU6(),
            nn.Linear(400, 300),
            nn.ReLU6(),
            nn.Linear(300, 10),
            nn.ReLU(),
            nn.Linear(10, a_dim),
            nn.Tanh(),
        )
        self.a_bound = float(a_bound[1])

    def forward(self, s):
        return self.net(s) * self.a_bound


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.w1_s = nn.Linear(s_dim, 400, bias=False)
        self.w1_a = nn.Linear(a_dim, 400, bias=False)
        self.b1 = nn.Parameter(torch.zeros(1, 400))
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 10)
        self.out = nn.Linear(10, 1)

    def forward(self, s, a):
        net = torch.relu6(self.w1_s(s) + self.w1_a(a) + self.b1)
        net = torch.relu6(self.l2(net))
        net = torch.relu(self.l3(net))
        return self.out(net)


# 定义ddpg智能体
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor_eval = Actor(s_dim, a_dim, a_bound).to(self.device)
        self.actor_target = Actor(s_dim, a_dim, a_bound).to(self.device)
        self.critic_eval = Critic(s_dim, a_dim).to(self.device)
        self.critic_target = Critic(s_dim, a_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())

        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=LR_A)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=LR_C)
        self.mse_loss = nn.MSELoss()

    # 定义选择的动作
    def choose_action(self, s):
        self.actor_eval.eval()
        with torch.no_grad():
            s_tensor = torch.tensor(s[np.newaxis, :], dtype=torch.float32, device=self.device)
            a = self.actor_eval(s_tensor).cpu().numpy()[0]
        self.actor_eval.train()
        return a

    def _soft_update(self, target_net, eval_net):
        for target_param, eval_param in zip(target_net.parameters(), eval_net.parameters()):
            target_param.data.copy_((1 - TAU) * target_param.data + TAU * eval_param.data)

    # 定义学习过程
    def learn(self):
        self._soft_update(self.actor_target, self.actor_eval)
        self._soft_update(self.critic_target, self.critic_eval)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        bs_t = torch.tensor(bs, dtype=torch.float32, device=self.device)
        ba_t = torch.tensor(ba, dtype=torch.float32, device=self.device)
        br_t = torch.tensor(br, dtype=torch.float32, device=self.device)
        bs_next_t = torch.tensor(bs_, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            a_next = self.actor_target(bs_next_t)
            q_next = self.critic_target(bs_next_t, a_next)
            q_target = br_t + GAMMA * q_next

        q_eval = self.critic_eval(bs_t, ba_t)
        critic_loss = self.mse_loss(q_eval, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        a_pred = self.actor_eval(bs_t)
        actor_loss = -self.critic_eval(bs_t, a_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    # 定义经验回放池
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    # 保存模型的函数
    def save_model(self, path='ddpg_model/model.pth'):
        payload = {
            'actor_eval': self.actor_eval.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_eval': self.critic_eval.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'pointer': self.pointer,
        }
        torch.save(payload, path)
        print("Model saved in path: %s" % path)

    # 加载模型的函数
    def load_model(self, path='ddpg_model/model.pth'):
        payload = torch.load(path, map_location=self.device)
        self.actor_eval.load_state_dict(payload['actor_eval'])
        self.actor_target.load_state_dict(payload['actor_target'])
        self.critic_eval.load_state_dict(payload['critic_eval'])
        self.critic_target.load_state_dict(payload['critic_target'])
        self.actor_optimizer.load_state_dict(payload['actor_optimizer'])
        self.critic_optimizer.load_state_dict(payload['critic_optimizer'])
        self.pointer = payload.get('pointer', self.pointer)
        print("Model restored from path: %s" % path)

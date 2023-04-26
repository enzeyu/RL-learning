import numpy as np
import matplotlib.pyplot as plt

class  BernoulliBandit:
    """ 伯努利多臂老虎机,输入K表示拉杆个数 """
    # k个拉杆,每个拉杆获奖概率为probs,最大的拉杆索引,最大的拉杆概率
    def __init__(self,K):
        self.probs = np.random.uniform(size=K) # 随机生成K个0～1的数,作为拉动每根拉杆的获奖概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆索引
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.K = K
    # 每拉一次的结果
    def step(self,k):
        # 当玩家选择了k号拉杆后, 根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

# 玩多臂老虎机的agent
class Solver:
    """ 多臂老虎机算法框架 """
    # 老虎机, 每根拉杆次数, agent的累计懊悔, 动作记录, 懊悔记录
    def __init__(self,bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # 每根拉杆的尝试次数
        self.regret = 0.  # 当前步的累积懊悔
        self.actions = []  # 列表记录每一步的动作
        self.regrets = []  # 列表记录每一步的累积懊悔
    def update_regret(self,k):
        # 计算累计懊悔并保存，k是本次动作选择拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k] # 计算懊悔
        self.regrets.append(self.regret)
    def run_one_step(self):
        # 返回当前动作选择哪一个拉杆，由具体的策略实现
        raise NotImplementedError
    def run(self,num_steps):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps): # 运行一步都要更新regret，记录动作，更新每根拉杆的尝试次数
            k = self.run_one_step() # 玩一次，返回动作
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

# 策略1：psilon贪婪算法
# EpsilonGreedy的父类是Solver
class EpsilonGreedy(Solver):
    """ epsilon贪婪算法,继承Solver类 """
    # epsilon为贪婪算法的参数
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        # 调用Solver的初始化方法
        super(EpsilonGreedy,self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化拉动所有杠杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)
    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0,self.bandit.K) # 随机拉一个
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k) # 得到本次动作的奖励1或0
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k]) # 2.2.4节证明了
        return k

def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

"""
随机生成K臂老虎机
"""
np.random.seed(1) # 设定随机种子，让实验可重复
K=10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
      (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

"""
epsilon-贪婪算法的累积懊悔
"""
np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

"""
different results with different epsilons
"""
np.random.seed(0)
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
# Different agent with different epsilons
epsilon_greedy_solver_list = [
    EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
]
epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)
plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

# 策略2：epsilon衰减贪婪算法
# DecayingEpsilongreedy的父类是Solver
class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类 """
    def __init__(self,bandit,init_prob=1.0):
        super(DecayingEpsilonGreedy,self).__init__(bandit)
        self.total_time = 0
        self.estimates = np.array([init_prob]*self.bandit.K)
    def run_one_step(self):
        self.total_time += 1
        if np.random.rand() < 1/self.total_time:
            k = np.random.randint(0,self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

"""
decaying epsilon-贪婪算法的累积懊悔
"""
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

# 策略3：UCB算法
# UCB的父类是Solver
class UCB(Solver):
    """ UCB算法,继承Solver类 """
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef
    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k
"""
UCB算法的累积懊悔
"""
np.random.seed(1)
coef = 1  # 控制不确定性比重的系数
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])

# 策略4：汤普森采样算法
# ThompsonSampling父类是Solver
class ThompsonSampling(Solver):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)
        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k

np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ["ThompsonSampling"])
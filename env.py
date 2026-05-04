import math
import numpy as np
np.random.seed(1)
np.set_printoptions(suppress=True)  # 禁止科学计数法


'''基本单位
长度：米
能量：焦耳
噪声功率：分贝毫瓦（dBm）
带宽：赫兹
时间：秒
数据量：比特位（bite）
功率：瓦特
'''
# 定义无人机的环境类
class UAVEnv(object):
    # 环境参数
    ground_length = ground_width = 100  # 场地长宽均为100m Deep reinforcement learning based computation offloadingfor xURLLC services with UAV-assisted IoT-based multi-access edgecomputing system
    B = 10 * 10 ** 6  # 10 MHz，终端和无人机之间的带宽 Intelligent Delay-Aware Partial Computing TaskOffloading for Multiuser Industrial Internet ofThings Through Edge Computing
    B2 = 10 * 10 ** 6  # 10 MHz，无人机和卫星之间的带宽 Deep_Reinforcement_Learning_for_Delay-Oriented_IoT_Task_Scheduling_in_SAGIN
    p_noisy_los = 10 ** (-13)  # 噪声功率-100dB，可视信道的噪声功率
    p_noisy_nlos = 10 ** (-11)  # 噪声功率-80dB 非可视信道的噪声功率
    r = 10 ** (-27)  # 芯片结构对cpu处理的影响因子
    s = 1000  # 单位bit处理所需cpu圈数1000
    alpha0 = 1e-3  # 距离为1m时的参考信道增益-30dB = 0.001， -50dB = 1e-5
    T = 1800  # 一个回合1800s
    t_fly = 1  # 一个时间步内飞行时间
    t_com = 11  # 一个时间步内的计算时间
    delta_t = t_fly + t_com  # 1s飞行, 后11s用于悬停计算
    slot_num = int(T / delta_t)  # 一个回合总的时间除以一个时间片得到时间片的个数

    # 终端的信息
    M = 5 # 终端的数量为M个
    block_flag_list = np.random.randint(0, 2, M)  # M个终端的遮挡情况
    loc_ue_list = np.random.randint(0, 101, size=[M, 2])  # 终端位置信息:x和y坐标在0-100m随机
    task_list = np.random.randint(2097153, 2621441, M)  # 终端每个时间生成随机计算任务的大小数据量约0.25MB，0.3125MB
    ue_battery_list = np.random.uniform(1, 2, M).astype(float)#每个终端的初始电量
    task_deadline = np.random.uniform(1, 3,M)#每个任务的截至时间
    # v_ue = 0  # 终端设备的移动速度
    # ue_id = 0  # 终端设备的标号
    p_uplink_max = 0.2  # 终端的最大上传功率 毫瓦特

    # 无人机的信息
    #无人机的初始电量确实会对收敛有巨大影响
    e_battery_uav = 8  # uav电池电量: 100J. ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning
    # m_uav = 9.65  # uav质量/kg
    height = 100  # UAV飞行高度是100m
    loc_uav = [50.0, 50.0]  # 无人机的初始位置在场地的正中间
    flight_speed = 4.  # 无人机飞行速度4m/s
    f_uav = 1e9  # UAV的计算频率500MHz,1GHz Deep_Reinforcement_Learning_for_Delay-Oriented_IoT_Task_Scheduling_in_SAGIN
    UAV_satellite_noise_p = 10 ** (-14)  # 无人机和卫星之间的噪声功率
    g_UAV_satellite = 10 ** (-12)  # 无人机和卫星之间的信道增益
    UAV_uplink_max = 5  # 无人机最大上传功率 5瓦特 Deep_Reinforcement_Learning_for_Delay-Oriented_IoT_Task_Scheduling_in_SAGIN
    uav_solar_recharge_rate= 0.01#无人机收集的能量0.01-0.08之间变化,终端收集的10倍 Dynamic_Edge_Computation_Offloading_for_Internet_of_Things_With_Energy_Harvesting_A_Learning_Method

    # 卫星的信息
    f_satellite = 5e9  # 卫星的计算频率5GHz Deep_Reinforcement_Learning_for_Delay-Oriented_IoT_Task_Scheduling_in_SAGIN
    propagation_delay = 0.00644  # 传播时延：0.322秒 Deep_Reinforcement_Learning_for_Delay-Oriented_IoT_Task_Scheduling_in_SAGIN

    action_bound = [-1, 1]  # 对应tanh激活函数
    action_dim = 2 * M + 1  # 动作维度
    state_dim = 6 * M + 3  # 状态维度

    # 定义环境初始化
    def __init__(self):
        """任务大小范围：2097153, 2621440
            终端电量范围：0，2
            无人机电量范围：0，1
            终端位置长宽范围：0，10000
            无人机位置长宽范围：0，10000
            遮挡标记范围：0，1
            截止时间范围：0，0.5
            无人机和卫星的计算资源
        """
        # 初始状态：
        self.start_state = np.append(self.task_list, self.ue_battery_list)  # 1.各个终端设备的任务大小,2.各个终端设备剩余的电量
        self.start_state = np.append(self.start_state, self.e_battery_uav)  # 3.无人机剩余的电量
        self.start_state = np.append(self.start_state, np.ravel(self.loc_ue_list))  # 4.终端的位置
        self.start_state = np.append(self.start_state, self.loc_uav)  # 5.无人机的初始位置
        self.start_state = np.append(self.start_state, self.block_flag_list)  # 6.遮挡标记
        self.start_state = np.append(self.start_state, self.task_deadline)  # 7.任务截至时间
        self.state = self.start_state

    # 重新定义重置  无人机电量，无人机位置，终端位置，终端任务，遮挡情况，无人机和卫星的计算资源
    def reset_env(self):
        self.task_list = np.random.randint(2097153,2621441, self.M)  # 生成每个终端的任务
        self.ue_battery_list = np.full_like(self.ue_battery_list, 1)#生成每个终端的电量
        self.e_battery_uav = 8  # uav电池容量: 100J
        self.loc_ue_list = np.random.randint(0, 101, size=[self.M, 2])  # 终端位置信息:x在0-100随机
        self.loc_uav = [50.0, 50.0]  # 无人机初始位置
        self.block_flag_list = np.random.randint(0, 2, self.M)  # 每个终端的遮挡情况
        self.task_deadline = np.random.uniform(1, 3, self.M)#所有终端任务的截止时间 单位秒
        self.f_uav=1e9#无人机的计算资源
        self.f_satellite=5e9#卫星的计算资源

    # 定义终端任务，截止时间情况
    def reset_ue_step(self):
        self.task_list = np.random.randint(2097153,2621441,  self.M)  # 生成每个终端的任务
        # self.loc_ue_list = np.random.randint(0, 10001, size=[self.M, 2])  # 位置信息:x在0-100随机
        self.block_flag_list = np.random.randint(0, 2, self.M)  # 每个终端的遮挡情况
        self.task_deadline = np.random.uniform(1, 3, self.M)#所有终端任务的截止时间

    # 重置环境
    def reset(self):
        self.reset_env()
        self.state = np.append(self.task_list, self.ue_battery_list)  # 1.各个终端设备的任务大小,2.各个终端设备剩余的电量
        self.state = np.append(self.state, self.e_battery_uav)  # 3.无人机剩余的电量
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))  # 4.终端的位置
        self.state = np.append(self.state, self.loc_uav)  # 5.无人机的位置
        self.state = np.append(self.state, self.block_flag_list)  # 6.遮挡标记
        self.state = np.append(self.state, self.task_deadline)  # 7.任务的截止时间
        return self.state

    # 定义取得状态方法
    def _get_obs(self):
        self.state = np.append(self.task_list, self.ue_battery_list)  # 1.各个终端设备的任务大小,2.各个终端设备剩余的电量
        self.state = np.append(self.state, self.e_battery_uav)  # 3.无人机剩余的电量
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))  # 4.终端的位置
        self.state = np.append(self.state, self.loc_uav)  # 5.无人机的位置
        self.state = np.append(self.state, self.block_flag_list)  # 6.遮挡标记
        self.state = np.append(self.state, self.task_deadline)  # 7.任务截止时间
        return self.state

    # 定义执行动作后环境的改变
    def step(self, action):
####################################################################################################################
        # 将动作值从 [-1, 1] 转换到 [0, 1]
        action = (action + 1) / 2
        # print(f"[调试] 归一化后的动作值: {[f'{a:.15f}' for a in action]}")

        # 提取卸载决策
        offloading_decision = [self.offloading_the_destination(element) for element in action[self.M + 1:2 * self.M + 1]]
        print(f"[调试 输出的动作] 卸载决策: {offloading_decision}")

        # 提取终端上传功率，若不卸载（offloading_decision == 0）则功率为0
        p_uplink = [
            max(element * self.p_uplink_max, 0.02) if offloading_decision[i] >= 0.1 else 0
            for i, element in enumerate(action[:self.M])
        ]
        print(f"[调试 输出的动作] 终端上传功率: {[f'{p:.15f}' for p in p_uplink]}W")

        # 提取无人机功率并限制范围
        UAV_uplink = max(action[self.M] * self.UAV_uplink_max, 0.2)

        # 如果所有终端都未选择卸载到卫星，将无人机功率设为 0
        if all(decision != 2 for decision in offloading_decision):
            UAV_uplink = 0
        print(f"[调试 输出的动作] 无人机上传功率: {UAV_uplink:.15f}W")

        # print("[调试] 处理后的动作分量: 终端上传功率:", [f"{p:.15f}" for p in p_uplink],
        #       "无人机功率:", f"{UAV_uplink:.15f}", "卸载决策:", offloading_decision)
###################################################################################################################
        done = False
        success = 0
        R_func = 0
        UAV_energy_consume_step = 0
        ue_total_energy_consume_step = 0
        offloading_uav_success=0
        offloading_satellite_success=0
        for ue_id in range(self.M):
            print(f"[调试] 正在处理终端 {ue_id}")
            if offloading_decision[ue_id] == 1:  # 卸载到无人机
                # 终端的传输时间
                t_ue_chuanshu = self.zhong_duan_de_chuan_shu_yan_chi(ue_id, self.task_list, self.loc_ue_list, self.loc_uav,p_uplink, self.block_flag_list)
                print(f"终端需要的传输时间{t_ue_chuanshu}秒，")
                # 终端的传输能耗
                e_ue = self.zhongduandechuanshunenghao(ue_id, p_uplink, t_ue_chuanshu)
                print(f"终端需要的传输能耗{e_ue}，焦耳")
                # 剩余计算时间
                remaining_time = self.task_deadline[ue_id] - t_ue_chuanshu
                print(f"剩余需要的计算时间{remaining_time}秒")
                # 需要的计算资源
                required_resources = self.calculate_required_resources(ue_id, remaining_time,self.task_list)  # 所需的计算资源=任务大小/剩余时间
                print(f"无人机需要的计算资源{required_resources}周期")
                # 无人机的计算能耗
                e_uav_comp = self.wurenjidejisuannenghao(ue_id, required_resources)
                print(f"无人机需要的计算能耗{e_uav_comp}焦耳")
                if e_ue>self.ue_battery_list[ue_id]:
                    print("终端传输能量不足")
                    success += 0  # 任务没成功
                    R_func +=(-0.3)   # 奖励函数为
                    continue
                else:
                    if required_resources>self.f_uav or required_resources<0:#如果无人机计算资源不足
                        print("计算资源无法满足")
                        success += 0#任务没成功
                        R_func += (-0.1)#奖励函数为
                        continue
                    else:
                        if e_uav_comp>self.e_battery_uav:
                            print("无人机计算能量不足")
                            success += 0  # 任务没成功
                            R_func += (-0.3)  # 奖励函数为
                            continue
                        else:
                            print("任务卸载到无人机成功")
                            # 如果终端电量、无人机电量、无人机计算资源都满足
                            # 更新状态和资源消耗
                            self.ue_battery_list[ue_id] -= e_ue#终端能耗
                            ue_total_energy_consume_step += e_ue  # 用来统计终端的能耗
                            self.e_battery_uav -= e_uav_comp#无人机能耗
                            UAV_energy_consume_step += e_uav_comp  # 用来统计无人机的总能耗
                            self.f_uav -= required_resources  # 无人机消耗的计算资源
                            success += 1
                            offloading_uav_success+=1
                            R_func += (1-(1e-1)*(ue_total_energy_consume_step+UAV_energy_consume_step))
                            # R_func += (1)
            elif offloading_decision[ue_id] == 2:  # 卸载到卫星
                # 传播时间
                t_propagation = self.propagation_delay
                print(f"需要的传播时间为{t_propagation}秒")
                # 终端的传输时间
                t_ue_chuanshu = self.zhong_duan_de_chuan_shu_yan_chi(ue_id, self.task_list, self.loc_ue_list, self.loc_uav,p_uplink, self.block_flag_list)
                print(f"终端需要的传输时间{t_ue_chuanshu}秒")
                # 无人机的传输时间
                t_uav_chuanshu = self.wurenjidechuanshuyanchi(ue_id, self.task_list, UAV_uplink)
                print(f"无人机需要的传输时间{t_uav_chuanshu}秒")
                # 剩余计算时间
                remaining_time = self.task_deadline[ue_id] - t_ue_chuanshu - t_uav_chuanshu - t_propagation
                print(f"剩余需要的计算时间{remaining_time}秒")
                # 需要的计算资源
                required_resources = self.calculate_required_resources(ue_id, remaining_time,self.task_list)  # 所需的计算资源=任务大小/剩余时间
                print(f"卫星需要的计算资源{required_resources}周期")
                # 终端的传输能耗
                e_ue = self.zhongduandechuanshunenghao(ue_id, p_uplink, t_ue_chuanshu)
                print(f"终端需要的传输能耗{e_ue}焦耳")
                # 无人机的传输能耗
                e_uav_chuanshu = self.wurenjidechuanshunenghao(UAV_uplink, t_uav_chuanshu)
                print(f"无人机需要的传输能耗{e_uav_chuanshu}焦耳")
                if e_ue > self.ue_battery_list[ue_id]:#终端没能量传输
                    print("终端没能量传输")
                    success += 0  # 任务没成功
                    R_func += (-0.3)  # 奖励函数为0
                    continue
                else:
                    if e_uav_chuanshu>self.e_battery_uav:#无人机没能量传输
                        print("无人机没能量传输")
                        success += 0  # 任务没成功
                        R_func += (-0.3)  # 奖励函数为0
                        continue
                    else:#无人机能传输
                        if required_resources>self.f_satellite or required_resources<0:#卫星的计算资源不够
                            print("卫星的计算资源无法满足")
                            success += 0  # 任务没成功
                            R_func += (-0.1)  # 奖励函数为0
                            continue
                        else:#卫星能计算
                            print("任务卸载到卫星成功")
                            self.ue_battery_list[ue_id] -= e_ue
                            ue_total_energy_consume_step += e_ue
                            self.e_battery_uav -= e_uav_chuanshu
                            UAV_energy_consume_step += e_uav_chuanshu
                            self.f_satellite -= required_resources
                            success += 1
                            offloading_satellite_success+=1
                            R_func += (1-(1e-1)*(ue_total_energy_consume_step+UAV_energy_consume_step))
                            # R_func += (1)
            else:
                success+=0
                R_func+=0
                print("不卸载")
            if all(battery < 0.001 for battery in self.ue_battery_list) or (self.e_battery_uav<5):
                done = True
                print("[调试] 所有终端电量均不足或者无人机没电了")
        print("[调试] 当前步骤结束，终端总能耗:", ue_total_energy_consume_step,"焦耳", "无人机总能耗:", UAV_energy_consume_step,"焦耳")
        self.reset_step()
        print("[调试] 环境已重置，准备进入下一步")
        return self._get_obs(), R_func, success, ue_total_energy_consume_step, UAV_energy_consume_step, offloading_uav_success,offloading_satellite_success,done

#用到的函数#######################################################################################################################
    # 终端随机移动后的位置，任务，遮挡情况
    def reset_step(self):
        # for i in range(self.M):
        #     tmp = [0, 0]  # /终端不移动
        #     # tmp = np.random.rand(2)#终端随机移动
        #     theta_ue = tmp[0] * np.pi * 2  # ue 随机移动角度
        #     dis_ue = tmp[1] * self.delta_t * self.v_ue  # ue 随机移动距离
        #     self.loc_ue_list[i][0] = self.loc_ue_list[i][0] + math.cos(theta_ue) * dis_ue
        #     self.loc_ue_list[i][1] = self.loc_ue_list[i][1] + math.sin(theta_ue) * dis_ue
        #     self.loc_ue_list[i] = np.clip(self.loc_ue_list[i], 0, self.ground_width)
        self.f_uav=1e9
        self.f_satellite=5e9
        self.reset_ue_step()  # 生成终端的任务 遮挡情况

    def zhong_duan_de_chuan_shu_yan_chi(self, ue_id, task_size, loc_ue, loc_uav, p_uplink, block_flag):#终端传输延迟
        # Calculate noise power for each device
        p_noise = self.calculate_p_noise(block_flag)
        # Get the location of the specific user
        loc_ue = loc_ue[ue_id]
        # Calculate the distance between UAV and the user
        dist = np.linalg.norm([loc_uav[0] - loc_ue[0], loc_uav[1] - loc_ue[1], self.height])#终端和无人机之间的距离
        # Calculate the channel gain between UAV and user
        g_uav_ue = self.alpha0 / dist ** 2 #终端和无人机之间的信道增益 Deep reinforcement learning based computation offloadingfor xURLLC services with UAV-assisted IoT-based multi-access edgecomputing system
        # Calculate interference (sum of uplink power of other users weighted by noise)
        interference = sum(p_uplink[i] * p_noise[i] for i in range(len(p_uplink)) if i != ue_id)#计算来自其他用户的干扰
        # Calculate transmission rate
        rate = self.B * np.log2(1 + (p_uplink[ue_id] * g_uav_ue) / (p_uplink[ue_id] * p_noise[ue_id] + interference))
        # Calculate transmission delay
        t_delay = task_size[ue_id] / rate
        return t_delay

    def zhongduandechuanshunenghao(self,ue_id,p_uplink,up_time):#终端传输能耗
        e=p_uplink[ue_id]*up_time#正确的反而不收敛？？？
        return e

    # def wurenjidejisuanyanchi(self, ue_id, task_list, f_uav):#无人机计算延迟
    #     # 计算总卸载任务大小，只考虑 offloading_to_uav == 1 的情况
    #     # sum_offloading_to_uav = sum(task * offloading for task, offloading in zip(task_list, offloading_decision) if offloading == 1)
    #     # 计算无人机的计算延迟
    #     # 如果 sum_offloading_to_uav 为 0，避免除以 0 的错误
    #     # if sum_offloading_to_uav == 0:
    #     #     return float('inf')  # 或者返回一个合适的值来表示无法处理的情况
    #     t = task_list[ue_id] / (f_uav/self.s)
    #     return t

    def wurenjidejisuannenghao(self,ue_id,f_remaining_uav):#无人机计算能耗
        e=self.r * (f_remaining_uav) ** 2 * self.task_list[ue_id]
        return e

    def wurenjidechuanshuyanchi(self, ue_id, task_list, UAV_uplink):#无人机传输延迟
        # 获取用户任务大小和卸载到卫星的比例
        task = task_list[ue_id]
        # 计算无人机到卫星的传输速率
        trans_rate_UAV_satellite = self.B2 * math.log2(1 + UAV_uplink*self.g_UAV_satellite / self.UAV_satellite_noise_p)
        # 计算传输延迟
        t = task / trans_rate_UAV_satellite
        return t

    def wurenjidechuanshunenghao(self,uav_uplink,t_chuanshu):#无人机传输能耗
        e=uav_uplink*t_chuanshu
        return e

    # def weixingdejisuanyanchi(self,ue_id,task_list,offloading_decision):#卫星计算延迟
    #     sum_offloading_to_satellite = sum(task * offloading/2 for task, offloading in zip(task_list, offloading_decision) if offloading==2)
    #     # 计算无人机的计算延迟
    #     t = task_list[ue_id] / ((task_list[ue_id] / sum_offloading_to_satellite) * self.f_satellite)
    #     return t

    def calculate_p_noise(self, block_flag):#计算终端与无人机的信道类型
        # Calculate p_noise for each device based on block_flag
        return [self.p_noisy_los if flag == 0 else self.p_noisy_nlos for flag in block_flag]

    def offloading_the_destination(self,action):#判断卸载目的地
        if action < 1/3:
            return 0
        elif 1 / 3 <= action < 2 / 3:
            return 1
        elif action > 2/3:
            return 2

    def calculate_required_resources(self, ue_id, remaining_time, task_list):#计算剩余所需资源
        # 获取指定终端的任务详情
        task = task_list[ue_id]
        # 计算所需资源
        required_resources = int(task / remaining_time)*self.s
        # 返回计算出的资源需求量
        return required_resources

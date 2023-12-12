import numpy as np
from catch import matrix_util
from catch import config
import math
import rtde_control
import rtde_receive
import rtde_io

# UR5连接参数
HOST = "192.168.101.3"
# HOST = "192.168.206.132"
PORT = 30003

# 信息字典,具体说明在config.py中
dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d', 'I target': '6d',
       'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
       'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
       'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
       'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
       'Tool Accelerometer values': '3d',
       'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd', 'softwareOnly2': 'd',
       'V main': 'd',
       'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
       'Elbow position': '3d', 'Elbow velocity': '3d'}


class RobotControl:

    def __init__(self):
        # 初始化socket来获得数据
        self.tool_acc = 0.5  # Safe: 0.5
        self.tool_vel = 0.2  # Safe: 0.2
        # UR官方的RTDE接口,可用于控制和读取数据
        # rtde_c复制UR5的控制
        self.rtde_c = rtde_control.RTDEControlInterface(HOST)
        # rtde_r负责UR5的数据读取
        self.rtde_r = rtde_receive.RTDEReceiveInterface(HOST)
        # rtde_io负责控制机器臂的数字输出等
        self.rtde_io = rtde_io.RTDEIOInterface(HOST)

    def __del__(self):
        self.rtde_r.disconnect()
        self.rtde_c.disconnect()

    def get_current_tcp(self):
        """ 获得XYZ RXRYRZ,XYZ单位是M,示教器上单位是mm.RXYZ和示教器上一致 """
        return self.rtde_r.getActualTCPPose()

    def get_speed(self):
        """ 获得机器臂的运动速度 """
        return self.rtde_r.getActualTCPSpeed()

    def get_current_angle(self):
        """ 获得各个关节的角度,返回数组,依次为机座,肩部,肘部,手腕1 2 3 """
        # 获得弧度数组
        actual = np.array(self.rtde_r.getActualQ())
        # 转化为角度
        actual = actual * 180 / math.pi
        return actual

    def get_current_radian(self):
        """ 返回各个关节的弧度，返回数组,依次为机座,肩部,肘部,手腕1 2 3 """
        return self.rtde_r.getActualQ()

    def get_current_pos(self):
        """ x, y, theta """
        tcp = self.get_current_tcp()
        rpy = matrix_util.rv2rpy(tcp[3], tcp[4], tcp[5])
        return np.asarray([tcp[0], tcp[1], rpy[-1]])

    def get_current_pos_same_with_simulation(self):
        tcp = self.get_current_tcp()
        rpy = matrix_util.rv2rpy(tcp[3], tcp[4], tcp[5])
        return np.asarray([tcp[1], tcp[0], rpy[-1]])

    def move_up(self, z):
        """机械臂末端向上移动多少mm"""
        tcp = self.get_current_tcp()
        tcp[2] = tcp[2] + z / 1000
        self.rtde_c.moveL(tcp, speed=self.tool_vel, acceleration=self.tool_acc)

    def move_down(self, z):
        """机械臂末端向下移动多少mm"""
        tcp = self.get_current_tcp()
        tcp[2] = tcp[2] - z / 1000
        self.rtde_c.moveL(tcp, speed=self.tool_vel, acceleration=self.tool_acc)

    def reset(self, tool_vel=0.8, tool_acc=0.5):
        """机器臂复位"""
        self.rtde_c.moveJ(
            q=[0.0, -1.570796314870016, -1.570796314870016, -1.570796314870016, 1.5707963705062866, 0.0],
            speed=tool_vel, acceleration=tool_acc)

    def moveJ_Angle(self, angles, tool_vel=0.8, tool_acc=0.5):
        """机器臂复位"""
        self.rtde_c.moveJ(q=[angles[0] / 180 * math.pi, angles[1] / 180 * math.pi, angles[2] / 180 * math.pi,
                             angles[3] / 180 * math.pi, angles[4] / 180 * math.pi, angles[5] / 180 * math.pi],
                          speed=tool_vel, acceleration=tool_acc)

    def open_sucker(self):
        """放气,数字输出0设置为高电平"""
        self.rtde_io.setStandardDigitalOut(0, True)

    def close_sucker(self):
        """ 吸气,数字输出0设置为低电平 """
        self.rtde_io.setStandardDigitalOut(0, False)

    def put_item(self, name):
        # 读取基础放置位置
        # [[乐芙球1号面初始堆叠位置],乐芙球平放高度单位m,乐芙球放置个数,面号]
        put_item = config.put_items[name]
        if put_item is not None:
            #  如果是1号面，则放置后Z轴增加
            if put_item[4] == 1:
                # 移动到堆放位置
                self.rtde_c.moveL(put_item[0])
                # 吸盘放气
                self.open_sucker()
                # Z轴增加
                put_item[0][2] = put_item[0][2] + put_item[1]
                # 该面抓取次数+1
                put_item[3] += 1
                # 回到默认位置
                self.reset()
            # 如果是2号或3号，则是Y轴递减
            else:
                # 移动到堆放位置上方
                self.rtde_c.moveL(put_item[0])
                # 下降
                self.move_down(put_item[2])
                # 吸盘放气
                self.open_sucker()
                # X轴减少
                put_item[0][0] = put_item[0][0] - put_item[1]
                # 该面抓取次数+1
                put_item[3] += 1
                # 回到默认位置
                self.reset()

        else:
            raise Exception("没有要抓的东西的数据，请设置！")

    def catch(self, item):
        """ 移动机械臂抓取 """
        # 打开存放抓取位姿信息的txt
        f = open(config.CATCH_POSE_PATH)
        # 用于存放位姿信息
        pose = []
        # 用来存放抓取面名字
        name = []
        # 开始处理txt
        for line in f.readlines():
            line = line.split()
            info = [i for i in line]
            # 前六个是位姿
            info = info[:6]
            # 最后一个是平面名字
            name.append(line[6])
            # 把位姿信息都变成float格式
            info = [float(i) for i in info]
            pose.append(info)
        pose = pose[0]
        # 单位换算，文件中是mm，但是程序里要m
        pose[0] = item.catch_pose[0] / 1000
        pose[1] = item.catch_pose[1] / 1000
        pose[2] = item.catch_pose[2] / 1000
        print('机械臂将要移动到:  x:', item.catch_pose[0], ' y:', item.catch_pose[1], ' z:', item.catch_pose[2],
              ' rx:', item.catch_pose[3], ' ry:', item.catch_pose[4], ' rz:', item.catch_pose[5])

        name = item.name
        print(name + " is catching!")
        # 吸盘吸气
        self.close_sucker()
        pose1 = pose
        # 上方修正
        pose1[2] = pose1[2] + 0.15
        # 使用moveL,直线移动到目标平面上方
        self.rtde_c.moveL(pose1, speed=0.2, acceleration=0.5)
        # 最后的关节角度和盒状物体保持一致
        back_angle = config.real_back_angle

        if back_angle > 0:
            back_angle = 90 - back_angle
        else:
            back_angle = -back_angle + 90

        # 获得当前各个关节的角度
        current_angle = self.get_current_angle()
        # 获得基座角度和180度的差值
        angle0 = current_angle[0] % 180
        gap_base = math.fabs(180 - angle0)

        current_angle[-1] = back_angle - gap_base
        # print("手腕3:", current_angle[-1])
        # 最后的关节转动
        self.rtde_c.moveJ(q=current_angle / 180 * math.pi)
        # 下降吸取
        self.move_down(160)
        # 上升
        self.move_up(220)
        # 回零
        self.reset()

        config.data['photo_flag'] = True

        # 放置物体
        self.put_item(name)



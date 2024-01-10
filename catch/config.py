# -*- coding: utf-8 -*-
"""
@Time ： 2022/10/25 14:28
@Auth ： 邓浩然
@File ：config.py
@IDE ：PyCharm
@Description：用来存储程序中用到的所有参数
"""

# 待识别照片存储位置:
PHOTO_PATH = 'catch_result/photo/'

# 最新拍摄的RGB图
COLOR_PATH = 'catch_result/photo/color.png'

# 最新拍摄的深度图
DEPTH_PATH = 'catch_result/photo/depth.png'

# 最新由深度估计产生的深度图
PREDICT_DEPTH_PATH = 'catch_result/photo/predict_depth.png'

# 最新拍摄的RGB图
CATCH_IMG_PATH = 'catch_result/photo/catch_point.png'

# masks图片位置
MASKS_COLOR_PATH = 'catch_result/photo/masks/'

# 存储机械臂坐标系下的点云的txt
BASE_CLOUD_PATH = 'catch_result/txt/base_cloud.ply'

# 抓取目标平面的点云
TARGET_POINT_CLOUD_PATH = 'catch_result/txt/target_point_cloud.txt'

# 给机械臂的抓取数据
CATCH_POSE_PATH = 'catch_result/txt/catch_pose.txt'

# 深度估计点云
PREDICT_DEPTH_POINT_CLOUD_PATH = 'catch_result/txt/predict_depth_point_cloud.txt'

# 深度相机点云文件存储
NPY_PATH = 'catch_result/txt/point_cloud.npy'

# 吸盘长度,单位mm
SUCKER_LENGTH = 208

# 计算抓取面的权重的参数,这个值越大,面积的权重越小
AREA_WEIGHT = 520

# 置信度阈值
CONFIDENCE = 0.8

# 相机内参矩阵
CAMERA_INTRINSIC = [[609.919868760916, 0, 327.571937094492],
                    [0, 607.733032143467, 241.738191162382],
                    [0, 0, 1]]

# 相机坐标系到机械臂坐标系的转换矩阵
CAMERA2BASE = [[0.99932303, 0.00383785, 0.03658904, 136.86641864],
               [0.00553989, -0.99890151, -0.0465304, -472.62827245],
               [0.03637027, 0.0467016, -0.99824654, 650.02743144],
               [0., 0., 0., 1.]]

# 深度估计校准值
K = -0.21787237475413496
B = 837.11642422768

# 是否写入机械臂坐标系的点云
# 是否写入相机坐标系的点云
WRITER_BASE_CLOUD = True
WRITER_TARGET_CLOUD = True
WRITER_CAMERA_CLOUD = True

# 是否写入机械臂坐标系的点云
# 是否写入相机坐标系的点云
# WRITER_BASE_CLOUD = False
# WRITER_TARGET_CLOUD = False
# WRITER_CAMERA_CLOUD = False


# 创建字典用于存储全局数据,只会保留最新的彩色图像,深度图像和点云,以及识别结果
# photo_flag:拍摄照片的标志,每次被设置成True都会拍一张照片
# photo_ready:拍照完成的标志，被设置为True后就会进行YOLO识别
# have_item:是否有目标待抓取，为False时程序就会停止
data = {'photo_flag': False, 'have_item': True}

# 存储实验结果
# 存储数据结果的文件夹
test_path = 'catch_result/history/'
# 实验时间
test_time = ''
# 实验次数
count = 1
# 最终路径
final_path = test_path + test_time + '/' + str(count) + '/'
# canny短边与照片边界的夹角,顺时针为负的,逆时针为正的(也可能相反)
real_back_angle = 0

normal_vector1 = 0
normal_vector2 = 0
normal_vector3 = 0
put_items = {
    # [[乐芙球1号面初始堆叠位置],乐芙球平放高度单位m,放置时的下降高度，乐芙球放置个数,面号]
    "LFQ-1": [[0.568, -0.019, -0.080, 3.142, 0, 0], 0.05, 0, 0, 1],
    "LFQ-2": [[0.616, 0.267, -0.011, 3.142, 0, 0], 0.055, 145, 0, 2],
    "LFQ-3": [[0.390, 0.256, 0.225, 3.142, 0, 0], 0.055, 177, 0, 3],
    "nabati-1": [[0.407, -0.005, -0.094, 3.142, 0, 0], 0.038, 0, 0, 1],
    "nabati-2": [[0.603, 0.466, -0.040, 3.142, 0, 0], 0.038, 97, 0, 2],
    "nabati-3": [[0.423, 0.463, 0.076, 3.142, 0, 0], 0.038, 215, 0, 3],
    "pie-1": [[0.256, 0.010, -0.078, 3.142, 0, 0], 0.055, 0, 0, 1],
    "pie-2": [[0.254, 0.269, -0.041, 3.142, 0, 0], 0.055, 136, 0, 2],
    "pie-3": [[0.266, 0.453, -0.031, 3.142, 0, 0], 0.055, 0, 0, 3],
    # "LFQ-1": [[0.474, -0.277, -0.092, 3.142, 0, 0], 0.05, 0, 0, 1],
    # "LFQ-2": [[0.474, -0.442, 0.104, 3.142, 0, 0], 0.055, 136, 0, 2],
    # "LFQ-3": [[-0.549, 0.550, 0.200, 3.142, 0, 0], 0.055, 0, 0, 3],
    # "nabati-1": [[0.504, 0.206, -0.107, 3.142, 0, 0], 0.038, 0, 0, 1],
    # "nabati-2": [[0.476, -0.033, 0.060, 3.142, 0, 0], 0.038, 117, 0, 2],
    # "nabati-3": [[-0.549, 0.350, 0.275, 3.142, 0, 0], 0.038, 0, 0, 3],
}

# 乐芙球1号面初始堆叠位置
LFQ_1_HOME = [-0.08485, 0.48697, -0.140, 3.142, 0, 0]
# 乐芙球平放高度,单位m
LFQ_1_HIGH = 0.05
# 乐芙球放置个数
LFQ_1_NUM = 0

# 乐芙球2号面初始堆叠位置
LFQ_2_HOME = [-0.35352, 0.48695, -0.0021, 3.142, 0, 0]
# 放置乐芙球2号面宽度,单位m
LFQ_2_WIDTH = 0.053
# 乐芙球放置个数
LFQ_2_NUM = 0

# 乐芙球3号面初始堆叠位置
LFQ_3_HOME = [-0.54908, 0.51420, 0.03030, 3.142, 0, 0]
# 放置乐芙球3号面宽度,单位m
LFQ_3_WIDTH = 0.053
# 乐芙球放置个数
LFQ_3_NUM = 0

# cuda配置
threads_per_block = 640
blocks_per_grid = 480

# 'MessageSize': 消息总长度(以字节为单位)
# 'Time': 控制器启动时间
# 'q target': 目标各个关节的角度
# 'qd target': 目标各个关节的速度
# 'qdd target': 目标各个关节加速度
# 'I target': 目标接头电流
# 'M target': 目标关节力矩(力矩)
# 'q actual': 实际各个关节的角度
# 'qd actual': 实际各个关节的速度
# 'I actual': 实际接头电流
# 'I control': 联合控制电流
# 'Tool vector actual': 末端的实际笛卡尔坐标:(x,y,z,rx,ry,rz)，其中rx,ry和rz是工具方向的旋转向量表示
# 'TCP speed actual': 用笛卡尔坐标表示的末端的实际速度
# 'TCP force': TCP中的广义力
# 'Tool vector target': 末端的笛卡尔坐标:(x,y,z,rx,ry,rz)，其中rx,ry和rz是末端方向的旋转向量表示
# 'TCP speed target': 末端在笛卡尔坐标系下的目标速度
# 'Digital input bits': 数字输入的当前状态。注意:这些是编码为int64_t的位，例如，值5对应于位0和位2设置为高
# 'Motor temperatures': 每个关节的温度，单位为摄氏度
# 'Controller Timer': 控制器实时线程执行时间
# 'Test value': 仅供万能机器人软件使用的值
# 'Robot Mode': 机器人模式
# 'Joint Modes': 联合控制模式
# 'Safety Mode': 安全模式
# 'empty1': 仅由万能机器人软件使用
# 'Tool Accelerometer values': 末端x,y和z加速度计值(软件版本1.7)
# 'empty2': 仅由万能机器人软件使用
# 'Speed scaling': 机械臂移动速度,eg: 轨迹限制器的速度缩放
# 'Linear momentum norm': 笛卡尔线性动量范数
# 'SoftwareOnly': 仅由万能机器人软件使用
# 'softwareOnly2': 仅由万能机器人软件使用
# 'V main': 主板:主电压
# 'V robot': 主板:机器人电压(48V)
# 'I robot': 主板:机器人电流
# 'V actual': 实际接头电压
# 'Digital outputs': 数字输出
# 'Program state': 程序状态
# 'Elbow position': 肘部位置
# 'Elbow velocity': 肘部速度


# 机器人模式
# -1 robot_mode_no_controller
# 0	 robot_mode_disconnected
# 1	 robot_mode_confirm_safety
# 2	 robot_mode_booting
# 3	 robot_mode_power_off
# 4	 robot_mode_power_on
# 5	 robot_mode_idle
# 6	 robot_mode_back drive
# 7	 robot_mode_running
# 8	 robot_mode_updating_firmware

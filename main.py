import os
import time
from ultralytics import YOLO
import cv2 as cv
from catch import calculate_pose
from catch import config
from catch import opencv_camera
from catch import robotcontrol
from queue import Queue

if __name__ == '__main__':
    # 初始化机械臂控制类
    rob = robotcontrol.RobotControl()

    print('\n')
    # 机器臂复位
    try:
        rob.reset()
    except:
        print("机械臂移动失败")

    # 存储本次实验结果
    test_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # 在history文件夹下创建本次实验的结果文件夹
    config.test_time = test_time
    config.final_path = config.test_path + config.test_time + '/' + str(config.count) + '/'
    os.makedirs(config.final_path)

    # 创建队列
    q = Queue(maxsize=2)
    # 创建相机线程
    opencv_cam = opencv_camera.Camera(q)
    # 线程开始运行
    opencv_cam.start()

    # 加载模型
    model = YOLO("weight/best-FF.pt")

    # 拍照操作
    config.data['photo_flag'] = True
    # 只要还有目标就继续循环
    while config.data['have_item']:
        # 如果拍照完成
        if q.full():
            # 预测
            results = model([q.get(), q.get()])
            # 保存结果图
            res_plotted = results[0].plot()
            # 保存至photo文件夹下
            cv.imwrite("catch_result/photo/result.png", res_plotted)
            # 保存至历史
            cv.imwrite(config.final_path + "result.png", res_plotted)
            # 对照片进行处理并获得位姿信息
            item = calculate_pose.get_catch_point(results)
            if item is not None:
                # 抓取计数+1
                config.count += 1
                # 新键下次实验的文件夹
                config.final_path = config.test_path + config.test_time + '/' + str(config.count) + '/'
                os.makedirs(config.final_path)
                # 机械臂抓取
                rob.catch(item)
            else:
                break
    # 关闭相机
    # opencv_cam.flag = False

# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/11 13:41
@Auth ： 邓浩然
@File ：opencv_camera.py
@IDE ：PyCharm
@Description：负责拍照的类,使用线程
"""
import threading
import time
from catch import config
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Camera(threading.Thread):
    def __init__(self, queue):
        """
        不使用空间滤波器是因为空间滤波会让边缘变得平滑从而影响角度拟合
        """
        threading.Thread.__init__(self)
        # 队列，用于线程通信
        self.queue = queue
        self.flag = True
        # 数据存放
        self.conf = config.data
        # 用于存放帧集合
        self.frame_list = []
        # 时间滤波器
        self.temporal = rs.temporal_filter()
        # 孔洞填充过滤器
        self.hole_filling = rs.hole_filling_filter()
        # 边缘滤波
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 8)
        # 深度图着色器
        self.colorizer = rs.colorizer()
        # 点云处理工具
        self.pc = rs.pointcloud()

    def depth_rendered(self, depth):
        index = (depth < 580)
        depth[index] = 580
        index = (depth > 770)
        depth[index] = 770
        # 使用Matplotlib显示深度图像，以下设置可以将白边去除
        w = 640
        h = 480
        dpi = 96
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        axes = fig.add_axes([0, 0, 1, 1])
        axes.set_axis_off()
        # 选择颜色方案进行着色
        axes.imshow(depth, cmap='viridis')
        # 保存图片
        plt.savefig(config.PHOTO_PATH+'depth_viridis.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_photo(self, color, depth):
        # 保存图片
        # 保存历史照片,使用时间命名
        name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        # 保存待识别的图片
        cv2.imwrite(config.PHOTO_PATH + 'color.png', color)
        # 保存深度图
        cv2.imwrite(config.PHOTO_PATH + 'depth.png', depth)
        # 保存历史彩色图片
        cv2.imwrite(config.final_path + name + "-color.png", color)
        # 保存历史深度图片
        cv2.imwrite(config.final_path + name + "-depth.png", depth)

        self.depth_rendered(depth)
        # 图片进入队列
        self.queue.put(config.PHOTO_PATH + 'color.png')
        self.queue.put(config.PHOTO_PATH + 'depth_viridis.png')

    # 负责对照片进行处理
    def take_photo(self, color_frame, color_image):
        # 时间滤波
        # temp_filtered = None
        # for x in range(10):
        #     temp_filtered = self.temporal.process(self.frame_list[x])
        # filled_depth = self.hole_filling.process(temp_filtered)
        # 全部滤波
        temp_filtered = None
        filled_depth = None
        for x in range(10):
            filled_depth = self.frame_list[x]
            filled_depth = self.spatial.process(filled_depth)
            filled_depth = self.temporal.process(filled_depth)
            # filled_depth = self.hole_filling.process(filled_depth)
        filtered_depth_image = np.asanyarray(filled_depth.get_data())
        temp_filtered = filled_depth
        # 保存图片
        self.save_photo(color_image, filtered_depth_image)

        # 从深度图获得点数据
        self.pc.map_to(color_frame)
        points = self.pc.calculate(temp_filtered)
        # 保存本次拍摄的ply点云
        points.export_to_ply(config.final_path + 'point_cloud.ply', color_frame)
        # 获取深度图点云的顶点坐标
        vtx = np.asanyarray(points.get_vertices())
        # 获得彩色点
        colorful = np.asanyarray(color_frame.get_data())
        colorful = colorful.reshape(-1, 3)
        # 写入point_cloud
        point_cloud = []
        index = 0
        for p in vtx:
            point_cloud.append(
                [p[0], p[1], p[2], colorful[index][0], colorful[index][1], colorful[index][2]])
            index += 1
        # 如果需要写入点云
        if config.WRITER_BASE_CLOUD:
            # 写入ply点云
            points.export_to_ply(config.BASE_CLOUD_PATH, color_frame)
        # 保存点云npy文件,这是后面点云处理要用到的数据
        np.save(config.NPY_PATH, np.asarray(point_cloud))
        # 保存历史纪录
        # np.save(config.final_path + 'point_cloud.npy', point_cloud)

        # 着色滤波后的深度图
        colorized_depth = np.asanyarray(self.colorizer.colorize(temp_filtered).get_data())
        cv2.imwrite(config.final_path + 'filtered-depth.png', colorized_depth)

        # 未滤波的深度图
        unfiltered_depth = np.asanyarray(self.colorizer.colorize(self.frame_list[0]).get_data())
        cv2.imwrite(config.final_path + 'unfiltered-depth.png', unfiltered_depth)

        # 重置
        self.frame_list.clear()
        print('拍照完成')

    def run(self):
        # 配置深度和颜色流
        pipeline = rs.pipeline()
        camera_config = rs.config()
        # 获取设备产品线，用于设置支持分辨率
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = camera_config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        # 开始红外射线辅助
        # depth_sensor = device.first_depth_sensor()
        # if depth_sensor.supports(rs.option.emitter_enabled):
        #     depth_sensor.set_option(rs.option.emitter_enabled, 0)
        # 设置为High Accuracy模式,高精度模式
        # preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        # for i in range(int(preset_range.max)):
        #     visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
        #     if visulpreset == "High Accuracy":
        #         depth_sensor.set_option(rs.option.visual_preset, i)
        #         break
        # 创建对齐对象（深度对齐颜色）
        align = rs.align(rs.stream.color)
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
        # 深度参数
        camera_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # 颜色参数
        camera_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        # 创建窗口用于显示
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # 开始流式传输
        pipeline.start(camera_config)
        try:
            # 跳过前30帧
            for i in range(30):
                pipeline.wait_for_frames()
            while config.data['have_item']:
                # 等待一对连贯的帧：深度和颜色
                frames = pipeline.wait_for_frames()
                # 对齐后再获取
                aligned_frames = align.process(frames)
                # 获得对齐后的深度和颜色帧
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # 需要进行拍照,就开始存放帧
                if config.data['photo_flag']:
                    self.frame_list.append(depth_frame)
                # 将图像转换为 numpy 数组,也成为了图像
                color_image = np.asanyarray(color_frame.get_data())
                # 在深度图像上应用颜色图
                depth_colormap = self.colorizer.colorize(depth_frame).get_data()

                # 因为opencv使用的是BGR,但是相机用的是RGB,所以要转换
                color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                # 需要进行拍照,我们会获取10帧来进行彩色图,深度图和点云的计算
                if len(self.frame_list) == 10:
                    # 存完了10帧，停止拍照
                    config.data['photo_flag'] = False
                    self.take_photo(color_frame, color_image_bgr)
                # 把颜色图和深度图左右合在一起
                images = np.hstack((color_image_bgr, depth_colormap))
                # 显示图像
                cv2.imshow('RealSense', images)
                cv2.waitKey(1)

        finally:
            # 停止流式传输
            pipeline.stop()

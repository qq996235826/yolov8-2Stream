# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/11 15:55
@Auth ： 邓浩然
@File ：item.py
@IDE ：PyCharm
@Description：一个待抓取面对象
"""
import math

import numpy as np

from catch import config


# 从YOLO获得所有识别面的信息
def get_items(results):
    items = []
    r = results[0].cpu().numpy()
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segmenation masks outputs
    name = ''
    for i in range(len(boxes.cls)):
        # 置信度
        confidence = boxes.conf[i]
        # 不满足置信度的直接跳过
        if confidence < config.CONFIDENCE:
            continue
        # catch的类别为最近的类
        if r.names[boxes.cls[i]] == 'catch':
            far = 999999999
            for index in range(len(boxes.cls)):
                if r.names[boxes.cls[i]] != 'catch':
                    far1 = math.sqrt(
                        (boxes.xywh[i][0] - boxes.xywh[index][0]) ** 2 + (boxes.xywh[i][1] - boxes.xywh[index][1]) ** 2)
                    if far1 < far:
                        far = far1
                        name = r.names[boxes.cls[index]]
        else:
            # 类别名
            name = r.names[boxes.cls[i]]
        # 预测框
        box = boxes.xyxy[i]
        # 中心点
        center_point = boxes.xywh[i]
        # mask
        mask = masks[i].data
        # 面积
        ones = np.where(mask != 0)
        area = len(ones[0])
        # 序号
        num = i
        # 创建对象
        item = YoloItem(name, box, center_point, mask, confidence, area, num)
        items.append(item)
    return items


def print_items(items):
    for item in items:
        item.print_self()


def print_items_catch(items):
    for item in items:
        item.print_catch_info()


class YoloItem:
    # def __getitem__(self, item):

    def __init__(self, name, box, center_point, mask, confidence, area, num):
        # 序号
        self.num = num
        # 名字
        self.name = name
        # 置信度
        self.confidence = confidence
        # 像素中心点,是xywh,前两个是中心点坐标,后两个是宽和高
        self.center_point = center_point
        # 面积
        self.area = area
        # 预测框点
        self.box = box
        # maks
        self.mask = mask
        # 腐蚀后的mask
        self.narrow_mask = None
        # 抓取权重
        self.weight = None
        # 机械臂坐标系下实际的抓取点,一般来说,若中心点有深度,则就是机械臂坐标系下的中心点
        self.catch_pose = None
        # 旋转角度,相对于照片边框
        self.angle = None
        # 拟合平面的点云
        self.target_point_cloud = None
        # 法向量
        self.normal_vector = None

    def print_self(self):
        print("序号: ", self.num, "  类名: ", self.name, "  置信度: ", self.confidence, "  像素中心点: ",
              self.center_point[0:2],
              "  面积: ", self.area)

    def print_catch_info(self):
        print("序号: ", self.num, "  类名: ", self.name, "  置信度: ", self.confidence, "  像素中心点: ",
              self.center_point[0:2],
              "  面积: ", self.area, "  抓取点: ", self.catch_pose)

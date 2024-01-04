import cv2
import numpy as np
import os, glob


def rgb2masks(label_name):
    # 获取标签图像的文件名
    lbl_id = os.path.split(label_name)[-1].split('.')[0]
    # 读取标签图像
    lbl = cv2.imread(label_name, 1)
    # 获取标签图像的高度和宽度
    h, w = lbl.shape[:2]
    # 存储颜色对应的索引的字典
    leaf_dict = {}
    # 索引值
    idx = 0
    # 创建与标签图像相同大小的全白图像
    white_mask = np.ones((h, w, 3), dtype=np.uint8) * 255
    # 遍历标签图像的每个像素
    for i in range(h):
        for j in range(w):
            # 如果当前颜色已经存在于字典中或为黑色，则继续循环
            if tuple(lbl[i][j]) in leaf_dict or tuple(lbl[i][j]) == (0, 0, 0):
                continue
            # 将当前颜色添加到字典中，并分配一个索引值
            leaf_dict[tuple(lbl[i][j])] = idx
            # 创建布尔掩码，标记与当前颜色相等的像素
            mask = (lbl == lbl[i][j]).all(-1)
            # 根据掩码生成彩色叶子图像
            leaf = np.where(mask[..., None], white_mask, 0)
            # 构造叶子图像的文件名
            mask_name = 'label/' + lbl_id + '_leaf_' + str(idx) + '.png'
            # 保存叶子图像
            cv2.imwrite(mask_name, leaf)
            # 更新索引值
            idx += 1


if __name__ == '__main__':
    # 标签图像文件夹路径
    label_dir = 'label'
    # 获取标签图像文件路径列表
    label_list = glob.glob(os.path.join(label_dir, '*.png'))
    # 遍历标签图像文件并处理
    for label_name in label_list:
        rgb2masks(label_name)

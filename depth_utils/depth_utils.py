import cv2
import matplotlib.pyplot as plt


def depth_rendered(depth_path, color):
    im_depth = cv2.imread(depth_path, flags=cv2.IMREAD_ANYDEPTH)

    # index = (im_depth == 65535)
    # im_depth[index] = 1200

    index = (im_depth < 580)
    im_depth[index] = 580
    index = (im_depth > 770)
    im_depth[index] = 770
    # 使用Matplotlib显示深度图像，以下设置可以将白边去除
    w = 640
    h = 480
    dpi = 96
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    axes = fig.add_axes([0, 0, 1, 1])
    axes.set_axis_off()
    # 选择颜色方案进行着色
    axes.imshow(im_depth, cmap=color)
    # 保存图片
    # plt.savefig(save_path + 'color-' +file, bbox_inches='tight', pad_inches=0)
    plt.savefig('../depth_' + color + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    depth_rendered('../depth.png', 'viridis')

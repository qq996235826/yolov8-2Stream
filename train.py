from ultralytics import YOLO
import cv2 as cv

if __name__ == '__main__':
    # model = YOLO('model/yolov8m-seg-2stream-LCA.yaml')  # build a new model from YAML
    # """
    # 更多详细配置在   ultralytics/yolo/cfg/default.yaml
    # """
    # model.train(data="dataset/yolo-10000.yaml", epochs=200, imgsz=640, device=0, batch=4, amp=False,
    #             name='LCA-yolo-10000')

    # val
    # model = YOLO('weight/best-add.pt')  # load an official model
    # metrics = model.val(data="dataset/yolo-10000.yaml", split=['test_rgb', 'test_depth'])

    # 推理
    model = YOLO('weight/best-add.pt')  # load a custom model

    # Predict with the model
    results = model(['catch_result/photo/color.png', 'catch_result/photo/depth_viridis.png'])  # predict on an image*
    # 保存结果图
    res_plotted = results[0].plot()
    # 保存至photo文件夹下
    cv.imwrite("result.png", res_plotted)

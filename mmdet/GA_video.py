
import mmcv
import cv2
import numpy as np
from mmdet.apis import inference, inference_detector


config_file = "mask_rcnn_ballon.py"
checkpoint_file = "work_dirs/mask_rcnn_ballon/epoch_20.pth"

video = mmcv.VideoReader("test_video.mp4")
model = inference.init_detector(config_file, checkpoint_file)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vWrite = cv2.VideoWriter("work_dirs/output_13.mp4", fourcc, video.fps, video.resolution, True)
for i in range(len(video)):
    bgr_img = video[i]
    try:
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)[:, :, None]
    except:
        continue
    gray_img = np.repeat(gray_img, 3, axis=-1)
    output = inference_detector(model, bgr_img)
    for picture in output[1][0]:
        gray_img[picture] = bgr_img[picture]

    model.show_result(
        gray_img,
        output[0],
        score_thr=.3,
        show=False,
        wait_time=0,
        win_name="show",
        bbox_color=None,
        text_color=(200, 200, 200),
        mask_color=None,
        out_file="cache_13.png"
    )
    vWrite.write(cv2.imread("cache_13.png")[..., ::-1])

vWrite.release()
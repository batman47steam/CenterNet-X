import os
import sys

import numpy as np

CENTERNET_PATH = 'lib/'
sys.path.insert(0, CENTERNET_PATH)
#sys.path.insert(0, CENTERNET_PATH)
print(sys.path)

from detectors.detector_factory import detector_factory
from opts import opts

import cv2

MODEL_PATH = '../models/ctdet_coco_dla_2x.pth'
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
# opt.debug = 2 不debug了
detector = detector_factory[opt.task](opt)

# 路径下的图片要都存放成1.jpg， 2.jpg之类的
img = '../images/'
save_file = open('centers.txt', mode='w+')
for i in range(len(os.listdir())):
    file_name = str(i) + ".jpg"
    img_path = os.path.join(img, file_name)
    #img = cv2.imread(os.path.join(img, file_name))
    ret = detector.run(img_path)['results']


    save_file.write('img' + str(i)+'\n')
    # 只要人和车的类别，其次是confidence > 0.3的才进行保留
    objects = []
    centers = []
    for category in ret:
        if category == 1 or category == 3:
            for dets in ret[category]:
                if dets[-1] > 0.4:
                    objects.append(dets[:-1]) # 只要坐标，不要置信度
    # 保存这张图片对应的所有目标
    np.savetxt(save_file, objects, fmt='%d')

save_file.close()

# 用opencv画出结果简单验证一下
# src = cv2.imread(img)
# for object in objects:
#     object = object.astype('int32')
#     x1, y1 = object[:2]
#     x2, y2 = object[2:]
#     cv2.rectangle(src, (x1,y1), (x2,y2), (255,0,0))
# cv2.imshow('result', src)
# cv2.waitKey(-1)

print(ret)
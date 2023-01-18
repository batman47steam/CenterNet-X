import os
import sys

import numpy as np

CENTERNET_PATH = 'lib/'
sys.path.insert(0, CENTERNET_PATH)
#sys.path.insert(0, CENTERNET_PATH)
print(sys.path)

from detectors.detector_factory import detector_factory
from opts import opts

import mmcv
from mmengine.visualization import Visualizer
import cv2

from typing import Sequence

def auto_arrange_images(image_list: list, image_column: int = 2) -> np.ndarray:
    """Auto arrange image to image_column x N row.
    Args:
        image_list (list): cv2 image list.
        image_column (int): Arrange to N column. Default: 2.
    Return:
        (np.ndarray): image_column x N row merge image
    """
    img_count = len(image_list)
    if img_count <= image_column:
        # no need to arrange
        image_show = np.concatenate(image_list, axis=1)
    else:
        # arrange image according to image_column
        image_row = round(img_count / image_column)
        fill_img_list = [np.ones(image_list[0].shape, dtype=np.uint8) * 255
                         ] * (
                             image_row * image_column - img_count)
        image_list.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(image_row):
            start_col = image_column * i
            end_col = image_column * (i + 1)
            merge_col = np.hstack(image_list[start_col:end_col])
            merge_imgs_col.append(merge_col)

        # merge to one image
        image_show = np.vstack(merge_imgs_col)

    return image_show

class ActivationsWrapper:

    def __init__(self, model, target_layers):
        self.model = model
        self.activations = []
        self.inputs = []
        self.hands = []
        self.image = None
        for target_layer in target_layers:
            self.hands.append(
               target_layer.register_forward_hook(self.save_activation)
            )

    def save_activation(self, module, input, output):
        self.activations.append(output)
        self.inputs.append(input)

    def __call__(self, img_path):
        self.activations = []
        results = detector.run(img_path)['results']
        return results, self.activations, self.inputs

    def release(self):
        for handle in self.handles:
            handle.remove()


MODEL_PATH = '../models/ctdet_coco_dla_2x.pth'
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
#opt.debug = 2
detector = detector_factory[opt.task](opt)
print(detector.model)
model = detector.model # 反正就是模范mmyolo的代码
# 路径下的图片要都存放成1.jpg， 2.jpg之类的
img = '../images/8.jpg'
layers = ['base', 'hm']
target_layers = []
for target_layer in layers:
    try:
        target_layers.append(eval(f'model.{target_layer}'))
    except Exception as e:
        print(detector.model)
        raise RuntimeError('layer does not exist', e)

activations_wrapper = ActivationsWrapper(model, target_layers)
# 你自己的result和mmyolo的result输出来的形式是不一样的，所以result没办法直接调用 visiual
_, featmaps, inputs = activations_wrapper(img)


visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')], save_dir='temp_dir')

if not isinstance(featmaps, Sequence):
    featmaps = [featmaps]

flatten_featmaps = []
flatten_inputs = []
for featmap in featmaps:
    if isinstance(featmap, Sequence):
        flatten_featmaps.extend(featmap)
    else:
        flatten_featmaps.append(featmap)

for input in inputs:
    if isinstance(input, Sequence):
        flatten_inputs.extend(input)
    else:
        flatten_inputs.append(input)

input_img = flatten_inputs[0][0]
img = input_img.detach().cpu().numpy().transpose(1, 2, 0)
mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
img = ((img * std + mean) * 255).astype(np.uint8)
cv2.imshow('input',img)
cv2.waitKey(-1)
src = mmcv.imread(img)
src = mmcv.imconvert(src, 'bgr', 'rgb')


# show the results
shown_imgs = []




#drawn_img = cv2.imread(img)
# centernet的原始图片进行的都是仿射变化，不是直接resize，所以这样最后的featuremap可能会对不上
#drawn_img = cv2.resize(drawn_img, (512,512))
for featmap in flatten_featmaps:
    shown_img = visualizer.draw_featmap(
        featmap[0],
        img,
        channel_reduction='squeeze_mean',
        topk=4,
        arrangement=[2,2])
    shown_imgs.append(shown_img)

visualizer.add_image('feat', shown_imgs[6])

# visualizer.show(shown_imgs[0])
# shown_imgs = auto_arrange_images(shown_imgs)
# visualizer.show(shown_imgs)





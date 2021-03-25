from mmdet.apis import init_detector, inference_detector
from mmdet.models import build_detector
import mmcv
from mmcv.utils import build_from_cfg
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import torch


class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results




config_file = '/home/cyril.cero/project/models/vanilla.py'
checkpoint_file = '/home/cyril.cero/project/models/epoch_8.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = init_detector(config_file, checkpoint_file, device=device)

model.eval()
with torch.no_grad():
    img = mmcv.imread('/home/cyril.cero/project/models/ch08af_20190804170000-13-0.jpg', 'color')
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    data = data.to(device)
    result = model(return_loss=False, rescale=False, **data)


print(f'{result}')


# visualize the results in a new window
# model.show_result(img, result)
# # or save the visualization results to image files
# model.show_result(img, result, out_file='/home/cyril.cero/project/models/result.jpg')

# test a video and show the results
# video = mmcv.VideoReader('/home/cyril.cero/project/models/ch08af_20190804170000-17.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     print(f'FRAME {frame} : {result}')
#     model.show_result(frame, result, wait_time=1)
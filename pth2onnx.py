import torch
import torchvision
import torchvision.models as models


dummy_input = torch.randn(1, 3, 1333, 800, device='cuda')
# the_model = TheModelClass(*args, **kwargs)


# checkpoint = torch.load('/home/cyril.cero/mmdetection/publish/vanilla-7cc3f6a9.pth')
# model = model.load_state_dict(checkpoint['state_dict'])


# model = model.load_state_dict(torch.load("/home/cyril.cero/mmdetection/publish/vanilla-7cc3f6a9.pth"))
# 

# model = torch.load('/home/cyril.cero/mmdetection/publish/vanilla-7cc3f6a9.pth', map_location='cuda:0')
model = torch.load('/home/cyril.cero/mmdetection/work_dirs/vanilla/latest.pth', map_location='cuda:0')
torch.onnx.export(model, dummy_input, "/home/cyril.cero/mmdetection/pytorch_converted.onnx", verbose=True)
print("done!")
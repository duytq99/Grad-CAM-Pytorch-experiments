import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import model
from model import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils import data

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            # print(name, module)
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        # print(self.feature_extractor.gradients)
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    print(input.shape)
    return input


def show_cam_on_image(img, mask, i):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img = img.numpy()
    img = np.squeeze(img)
    img = np.transpose(img, (1, 2, 0))
    cam = heatmap*0.7 + np.float32(img)
    cam = cam / np.max(cam)

    plt.imshow(cam)
    plt.savefig('output/cam {}.png'.format(i))
    # plt.show()

    # cv2.imwrite("cam.jpg", np.uint8(255 * cam))
    # cv2.imshow('cam', np.uint8(255 * cam))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        # print(self.model(input))
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        # print(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

import torchsummary
if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model = model.resnet50(pretrained=False)
    model.load_state_dict(torch.load('resnet50_flower_real.pth',  map_location=torch.device('cpu')))
    # checkpoint = torch.load('resnet50_flower_real.pth',  map_location=torch.device('cpu'))
    # model = checkpoint['model']
    # model.load_state_dict(checkpoint['state_dict'])

    # torchsummary.summary(model,(3,128,128))
    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)    # khoi sequence thu 4
    #0 1 2
    # print(grad_cam)
    # img = cv2.imread('image_05636.jpg', 1)
    # img = np.float32(cv2.resize(img, (224, 224))) / 255
    # input = preprocess_image(img)

    transform = transforms.Compose([transforms.Resize((128, 128)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = datasets.ImageFolder(root='E:/QCNN/data/sorted/test', transform=transform)
    datavis = datasets.ImageFolder(root='E:/QCNN/data/sorted/test', transform=transforms.Compose([transforms.Resize((128, 128)), 
                                    transforms.ToTensor()]))
    # define the dataloader to load that single image
    dataloader = data.DataLoader(dataset=dataset, shuffle=True, batch_size=32)
    visloader = data.DataLoader(dataset=datavis, shuffle=True, batch_size=32)

    img, _ = next(iter(dataloader))
    # print(img.shape)
    img_vis, _ = next(iter(visloader))

    for i in range(32):

        npimg = img_vis[i].numpy()
        npimg = np.squeeze(npimg)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show()


        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None
        mask = grad_cam(img[i:i+1,:,:,:], target_index)
        # print(mask) = print(cam) from __call__

        show_cam_on_image(img_vis[i], mask, i)

        # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        # print(model._modules.items())
        # gb = gb_model(input, index=target_index)
        # gb = gb.transpose((1, 2, 0))
        # cam_mask = cv2.merge([mask, mask, mask])
        # cam_gb = deprocess_image(cam_mask*gb)
        # gb = deprocess_image(gb)

        # cv2.imwrite('gb.jpg', gb)
        # cv2.imwrite('cam_gb.jpg', cam_gb)
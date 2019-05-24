import torchvision.models.alexnet as alexnet
import torchvision.models.densenet as densenet
import torchvision
from torchvision import transforms,datasets
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import caffe_classes

train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
data_transforms = {
    'train': train_transform,
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause a bit so that plots are updated
    plt.imsave("tmp",inp)


model = alexnet(pretrained=True)
model.eval()
# Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))
#
# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[class_names[x] for x in classes])
#
# output = model(inputs)
# _,preds = torch.max(output,1)
# print(class_names[classes[0]],class_names[classes[1]],class_names[classes[2]],class_names[classes[3]] )
# print(preds)

# test image ndarray (500,375,3)
image_bgr = cv2.imread('data/t1.jpg')
image_rgb = image_bgr[...,::-1]

transform1 = transforms.Compose([transforms.ToPILImage(),
                                 transforms.Resize(224),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()
                                 ])
invers1 = transforms.Compose([
    transforms.ToPILImage()
])
# transform2 = transforms.Compose([transforms.RandomResizedCrop(224)])

# cv2.namedWindow("test", 0)
# cv2.imshow("test",test_image)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()
# test_image = transform1(test_image)
input_image = transform1(image_rgb)

image_to_show = invers1(input_image)
# cv2.imshow(image_to_show)
# cv2.waitKey(5000)
plt.imshow(image_to_show)
plt.show()
# imshow(test_image)

test_image = input_image.unsqueeze(0)
output = model(test_image)
scores ,preds = torch.max(output,1)
print(preds,scores)
print(caffe_classes.class_names[preds])

# print(test_image)
# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])
#
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)

# visualize_model(model)
# model()

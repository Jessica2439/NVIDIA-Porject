# Houseplant health detection

For people who are starters of growing houseplants, having something that tells you if your plant is sick might be helpful. Sometimes it is obvious that the plant is sick; however, sometimes it is not so obvious. This model can help identify if there is any problem with the plant so that the plant grower can do something as soon as possible.
This model can also be a basement for further training, for example, to identify plant diseases.

## The Algorithm

The model is based on ResNet-18, which is a deep convolutional neural network architecture that has 18 layers. The architecture is specifically designed for image classification tasks in computer vision. It was pre-trained with millions of images of a thousand categories, which builds a powerful foundation for further training.

## Running this project

An example of how to use the model could be as shown below.

pip install onnxruntime #insall the library in terminal

import onnxruntime
import numpy as np

onnx_model_path = "path_to_your_model.onnx"  # Replace with the path to your .onnx model file

sess = onnxruntime.InferenceSession(onnx_model_path) # Create an ONNX runtime session

example_input = np.random.randn(1, 3, 224, 224).astype(np.float32) # Set up example input data (replace with your data)

output = sess.run(None, {'input': example_input})  # Replace 'input' with your model's input node name # Run inference

print("Output:", output)

import onnxruntime

# model = onnxruntime.InferenceSession('../models/yolov8n-face-dynamic.onnx') # dynamic
model = onnxruntime.InferenceSession('../models/yolov8n-face.onnx') # static
input = model.get_inputs()
output = model.get_outputs()

print(input[0].shape)
print(output[0].shape)
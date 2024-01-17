from modules.batch_predict import main, clear_model

image_paths = ['./images/1.jpg', 'images/2.jpg', 'images/3.jpg']
model_path = './models/yolov8n-face-dynamic.onnx'

if __name__ == '__main__':
    clear_model()
    main(image_paths=image_paths, model_path=model_path)
from src.classification import single_image_classifier
from PIL import Image

if __name__ == "__main__":
    tf = single_image_classifier.define_transform()
    img = Image.open("flowers/test_img_rose.jpg")
    img_tensor = single_image_classifier.transform_image(img, tf)
    model = single_image_classifier.load_model('classification_models/flower-cnn.pth')
    prediction = single_image_classifier.predict_image(img_tensor, model)


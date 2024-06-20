#זה הקוד שעובד

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import shutil
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the saved model
def load_saved_model(model_path):
    print(f'טוען מודל מ-{model_path}')
    return tf.keras.models.load_model(model_path)

# פונקציה לטעינה וטיפול בתמונה
def load_and_preprocess_image(image_path):
    print(f'טוען ומעבד תמונה מ-{image_path}')
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# פונקציה לחיזוי מחלקות מתמונות בתיקייה
def predict_from_folder(folder_path, loaded_model):
    predictions = []
    filenames = sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0]))
    print(f'מנבא מחלקות לתמונות בתיקייה: {folder_path}')

    for filename in filenames:
        image_path = os.path.join(folder_path, filename)
        image = load_and_preprocess_image(image_path)
        prediction = loaded_model.predict(image)
        predicted_class_index = np.argmax(prediction)
        predictions.append(int(predicted_class_index))
    return predictions

# פונקציה לעדכון סטטוס תמונה
def update_image(index, status, folder_path):
    image_path = os.path.join(folder_path, f'{index}.png')
    print(f'מעדכן תמונה בנתיב: {image_path}')
    try:
        if os.path.exists(image_path):
            if status == 1:  # אם הסטטוס הוא 1, מחליפים לתמונה מלאה
                new_image_path = full_image
            else:  # אם הסטטוס הוא 0, מחליפים לתמונה ריקה
                new_image_path = empty_image

            print(f'מעתיק {new_image_path} אל {image_path}')
            shutil.copy(new_image_path, image_path)
            return {'message': 'עדכון חניה בוצע בהצלחה'}, 200
        else:
            print(f'התמונה {image_path} לא קיימת')
            return {'message': 'התמונה לא קיימת'}, 400
    except Exception as e:
        print(f'שגיאה: {e}')
        return {'message': str(e)}, 500

# נתיבי קבצים
folder_path = r'C:\Users\User\Downloads\project-Bina\python\codes\path_to_your_desired_folder\car1233'
empty_image = r'C:\Users\User\Downloads\project-Bina\python\codes\path_to_your_desired_folder\71.png'
full_image = r'C:\Users\User\Downloads\project-Bina\python\codes\path_to_your_desired_folder\70.png'
model_path = r'C:\Users\User\Downloads\project-Bina\python\codes\saved_models\vgg16-middlePart.h5'

# טעינת המודל
loaded_model = load_saved_model(model_path)

# נתיב לעדכון סטטוס תמונה
@app.route('/update', methods=['POST'])
def update():
    try:
        data = request.json
        index = data.get('index')
        status = data.get('status')
        print(f'קיבל בקשה לעדכן אינדקס {index} עם סטטוס {status}')

        if index is None or status is None:
            print('נדרשים אינדקס וסטטוס')
            return jsonify({'message': 'נדרשים אינדקס וסטטוס'}), 400

        response, status_code = update_image(index, status, folder_path)

        if status_code == 200:
            predictions = predict_from_folder(folder_path, loaded_model)
            return jsonify(predictions)
        else:
            return jsonify(response), status_code
    except Exception as e:
        print(f'חריגה לא נתפסה: {e}')
        return jsonify({'message': str(e)}), 500

# נתיב לקבלת חיזויים
@app.route('/search', methods=['GET'])
def search():
    predictions = predict_from_folder(folder_path, loaded_model)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)


























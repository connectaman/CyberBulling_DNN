from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import numpy as np
import keras
from keras import models
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array , load_img


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def pred_clean(pred):
  if pred > 0.5:
    return 1
  else:
    return 0

def pred_class(pred):
  res = pred_clean(pred)
  labels = ['Bulling','No Bulling']
  return labels[res]

@app.route("/")
def index():
    return render_template('index.html')
@app.route("/classify",methods=['POST'])
def classify():
    
    image = request.files['file']
    image.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(image.filename)))
    #import model
    model=models.load_model(r"D:\Projects\X-cencia Projects\Bullying vs No Bullying\model\BvsNo_10epoch_85_acc_model.h5")
    path="static/uploads/"+secure_filename(image.filename)
    img = load_img(path,target_size=(128,128))
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = img / 255
    res = model.predict(img)
    final_response = pred_class(res)
    print(final_response)
    return render_template('classification.html',image=secure_filename(image.filename),result=final_response)


if __name__=='__main__':
    app.run(debug=True)

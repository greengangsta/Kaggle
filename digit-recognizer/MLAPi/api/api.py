import flask
from flask import request, jsonify



########################################
import numpy as np
import pandas as pd
import cv2
import warnings 
warnings.filterwarnings('ignore')
import pickle
from keras.models import model_from_json
########################################


#img = cv2.imread('test')

app = flask.Flask(__name__,static_url_path = '/static')
app.config['Debug'] = True

@app.route('/',methods = ['GET'])
def home():
	return app.send_static_file('index.html')

@app.route('/handle_data',methods = ['POST'])
def handle_data():
	n1 = int(request.form['n1'])
	n2 = int(request.form['n2'])
	a = n1 + n2
	return jsonify({"sum": a})

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['cat']
		f.save('static/test')
		return app.send_static_file('classifier.html')
	else :
		return "An error occured."

@app.route('/classifier',methods = ['POST','GET'])
def cat_classifier():
	img = cv2.imread('static/test')
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img,(784,1))
	f = open('static/scaler.txt','rb')
	scaler = pickle.loads(f.read())
	img = scaler.transform(img)
	json_file = open('static/classifier.json','r')
	loaded_classifier_json = json_file.read()
	json_file.close()
	loaded_classifier = model_from_json(loaded_classifier_json)
	loaded_classifier.load_weights('static/classifier.h5')
	loaded_classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics = ['accuracy']) 
	pred = loaded_classifier.predict(img)

	return {"results":"Executed Successfully. The digits is : " + str(np.argmax(pred,axis= -1))}


app.run()

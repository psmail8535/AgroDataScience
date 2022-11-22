from flask import Flask, render_template, request, jsonify
'''
import numpy as np

import cv2
import mahotas.features.texture as mht
import joblib
from sklearn import preprocessing

def thresholdImg(img):
	th = cv2.inRange(img, (0, 0, 0), (110, 150, 150))
	cth = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	and_img = cv2.bitwise_and(img,cth)
	return and_img

def getDataTekstur(img):
	pb = 1
	w = int(864/pb)
	h = int(1152/pb)
	img = cv2.resize(img, (w,h))
	
	img = thresholdImg(img)	
	b,g,r = cv2.split(img)
	features = mht.haralick(r, ignore_zeros=True, 
		preserve_haralick_bug=True, 
		compute_14th_feature=True)
	print('features.shape: ', features.shape)
	print('features: ', features)
	print('features[0]: ', features[0])
	n_ch, n_ft = features.shape
	return features[0][0:n_ft]

filename = 'alpukat_classifier_model.sav'
filename = 'model_knn_rot.sav'
filename = 'model_lda_rot2.sav'
#filename = 'model_svm_rot2.sav'
loaded_model = joblib.load(filename)

filenameScaler = 'alpukat_classifier_scaler.sav'
filenameScaler = 'scaler_knn_rot.sav'
filenameScaler = 'scaler_lda_rot2.sav'
#filenameScaler = 'scaler_svm_rot2.sav'
scalerData = joblib.load(filenameScaler)
'''
#from werkzeug import secure_filename
app = Flask(__name__)

@app.route('/')
def upload_file():
	return render_template('index.html')
'''	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
	global loaded_model
	print('server side')
	if request.method == 'POST':
		predicted = ''
		try:
			f = request.files['file']
			#read image file string data
			filestr = f.read()
			#convert string data to numpy array
			npimg = np.fromstring(filestr, np.uint8)
			# convert numpy array to image
			img = cv2.imdecode(npimg, 1) # cv2.CV_LOAD_IMAGE_UNCHANGED
			
			v_single_test = getDataTekstur(img)
			v_single_test = v_single_test.reshape(1, len(v_single_test))
			
			print('Fitur: \n', v_single_test)
			
			#min_max_scaler = preprocessing.MinMaxScaler()
			#v_single_test = min_max_scaler.fit_transform(v_single_test)
			
			v_single_test = scalerData.transform(v_single_test)
			print('Fitur scaler: \n', v_single_test)
			
			#prob = loaded_model.predict_proba(v_single_test)	
			predicted = loaded_model.predict(v_single_test)	
			strPred = str.title(str(predicted[0]).replace('_',' '))
			print('predicted: ', strPred)
			print('Upload berhasil')

		except:
			print('Terjadi kesalahan.')
			return jsonify({'info': 'Terjadi kesalahan.', 'predicted':'Terjadi kesalahan'}), 200

		return jsonify({'info': 'Upload berhasil', 'predicted': strPred }), 200
'''
if __name__ == '__main__':
   app.run()

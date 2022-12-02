from fastapi import FastAPI, requests
import numpy as np
import os
import cv2
import uvicorn
from keras.models import load_model

# Cấu hình
model_file = "models/cat_dog_classifier.hdf5"

app = FastAPI()
#app.config['UPLOAD_FOLDER'] = "static"

model = load_model(model_file)

@app.post('/UPLOAD_FOLDER')
async def index():
		try:
			if requests.method == 'POST':
				image = requests.files['file']
				if image:
					# Lưu file
					path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
					image.save(path_to_save)
					frame = cv2.imread(path_to_save)

				# Xử lý file
					frame = cv2.resize(frame, dsize=(150,150))
				# Convert thành tensor
					frame = np.expand_dims(frame, axis=0)  # Used for single images
					# Đưa vào model
					prediction_prob = model.predict(frame)[0][0]

					# Xét kết quả
					if prediction_prob < 0.5: # 0-0.5: cat, else dog
						output = "cat"
					else:
						output = "dog"

					return output
			else:
				return "We only accept POST with image file"
		except Exception as ex:
			print(ex)
			return "Error: " + str(ex)


if __name__ == '__main__':
	#app.run(host='0.0.0.0', port = 2345)
	uvicorn.run(app,host='0.0.0.0',port=8008)


from flask import Flask, render_template, url_for, request
import random
import os
import json
from collections import deque


app = Flask(__name__)


# Deque allowing to remember previous image
prev_img = deque(maxlen=2)


# Routes
@app.route('/')
@app.route('/index')
@app.route('/home')
def index():
	return render_template('index.html')

@app.route('/disaneuron', methods=['GET', 'POST'])
def disaneuron():
	# Load global variable prev_img and forms
	global prev_img
	# Deal with forms
	if request.method == 'POST':
		if 'previous' in request.form:
			if len(prev_img) == 2:
				prev_img.pop() # deleting unused image
			return render_template('disaneuron.html', img=prev_img[0], prev_len=len(prev_img))
		elif 'yesneu' in request.form:
			label = 1
		elif 'noneu' in request.form:
			label = 0
		with open('labels.json', 'r+') as f:
		    data = json.load(f)
		    img_path = prev_img[-1]
		    data[img_path[16:-4]] = label
		    f.seek(0) # should reset file position to the beginning.
		    json.dump(data, f, indent=4)
		    f.truncate() # remove remaining part	
	# Define new image
	image_files = os.listdir(os.path.join(os.getcwd(), 'static', 'dataset'))
	image_file = url_for('static', filename='dataset/' + random.choice(image_files))
	prev_img.append(image_file)
	# Return disaneuron.html
	return render_template('disaneuron.html', img=image_file, prev_len=len(prev_img))


# Run app in debug mode
if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=5001) 
	# host='0.0.0.0' is to make it accessible on the network through IP address
	# there is a bug whenever server is active on a port, then shut down then relaunched,
	# therefore, port=n allows to change port

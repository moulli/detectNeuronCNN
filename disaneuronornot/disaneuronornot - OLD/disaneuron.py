from flask import Flask, render_template, url_for, request
import random
import os
import json
from collections import deque
from forms import PreviousImage, ChangeNeuron


app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


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
	prev_form = PreviousImage()
	form = ChangeNeuron()
	# Deal with forms
	# Deal with form to get back
	if prev_form.validate_on_submit() and prev_form.submit.data:
		if len(prev_img) == 2:
			prev_img.pop() # deleting unused image
		return render_template('disaneuron.html', img=prev_img[0], prev_form=prev_form, form=form, prev_len=len(prev_img))
	# Deal with form to assess if neuron
	if form.validate_on_submit():
		if form.submit_Y.data:
			label = 1
		elif form.submit_N.data:
			label = 0
		with open('labels.json', 'r+') as f:
		    data = json.load(f)
		    data[prev_img[-1]] = label
		    f.seek(0) # should reset file position to the beginning.
		    json.dump(data, f, indent=4)
		    f.truncate() # remove remaining part
	# Define new image
	image_files = os.listdir(os.path.join(os.getcwd(), 'static', 'dataset'))
	image_file = url_for('static', filename='dataset/' + random.choice(image_files))
	prev_img.append(image_file)
	# Return disaneuron.html
	return render_template('disaneuron.html', img=image_file, prev_form=prev_form, form=form, prev_len=len(prev_img))


# Run app in debug mode
if __name__ == '__main__':
	app.run(debug=True)
from flask_wtf import FlaskForm
from wtforms import SubmitField


class ChangeNeuron(FlaskForm):
	submit_Y = SubmitField('HEY DIS A NEURON!')
	submit_N = SubmitField('Definitely NOT a neuron')

class PreviousImage(FlaskForm):
	submit = SubmitField('Oops get me back to previous image')
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from helper_functions_v1 import *
 
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'YOUR SECRET KEY'
 
class ReusableForm(Form):
    name = TextField('Input:', validators=[validators.required()])
 
 
@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
 
    print(form.errors)
    if request.method == 'POST':
        sent=request.form['name']
        
        sentence_cosine_similarity_list = calculate_sentence_similarity(get_vector_representation(sent)[0])
        top_five_similar_sentences = find_top_five_max_element_list(sentence_cosine_similarity_list)
        output = display_items_reversed_order(top_five_similar_sentences)
        print(output)
        
        if form.validate():
            # Save the comment here.
            flash(output)
        else:
            flash('All the form fields are required. ')
 
    return render_template('index.html', form=form)
 
if __name__ == "__main__":
    app.run()

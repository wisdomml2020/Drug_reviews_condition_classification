from flask import Flask, render_template,request,redirect,session
import os
import joblib
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

app = Flask(__name__)

app.secret_key=os.urandom(24)
# Model saved with Keras model.save()
MODEL_PATH = 'model/passmodel.pkl'

TOKENIZER_PATH ='model/tfidfvectorizer.pkl'

DATA_PATH ='data/drugsComTrain.csv'

# loading vectorizer
vectorizer = joblib.load(TOKENIZER_PATH)
# loading model
model = joblib.load(MODEL_PATH)

#getting stopwords
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

@app.route('/')
def login():
	return render_template('login.html')

@app.route("/logout")
def logout():
	session.clear()
	return redirect('/')

@app.route('/index')
def index():
	if 'user_id' in session:
		
		
		return render_template('home.html')
	else:
		return redirect('/')






@app.route('/login_validation', methods=['POST'])
def login_validation():
	username = request.form.get('username')
	password = request.form.get('password')
	
	session['user_id']=username
	session['domain']=password
	
	
	
	if username=="admin@gmail.com" and password=="admin":
		
		
        
		return render_template('home.html')
		#return render_template('login.html', data=payload)
	
	
	else:
		
		err="Kindly Enter valid User ID/ Password"
		return render_template('login.html', lbl=err)

	return ""





			
			


@app.route('/predict',methods=["GET","POST"])
def predict():
	if request.method == 'POST':
		raw_text = request.form['rawtext']
		
		
		if raw_text != "":
			clean_text = cleanText(raw_text)
			clean_lst = [clean_text]
   
			tfidf_vect = vectorizer.transform(clean_lst)
			prediction = model.predict(tfidf_vect)
			predicted_cond = prediction[0]
			df = pd.read_csv(DATA_PATH)
			top_drugs = top_drugs_extractor(predicted_cond,df)
			
					
			return render_template('predict.html',rawtext=raw_text,result=predicted_cond,top_drugs=top_drugs)
		else:
			 raw_text ="There is no text to select"

   


def cleanText(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(lemmitize_words))


def top_drugs_extractor(condition,df):
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst







if __name__ == "__main__":
	
	app.run(debug=True, host="localhost", port=8080)
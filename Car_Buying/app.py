from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 


from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
   
	

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
    return render_template("index.html")

# @app.route('/preview')
# def preview():
    # df = pd.read_csv("C:\Users\Dell\Desktop\Random_Forest_Classification\Social_Network_Ads.csv")
    # return render_template("preview.html",df_view = df)

@app.route('/predict',methods=["POST"])
def predict():
	df = pd.read_csv("C:\Users\Dell\Desktop\Random_Forest_Classification\Buyingdata.csv")
	
	rfmodel = open('C:\Users\Dell\Desktop\\randomforestmodel.pkl', 'rb')
	ranformod = joblib.load(rfmodel)
	sc=StandardScaler()
	
	if request.method == 'POST':
		age = request.form['age']
		salary = request.form['salary']
        
		#normalize data to (-1,1) 
		sample_data = np.array([age,salary]).reshape(1,2)
		sample_data = sc.fit_transform(sample_data)
        
		class_predicted = ranformod.predict(sample_data)[0]
	return render_template('results.html',prediction=class_predicted,name = sample_data)
	


if __name__ == '__main__':
	app.run(debug=True)
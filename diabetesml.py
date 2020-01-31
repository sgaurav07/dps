from flask import Flask ,request, jsonify , render_template
import pandas as pd
from sklearn.externals import joblib
import os

app = Flask(__name__) 
# Directory to upload data and save model
UPLOAD_DIRECTORY = os.path.join(os.getcwd(),'data/processed')
MODEL_DIRECTORY = os.path.join(os.getcwd(),'model')
PREDICT_DIRECTORY = os.path.join(os.getcwd(),'predict')
TEST_DIRECTORY = os.path.join(os.getcwd(),'data/testdata')
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

if not os.path.exists(MODEL_DIRECTORY):
    os.makedirs(MODEL_DIRECTORY)

if not os.path.exists(PREDICT_DIRECTORY):
    os.makedirs(PREDICT_DIRECTORY)

if not os.path.exists(TEST_DIRECTORY):
    os.makedirs(TEST_DIRECTORY)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    #fetching values form json
    output_data = {}
    inputData = []
    #adding values into a list
    inputData.append(int(float(request.form['Pregnancies'])))
    inputData.append(int(float(request.form['PlasmaGlucose'])))
    inputData.append(int(float(request.form['DiastolicBloodPressure'])))
    inputData.append(int(float(request.form['TricepsThickness'])))
    inputData.append(int(float(request.form['SerumInsulin'])))
    inputData.append(float(request.form['BMI']))
    inputData.append(float(request.form['DiabetesPedigree']))
    inputData.append(int(float(request.form['Age'])))
    output_data = dataPrediction(inputData)
    return jsonify(output_data)     #returning in form of json

def dataPrediction(inputdata):
    #loading the pre-created model
    with open(os.path.join(MODEL_DIRECTORY,"diabetesPredictionModel.sav"),'rb') as model:    
        loaded_model = joblib.load(model)
        #predict from the loaded model
        inputdata = pd.DataFrame(inputdata).T           #Transposing the dataframe
        diabetes_prediction = loaded_model.predict(inputdata)       #prediction of the inputdata
        diabetes_prediction = int(diabetes_prediction[len(diabetes_prediction)-1])  
        output_data = {"DiabetesPrediction":diabetes_prediction}
        model.close()
        return output_data  #returning output data as dictionary

if __name__ =='__main__':  
    app.run()

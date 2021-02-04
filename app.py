import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open(f'model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        race=flask.request.form['race']
        gender=flask.request.form['gender']
        age=flask.request.form['age']
        admission_type_id=flask.request.form['admission_type_id']
        discharge_disposition_id=flask.request.form['discharge_disposition_id']
        admission_source_id=flask.request.form['admission_source_id']
        time_in_hospital=flask.request.form['time_in_hospital']
        num_lab_procedures=flask.request.form['num_lab_procedures']
        num_procedures=flask.request.form['num_procedures']
        num_medications=flask.request.form['num_medications']
        number_outpatient=flask.request.form['number_outpatient']
        number_emergency=flask.request.form['number_emergency']
        number_inpatient=flask.request.form['number_inpatient']
        number_diagnoses=flask.request.form['number_diagnoses']
        max_glu_serum=flask.request.form['max_glu_serum']
        A1Cresult=flask.request.form['A1Cresult']
        metformin=flask.request.form['metformin']
        repaglinide=flask.request.form['repaglinide']
        nateglinide=flask.request.form['nateglinide']
        chlorpropamide=flask.request.form['chlorpropamide']
        glimepiride=flask.request.form['glimepiride']
        acetohexamide=flask.request.form['acetohexamide']
        glipizide=flask.request.form['glipizide']
        glyburide=flask.request.form['glyburide']
        tolbutamide=flask.request.form['tolbutamide']
        pioglitazone=flask.request.form['pioglitazone']
        rosiglitazone=flask.request.form['rosiglitazone']
        acarbose=flask.request.form['acarbose']
        miglitol=flask.request.form['miglitol']
        troglitazone=flask.request.form['troglitazone']
        tolazamide=flask.request.form['tolazamide']
        insulin=flask.request.form['insulin']
        glyburide_metformin=flask.request.form['glyburide_metformin']
        glipizide_metformin=flask.request.form['glipizide_metformin']
        glimepiride_pioglitazone=flask.request.form['glimepiride_pioglitazone']
        metformin_rosiglitazone=flask.request.form['metformin_rosiglitazone']
        metformin_pioglitazone =flask.request.form['metformin_pioglitazone']
        change=flask.request.form['change']
        diabetesMed=flask.request.form['diabetesMed']
        # Extract the input

        # Make DataFrame for model
        input_variables = pd.DataFrame([[race,
        gender,
        age,
        admission_type_id,
        discharge_disposition_id,
        admission_source_id,
        time_in_hospital,
        num_lab_procedures,
        num_procedures,
        num_medications,
        number_outpatient,
        number_emergency,
        number_inpatient,
        number_diagnoses,
        max_glu_serum,
        A1Cresult,
        metformin,
        repaglinide,
        nateglinide,
        chlorpropamide,
        glimepiride,
        acetohexamide,
        glipizide,
        glyburide,
        tolbutamide,
        pioglitazone,
        rosiglitazone,
        acarbose,
        miglitol,
        troglitazone,
        tolazamide,
        insulin,
        glyburide_metformin,
        glipizide_metformin,
        glimepiride_pioglitazone,
        metformin_rosiglitazone,
        metformin_pioglitazone ,
        change,
        diabetesMed]],
                                       columns=['race',
                                       'gender',
                                       'age',
                                       'admission_type_id',
                                       'discharge_disposition_id',
                                       'admission_source_id',
                                       'time_in_hospital',
                                       'num_lab_procedures',
                                       'num_procedures',
                                       'num_medications',
                                       'number_outpatient',
                                       'number_emergency',
                                       'number_inpatient',
                                       'number_diagnoses',
                                       'max_glu_serum',
                                       'A1Cresult',
                                       'metformin',
                                       'repaglinide',
                                       'nateglinide',
                                       'chlorpropamide',
                                       'glimepiride',
                                       'acetohexamide',
                                       'glipizide',
                                       'glyburide',
                                       'tolbutamide',
                                       'pioglitazone',
                                       'rosiglitazone',
                                       'acarbose',
                                       'miglitol',
                                       'troglitazone',
                                       'tolazamide',
                                       'insulin',
                                       'glyburide-metformin',
                                       'glipizide-metformin',
                                       'glimepiride-pioglitazone',
                                       'metformin-rosiglitazone',
                                       'metformin-pioglitazone',
                                       'change',
                                       'diabetesMed'])

        # Get the model's prediction
        readmitted = model.predict(input_variables)[0]
        if readmitted == 1:
            readmitted = 'Yes'
        else:
            readmitted = 'No'
        return flask.redirect(flask.url_for('result', readmitted=readmitted))
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        # return flask.render_template('main.html',
        #                              original_input={'race':race,
        #                              'gender':gender,'age':age,'admission_type_id':admission_type_id,
        #                              'discharge_disposition_id':discharge_disposition_id,'admission_source_id':admission_source_id,
        # 'time_in_hospital':time_in_hospital,'num_lab_procedures':num_lab_procedures,
        # 'num_procedures':num_procedures,'num_medications':num_medications,
        # 'number_outpatient':number_outpatient,'number_emergency':number_emergency,
        # 'number_inpatient':number_inpatient,'number_diagnoses':number_diagnoses,
        # 'max_glu_serum':max_glu_serum, 'A1Cresult':A1Cresult,'metformin':metformin,
        # 'repaglinide':repaglinide,'nateglinide':nateglinide,'chlorpropamide':chlorpropamide,
        # 'glimepiride':glimepiride,'acetohexamide':acetohexamide,'glipizide':glipizide,'glyburide':glyburide,
        # 'tolbutamide':tolbutamide,'pioglitazone':pioglitazone,'rosiglitazone':rosiglitazone,
        # 'acarbose':acarbose,'miglitol':miglitol,
        # 'troglitazone':troglitazone,'tolazamide':tolazamide,'insulin':insulin,'glyburide-metformin':glyburide_metformin,
        # 'glipizide-metformin':glipizide_metformin,'glimepiride-pioglitazone':glimepiride_pioglitazone,
        # 'metformin-rosiglitazone':metformin_rosiglitazone, 'metformin-pioglitazone':metformin_pioglitazone,
        # 'change':change,'diabetesMed':diabetesMed},
        #                              result=readmitted,
        #                              )
@app.route('/result')
def result():
    readmitted = flask.request.args.get('readmitted', None)
    return flask.render_template('result.html', readmitted=readmitted)
if __name__ == '__main__':
    app.run(debug=True)
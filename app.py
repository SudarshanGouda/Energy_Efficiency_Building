from flask import Flask, render_template, request
import numpy as np
import warnings
import pickle

warnings.filterwarnings('ignore')

file1 = open('final_model.pkl', 'rb')
XG = pickle.load(file1)
file1.close()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        my_dict = request.form

        Relative_Compactness = float(my_dict['Relative_Compactness'])
        Surface_Area = float(my_dict['Surface_Area'])
        Wall_Area = float(my_dict['Wall_Area'])
        Roof_Area = float(my_dict['Roof_Area'])
        Overall_Height = float(my_dict['Overall_Height'])
        Orientation = float(my_dict['Orientation'])
        Glazing_Area = float(my_dict['Glazing_Area'])
        Glazing_Area_Distribution = float(my_dict['Glazing_Area_Distribution'])

        input_features = [[Relative_Compactness, Surface_Area, Wall_Area, Roof_Area, Overall_Height,
                           Orientation, Glazing_Area,Glazing_Area_Distribution]]

        s = np.load('std.npy')
        m = np.load('mean.npy')

        new_data = (np.array(input_features - m)) / s
        prediction_Heating_Load = XG.predict(new_data)[0][0].round(2)
        prediction_Cooling_Load = XG.predict(new_data)[0][1].round(2)

        # <p class="big-font">Hello World !!</p>', unsafe_allow_html=True

        string = 'Heating Load is : ' + str(prediction_Heating_Load)+' And Cooling Load is :'+str(prediction_Cooling_Load)

        return render_template('show.html', string=string)

    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)
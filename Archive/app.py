from flask import Flask, render_template,send_file, request, jsonify
import numpy as np
import pandas as pd
from io import BytesIO
# import webview

app = Flask(__name__)

data_map = {
    "New Players":
    [2215265,	1600710,	1420644,	1317798,	1202895,	1069741,	921308,	796497,	705757,	633193,	653352,	629993,	505238,	446584,	414116,	359104,	331500,	360056,	344169,	282823,	277947,	207307,	190784,	185244,	213023,	206675,	144579,	130470,	132819,	124262,	123275,	155383,	156953,	107535,	93086,	85726,	77790,	83330,	104920,	103024,	83770,	57257,	55600,	45321,	51656,	77378,	76819,	42599,	36279,	36645,	37268,	41267,	59136,	56697,	31572,	26448,	25777,	24161,	27125,	41996,	41511,	24548,	21313,	26334,	25441,	28470],
    "ARPDAU":
    [4.61292784200566,	1.91866368448783,	1.28038701250617,	1.23404770674473,	1.11036790875823,	1.0093402197178,	0.874798200882663,	0.513335969698436,	0.314928792769362,	0.278975158887038,	0.280834375754283,	0.262523263439822,	0.248582483843808,	0.211167237755904,	0.180636208052219,	0.176341559688798,	0.177483712582117,	0.194611033529702,	0.189780934568015,	0.19390212574431,	0.162947925873501,	0.135979383468913,	0.127204285966179,	0.126693459261494,	0.140975616408997,	0.139280790823079,	0.155526307058138,	0.405022379481651,	0.193308624637693,	0.147165333259743,	0.137349708043534,	0.148612195899278,	0.135974185155756,	0.121771448733794,	0.104821708529304,	0.209238006584708,	0.142706874815305,	0.128389872952256,	0.13688908145045,	0.127814582831904,	0.112835535989486,	0.0951311912014044,	0.089991580854122,	0.0919226801781138,	0.100461032033337,	0.110307564696922,	0.0987162909633745,	0.0953144976251429,	0.0901829615453515,	0.0762389536740136,	0.0726341630822733,	0.13094776614169,	0.106729818806191,	0.0966742226396618,	0.0861550358311936,	0.0766272973472634,	0.0684204736211031,	0.0682309765048709,	0.0743881702524909,	0.0736024668636479,	0.0701277822428626,	0.0573838294701091,	0.060628106536622,	0.0577014015739966,	0.0593695645068151,	0.0590416981505469,	0.0663579750260901,	0.0602307651687173,	0.0555601474599758,	0.0621991344931642,	0.215366379400638,	0.133683908705127,	0.1121310086017,	0.123082714535798]
    }

@app.route('/')
def index():
    return render_template('index.html', all_data=list(data_map.keys()))

@app.route('/calculate', methods=['POST'])
def updateChart():
    response = request.get_json()
    selected_data = response["selected_data"]
    # uploaded = response["uploaded"]
    window_size = response["window_size"]
    smooth_factor = response["smooth_factor"]
    vertical_shift = response["vertical_shift"]
    start_end_same = response["start_end_same"]

    # if uploaded == 0:
    #     selected_data = response["selected_data"]
    #     original_data = data_map[selected_data]
    # elif uploaded == 1:
    #     selected_data = request.files['selected_data']
    #     df = pd.read_excel(selected_data)
    #     original_data = df.iloc[:,0].values.tolist()
    
    original_data = data_map[selected_data]

    # Calculate baseline curve with local minima and moving average
    baseline = calculate_baseline(
        data=original_data, 
        window_size=window_size, 
        smooth_factor=smooth_factor,
        start_end_same=start_end_same,
        vertical_shift=vertical_shift
    )

    return jsonify(
        {
            "labels":np.arange(0,len(original_data)).tolist(),
            "original_data":original_data,
            "baseline":baseline.tolist()
        }
    )

@app.route('/export_to_excel', methods=['POST'])
def export_to_excel():
    response = request.get_json()

    original_data = response["original_data"]
    baseline = response["baseline"]
    print(original_data)
    # Combine data and baseline into a DataFrame
    df = pd.DataFrame({'original_data': original_data, 'baseline': baseline})

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    #taken from the original question
    df.to_excel(writer, startrow=0, merge_cells=False, sheet_name="Sheet_1",index=False)

    #the writer has done its job
    writer.close()

    #go back to the beginning of the stream
    output.seek(0)

    #finally return the file
    return send_file(output, download_name="chart_data.xlsx", as_attachment=True)

    # return send_file(df, download_name="chart_data.csv", as_attachment=True)

    # return jsonify({'success': True})

# HELPER FUNCTION
def calculate_baseline(data, window_size, smooth_factor, start_end_same:bool=True,vertical_shift:int=0):
    baseline = np.copy(data)  # Create a copy of the data to store the baseline curve
    for i in range(window_size, len(data) - window_size):
        window = data[i - window_size : i + window_size + 1]
        local_min = np.min(window)
        baseline[i] = local_min
    
    # Apply moving average smoothing
    smoothed_baseline = np.convolve(baseline, np.ones(smooth_factor) / smooth_factor, mode='same')
    
    # smoothed_baseline = np.copy(baseline)
    # for i in range(smooth_factor, len(baseline) - smooth_factor):
    #     smoothed_baseline[i] = np.mean(baseline[i - smooth_factor : i + smooth_factor + 1])
    
    # Apply vertical shift
    min_val = np.min(smoothed_baseline)
    max_val = np.max(smoothed_baseline)
    range_val = max_val - min_val
    scaled_shift = vertical_shift * range_val
    shifted_baseline = smoothed_baseline + scaled_shift

    # Keep the first and last points the same as original data
    if start_end_same:
        shifted_baseline[0] = data[0]
        shifted_baseline[-1] = data[-1]
    
    return shifted_baseline
    

if __name__ == '__main__':
    app.run(debug=True)
    # webview.start()

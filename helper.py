import numpy as np
import pandas as pd
from io import BytesIO
from scipy.signal import convolve

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
    
# data = [2215265,	1600710,	1420644,	1317798,	1202895,	1069741,	921308,	796497,	705757,	633193,	653352,	629993,	505238,	446584,	414116,	359104,	331500,	360056,	344169,	282823,	277947,	207307,	190784,	185244,	213023,	206675,	144579,	130470,	132819,	124262,	123275,	155383,	156953,	107535,	93086,	85726,	77790,	83330,	104920,	103024,	83770,	57257,	55600,	45321,	51656,	77378,	76819,	42599,	36279,	36645,	37268,	41267,	59136,	56697,	31572,	26448,	25777,	24161,	27125,	41996,	41511,	24548,	21313,	26334,	25441,	28470]
# print(data)
# calculate_baseline(data, 1, 3, True,0.01)
    
# def calculate_baseline(data, window_size, smooth_factor, start_end_same:bool=True,vertical_shift:int=0):
#     # baseline = data.copy().astype(float)
#     # baseline = np.copy(data)  # Create a copy of the data to store the baseline curve
#     baseline = []
#     for i in range(window_size, len(data) - window_size):
#         window = data[i - window_size : i + window_size + 1]
#         local_min = np.min(window)
#         # baseline[i] = local_min
#         baseline.append(local_min)

#     # baseline = baseline.values.flatten()
#     # Apply convolution using np.convolve
#     kernel = np.ones(smooth_factor) / smooth_factor
#     smoothed_baseline = np.convolve(baseline, kernel, mode='same')
#     # smoothed_baseline = np.convolve(baseline, np.ones(smooth_factor) / smooth_factor, mode='same')
    
#     # smoothed_baseline = convolve(baseline, np.ones(smooth_factor) / smooth_factor, mode='same')

#     # smoothed_baseline = np.copy(baseline)
#     # for i in range(smooth_factor, len(baseline) - smooth_factor):
#     #     smoothed_baseline[i] = np.mean(baseline[i - smooth_factor : i + smooth_factor + 1])
        

#     # # Apply vertical shift
#     min_val = np.min(smoothed_baseline)
#     max_val = np.max(smoothed_baseline)
#     range_val = max_val - min_val
#     scaled_shift = vertical_shift * range_val
#     shifted_baseline = smoothed_baseline + scaled_shift
    
#     # # Keep the first and last points the same as original data
#     if start_end_same:
#         shifted_baseline[0] = data.values[0]
#         shifted_baseline[-1] = data.values[-1]
    
#     return smoothed_baseline


# def export_to_excel():
#     response = request.get_json()

#     original_data = response["original_data"]
#     baseline = response["baseline"]
#     print(original_data)
#     # Combine data and baseline into a DataFrame
#     df = pd.DataFrame({'original_data': original_data, 'baseline': baseline})

#     output = BytesIO()
#     writer = pd.ExcelWriter(output, engine='xlsxwriter')

#     #taken from the original question
#     df.to_excel(writer, startrow=0, merge_cells=False, sheet_name="Sheet_1",index=False)

#     #the writer has done its job
#     writer.close()

#     #go back to the beginning of the stream
#     output.seek(0)

#     #finally return the file
#     return send_file(output, download_name="chart_data.xlsx", as_attachment=True)

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

import re
import pandas as pd

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    # convert message_date type
    # df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ')
    try :
      df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %H:%M - ')
    except :
      df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df
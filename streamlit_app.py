import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import helper

st.sidebar.title("Baseline Generation Tool")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])
if uploaded_file is not None:
    try:
        # Use BytesIO to handle binary data
        bytes_data = uploaded_file.getvalue()

        # Read the Excel file directly into a DataFrame
        df = pd.read_excel(BytesIO(bytes_data), engine='openpyxl')

        st.write("Choose below parameters to get the baseline")
        window_size = st.slider(
            'Select window_size',
                0, 10,1)
        
        smooth_factor = st.slider(
            'Select smooth_factor',
                0, 10,1)
        
        vertical_shift = st.slider(
            'Select vertical_shift',
                -1.0, 1.0,0.0)
                
        first_and_end_point = st.checkbox('Choose Start and End point as Same')
        st.write(first_and_end_point)
        
        baseline = helper.calculate_baseline(df.iloc[:, 0].tolist(), window_size, smooth_factor, first_and_end_point,vertical_shift)
        
        uploaded_df = pd.DataFrame({'Actual':df.iloc[:, 0].tolist(), 'Baseline' : baseline})
        
        st.line_chart(
        uploaded_df, x = None, y = ["Actual", "Baseline"], color=["#FF0000", "#0000FF"]  # Optional
        )
        st.dataframe(uploaded_df)
        
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        csv = convert_df(uploaded_df)
        st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
        )


    except pd.errors.ParserError:
        st.error("Error: Unable to parse the Excel file.")

# Fetal Health Classification Machine Learning
# IME 565 Midterm, Fall 2024
# Chandler Kuhn

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")

fetalhealth_df = pd.read_csv('fetal_health.csv')

# Set up the title and image of the app
st.title('Fetal Health Classifier: A Machine Learning App') 
st.image('fetal_health_image.gif', width = 400)

# Sidebar for input collection
st.sidebar.header('Fetal Health Features Input')
uploaded_file = st.sidebar.file_uploader("Upload your data", type=["csv"])
st.sidebar.warning('Ensure your data follows the following format:', icon=':material/warning:')  
sample_df = fetalhealth_df.drop(columns = ['fetal_health'])
st.sidebar.write(sample_df.head(5))
ml_type = st.sidebar.radio('Select Preferred Machine Learning Model', ["Random Forest", "Decision Tree", "AdaBoost", "Soft Voting Classifier"])
st.sidebar.info(f'You selected: {ml_type}', icon=':material/check_circle:')

if uploaded_file is None:
    st.info('Please upload data to proceed.', icon=':material/data_info_alert:')
else:
    st.success('CSV file successfully uploaded.', icon=':material/verified:')

# Load the pre-trained models from the pickle files
rf_pickle = open('fetal_health_1.pickle', 'rb') 
clf_rf = pickle.load(rf_pickle) 
rf_pickle.close()

dt_pickle = open('fetal_health_0.pickle', 'rb') 
clf_dt = pickle.load(dt_pickle) 
dt_pickle.close()

ab_pickle = open('fetal_health_2.pickle', 'rb') 
clf_ab = pickle.load(ab_pickle) 
ab_pickle.close()

v_pickle = open('fetal_health_3.pickle', 'rb') 
clf_v = pickle.load(v_pickle) 
v_pickle.close()

def color_cells(val):
    color = input_df.loc[input_df['Predicted Fetal Health'] == val, 'Color'].values[0]
    return color

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    for index, row in input_df.iterrows():
        baseline = row['baseline value']
        accelerations = row['accelerations']
        fetal_movement = row['fetal_movement']
        uterine_contractions = row['uterine_contractions']
        light_decelerations = row['light_decelerations']
        severe_decelerations = row['severe_decelerations']
        prolongued_decelerations = row['prolongued_decelerations']
        abnormal_short_term_variability = row['abnormal_short_term_variability']
        mean_value_of_short_term_variability = row['mean_value_of_short_term_variability']
        percentage_of_time_with_abnormal_long_term_variability = row['percentage_of_time_with_abnormal_long_term_variability']
        mean_value_of_long_term_variability = row['mean_value_of_long_term_variability']
        histogram_width = row['histogram_width']
        histogram_min = row['histogram_min']
        histogram_max = row['histogram_max']
        histogram_number_of_peaks = row['histogram_number_of_peaks']
        histogram_number_of_zeroes = row['histogram_number_of_zeroes']
        histogram_mode = row['histogram_mode']
        histogram_mean = row['histogram_mean']
        histogram_median = row['histogram_median']
        histogram_variance = row['histogram_variance']
        histogram_tendency = row['histogram_tendency']
        
        df = [[baseline, accelerations, fetal_movement, uterine_contractions, light_decelerations, severe_decelerations, prolongued_decelerations, abnormal_short_term_variability, mean_value_of_short_term_variability, percentage_of_time_with_abnormal_long_term_variability, mean_value_of_long_term_variability, histogram_width, histogram_min, histogram_max, histogram_number_of_peaks, histogram_number_of_zeroes, histogram_mode, histogram_mean, histogram_median, histogram_variance, histogram_tendency]]

        if ml_type == "Random Forest":
            pred_value = clf_rf.predict(df)
            pred_proba = clf_rf.predict_proba(df)
            clf = clf_rf
        elif ml_type == "Decision Tree":
            pred_value = clf_dt.predict(df)
            pred_proba = clf_dt.predict_proba(df)
            clf = clf_dt
        elif ml_type == "AdaBoost":
            pred_value = clf_ab.predict(df) 
            pred_proba = clf_ab.predict_proba(df)
            clf = clf_ab
        else:
            pred_value = clf_v.predict(df)
            pred_proba = clf_v.predict_proba(df)
            clf = clf_v
        
        max_proba = np.max(pred_proba)

        if pred_value == 1:
            pred_value = "Normal"
            input_df.at[index, 'Color'] = 'background-color: limegreen'
        elif pred_value == 2:
            pred_value = "Suspect"
            input_df.at[index, 'Color'] = 'background-color: yellow'
        else:
            pred_value = "Pathological"
            input_df.at[index, 'Color'] = 'background-color: orange'

        input_df.at[index, 'Predicted Fetal Health'] = pred_value
        input_df.at[index, 'Prediction Probability (%)'] = f"{max_proba*100:.2f}"


    # Display input summary
    styled_df = input_df.drop(columns = ['Color'])
    # NOTE: Co-pilot used to suggest method to color cells based on predicted fetal health (below)
    styled_df = styled_df.style.applymap(color_cells, subset=['Predicted Fetal Health'])

    st.write("Updated Table with Predicted Fetal Health")
    st.write(styled_df)

    # Additional tabs for DT model performance
    st.subheader("Prediction Performance")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

    # Tab 1: Feature Importance Visualization
    with tab1:
        st.write("### Feature Importance")
        if ml_type == "Random Forest":
            st.image('feature_imp_0.svg')
        elif ml_type == "Decision Tree":
            st.image('feature_imp_1.svg')
        elif ml_type == "AdaBoost":
            st.image('feature_imp_2.svg')
        else:
            st.image('combined_feature_importances.svg')
        st.caption("Features used in this prediction ranked by relative importance.")
    # Tab 2: Confusion Matrix
    with tab2:
        st.write("### Confusion Matrix")
        if ml_type == "Random Forest":
            st.image('confusion_mat_0.svg')
        elif ml_type == "Decision Tree":
            st.image('confusion_mat_1.svg')
        elif ml_type == "AdaBoost":
            st.image('confusion_mat_2.svg')
        else:
            st.image('confusion_mat_3.svg')
        st.caption("Confusion Matrix of model predictions.")
    # Tab 3: Classification Report
    with tab3:
        st.write("### Classification Report")
        if ml_type == "Random Forest":
            report_df = pd.read_csv('class_report_0.csv', index_col = 0).transpose()
        elif ml_type == "Decision Tree":
            report_df = pd.read_csv('class_report_1.csv', index_col = 0).transpose()
        elif ml_type == "AdaBoost":
            report_df = pd.read_csv('class_report_2.csv', index_col = 0).transpose()
        else:
            report_df = pd.read_csv('class_report_3.csv', index_col = 0).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each Health Classification.")
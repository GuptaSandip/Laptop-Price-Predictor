# import streamlit as st 
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.preprocessing import OneHotEncoder

# # Load the model
# with open('pipe.pkl','rb') as file:
#     rf = pickle.load(file)

# # Load the data
# data = pd.read_csv('traindata.csv')

# st.title('Laptop Price Predictor')

# # Input fields for user
# company = st.selectbox('Brand', data['Company'].unique())
# type = st.selectbox('Type',data['TypeName'].unique())
# ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
# os = st.selectbox("OS", data['OpSys'].unique())
# weight = st.number_input('Weight of the laptop')
# touchScreen = st.selectbox('TouchScreen', ['No', 'Yes'])
# ips = st.selectbox('IPS', ['No', 'Yes'])
# screen_size = st.number_input('Screen Size')
# resolution = st.selectbox("Screen Resolution", [
#     '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
#     '2880x1800', '2560x1600', '1440x900', '2304x1440', '2256x1504', 
#     '840x2160', '2160x1440', '366x786', '2736x1824', '2400x1600', 
#     '1920x1200', '2560x1440'
# ])
# cpu = st.selectbox('CPU', data['Cpu Name'].unique())
# hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
# sdd = st.selectbox('SDD(in GB)', [0, 128, 256, 512, 1024, 2048])
# gpu = st.selectbox('GPU(in GB)', data['Gpu brand'].unique())

# if st.button('Predict Price'):
#     # Convert categorical variables to numerical using encoding
#     touchScreen = 1 if touchScreen == 'Yes' else 0
#     ips = 1 if ips == 'Yes' else 0
    
#     x_resolution = int(resolution.split('x')[0])
#     y_resolution = int(resolution.split('x')[1])
#     ppi = ((x_resolution**2) + (y_resolution**2))**0.5 / screen_size
    
#     query_dict = {
#         'Company': [company],
#         'TypeName': [type],
#         'Ram': [ram],
#         'Weight': [weight],
#         'TouchScreen': [touchScreen],
#         'IPS': [ips],
#         'PPI': [ppi],
#         'Cpu Name': [cpu],
#         'HDD': [hdd],
#         'SDD': [sdd],
#         'Gpu brand': [gpu],
#         'OpSys': [os]
#     }
    
#     query = pd.DataFrame(query_dict)
    
#     # Ensure the query data goes through the same preprocessing as the training data
#     prediction = int(np.exp(rf.predict(query)[0]))
    
#     st.title(f"Predicted price for this laptop could be between {prediction-1000}₹ to {prediction+1000}₹")


import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

# Load the model
with open('pipe.pkl', 'rb') as file:
    rf = pickle.load(file)

# Load the data
data = pd.read_csv('traindata.csv')

# Define profile details
profile_name = "Sandip Gupta"
profile_professional_summary = """
I am a passionate data science enthusiast with experience in machine learning, data analysis, and software development. 
I enjoy solving complex problems and building innovative solutions.
"""
profile_skills = ["Python", "Data Analysis", "Streamlit", "Pandas", "NumPy", "Machine Learning", 'Computer Vision', 'NLP']
profile_projects = [
    {"name": "Laptop Price Predictor", "description": "A web app to predict laptop prices based on various features."}
]
profile_linkedin = "https://www.linkedin.com/in/sandip-gupta11/"
profile_github = "https://github.com/guptasandip"
profile_email = "jobsforsandipgupta@gmail.com"

# Sidebar with profile details
st.sidebar.title("About Me")
st.sidebar.write(f"Name: {profile_name}")
st.sidebar.write(f"Professional Summary: {profile_professional_summary}")
st.sidebar.write("Skills:")
for skill in profile_skills:
    st.sidebar.write(f"- {skill}")
st.sidebar.write("Projects:")
for project in profile_projects:
    st.sidebar.write(f"- **{project['name']}**: {project['description']}")

# Social links with icons
st.sidebar.write("Connect With Me:")
linkedin_icon = '<a href="https://www.linkedin.com/in/sandip-gupta11/"><img src="https://img.icons8.com/color/48/000000/linkedin.png" style="margin-right: 10px;"></a>'
github_icon = '<a href="https://github.com/guptasandip"><img src="https://img.icons8.com/nolan/48/github.png" style="margin-right: 10px;"></a>'
email_icon = '<a href="mailto:jobsforsandipgupta@gmail.com"><img src="https://img.icons8.com/color/48/000000/gmail.png" style="margin-right: 10px;"></a>'

st.sidebar.markdown(linkedin_icon + github_icon + email_icon, unsafe_allow_html=True)

# Main content
st.title('Laptop Price Predictor')

st.header("Welcome!")
st.write("""
Hello! I'm Sandip Gupta, a dedicated data science enthusiast. I'm thrilled to share this project with you. 
Feel free to explore the app and see the power of machine learning in action. I'm always open to connecting and discussing 
new ideas, so don't hesitate to reach out through my LinkedIn or GitHub!
""")

st.header("Project Objective")
st.write("""
The objective of this project is to develop a machine learning-based web application that predicts the price of a laptop based on its specifications. By inputting various features such as brand, type, RAM, operating system, weight, screen size, resolution, CPU, HDD, SSD, and GPU, users can receive an estimated price range for the laptop. This tool aims to assist potential buyers in making informed purchasing decisions by providing a quick and accurate price estimate.
""")

st.header("Project Description")
st.write("""
This project involves the creation of a web application using Streamlit, a popular open-source framework for data science and machine learning projects. The application leverages a machine learning model trained on a dataset of laptop specifications and prices to make predictions. Key features of the application include:

- **User-friendly Interface:** Simple and intuitive input fields for users to enter laptop specifications.
- **Real-time Predictions:** Instant price predictions based on user inputs.
- **Professional Design:** A well-designed layout with an "About Me" section showcasing the My profile, skills, projects, and contact information.

The project demonstrates the practical application of machine learning in solving real-world problems and provides an example of how data science can be used to enhance decision-making processes.

""")

st.header("Explore the Prediction!")

# Input fields for user
company = st.selectbox('Brand', data['Company'].unique())
type = st.selectbox('Type', data['TypeName'].unique())
ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
os = st.selectbox("OS", data['OpSys'].unique())
weight = st.number_input('Weight of the laptop')
touchScreen = st.selectbox('TouchScreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox("Screen Resolution", [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
    '2880x1800', '2560x1600', '1440x900', '2304x1440', '2256x1504', 
    '840x2160', '2160x1440', '366x786', '2736x1824', '2400x1600', 
    '1920x1200', '2560x1440'
])
cpu = st.selectbox('CPU', data['Cpu Name'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
sdd = st.selectbox('SDD(in GB)', [0, 128, 256, 512, 1024, 2048])
gpu = st.selectbox('GPU(in GB)', data['Gpu brand'].unique())

if st.button('Predict Price'):
    # Convert categorical variables to numerical using encoding
    touchScreen = 1 if touchScreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    
    x_resolution = int(resolution.split('x')[0])
    y_resolution = int(resolution.split('x')[1])
    ppi = ((x_resolution**2) + (y_resolution**2))**0.5 / screen_size
    
    query_dict = {
        'Company': [company],
        'TypeName': [type],
        'Ram': [ram],
        'Weight': [weight],
        'TouchScreen': [touchScreen],
        'IPS': [ips],
        'PPI': [ppi],
        'Cpu Name': [cpu],
        'HDD': [hdd],
        'SDD': [sdd],
        'Gpu brand': [gpu],
        'OpSys': [os]
    }
    
    query = pd.DataFrame(query_dict)
    
    # Ensure the query data goes through the same preprocessing as the training data
    prediction = int(np.exp(rf.predict(query)[0]))
    
    st.title(f"Predicted price for this laptop could be between {prediction-1000}₹ to {prediction+1000}₹")




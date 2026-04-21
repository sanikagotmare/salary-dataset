
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
# Ensure 'best_model.pkl' is in the same directory as this script
try:
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'best_model.pkl' not found. Make sure the model file is in the same directory.")
    st.stop()

# Define the unique values for categorical features and model columns
unique_genders = ['Male', 'Female', 'Other']
unique_education_levels = ["Bachelor's", "Master's", 'PhD', "Bachelor's Degree", "Master's Degree", 'High School', 'phD']
unique_job_titles = [
    'Software Engineer', 'Data Analyst', 'Senior Manager', 'Sales Associate', 'Director',
    'Marketing Analyst', 'Product Manager', 'Sales Manager', 'Marketing Coordinator', 'Senior Scientist',
    'Software Developer', 'HR Manager', 'Financial Analyst', 'Project Manager', 'Customer Service Rep',
    'Operations Manager', 'Marketing Manager', 'Senior Engineer', 'Data Entry Clerk', 'Sales Director',
    'Business Analyst', 'VP of Operations', 'IT Support', 'Recruiter', 'Financial Manager',
    'Social Media Manager', 'Cloud Engineer', 'Delivery Driver', 'Content Creator', 'Accountant',
    'Digital Marketing Specialist', 'IT Manager', 'Business Development Manager', 'Web Developer',
    'Research Scientist', 'Data Scientist', 'Software Project Manager', 'Product Designer',
    'Marketing Specialist', 'Research Director', 'UI/UX Designer', 'Operations Director', 'Network Engineer',
    'Student', 'Public Relations Specialist', 'Front End Developer', 'Project Engineer', 'Copywriter',
    'Sales Executive', 'UX Designer', 'Account Manager', 'Graphic Designer', 'HR Business Partner',
    'Financial Advisor', 'Training Specialist', 'Creative Director', 'Operations Analyst', 'Senior HR Manager',
    'Supply Chain Manager', 'DevOps Engineer', 'Technical Writer', 'VP of Marketing', 'Data Entry Specialist',
    'Business Consultant', 'IT Administrator', 'Security Analyst', 'Social Media Specialist',
    'Chief Technology Officer', 'Sales Representative', 'Junior Developer', 'Applications Engineer',
    'Sales Manager (Retail)', 'Director of HR', 'Full Stack Developer', 'Administrative Assistant',
    'Medical Doctor', 'IT Technician', 'Help Desk Analyst', 'Digital Content Creator', 'Customer Success Manager',
    'Event Manager', 'Management Consultant', 'Director of Sales', 'Automation Engineer',
    'Head of Marketing', 'Analyst', 'Senior Software Engineer', 'Chief Marketing Officer', 'Legal Counsel',
    'Software Engineer Manager', 'Director of Product Management', 'Area Sales Manager', 'Electrical Engineer',
    'Senior Marketing Manager', 'Customer Service Manager (Retail)', 'Recruitment Coordinator',
    'Warehouse Manager', 'Training Manager', 'Public Relations Manager', 'Junior Designer',
    'Director of Finance', 'Junior HR Generalist', 'Principal Engineer', 'Back end Developer',
    'Software Trainer', 'Product Marketing Manager', 'E-commerce Manager', 'User Experience Designer',
    'Game Developer', 'Service Manager', 'Business Intelligence Analyst', 'Supply Chain Analyst',
    'Facilities Manager', 'Chief Data Officer', 'Supply Chain Coordinator', 'Data Analyst (Entry-Level)',
    'Quality Assurance Engineer', 'Junior Software Engineer', 'Executive Assistant',
    'Senior Financial Analyst', 'Junior HR Business Partner', 'Senior Project Manager',
    'Senior Product Manager', 'Vice President of Sales', 'Senior Data Scientist',
    'Customer Success Specialist', 'Office Manager', 'Chief Operating Officer',
    'Principal Consultant', 'SEO Specialist', 'Director of Engineering', 'Financial Planner',
    'Hardware Engineer', 'Program Manager', 'Strategy Consultant', 'Research Assistant',
    'Junior Accountant', 'Digital Content Specialist', 'Software Architect', 'CEO',
    'Marketing Director', 'Director of HR Technology', 'Chief Financial Officer', 'Director of IT',
    'Operations Coordinator', 'Principal Software Engineer', 'Web Designer', 'Copy Editor',
    'Chief Technology Officer (CTO)', 'Junior Data Analyst', 'Sales Operations Manager',
    'Quality Control Inspector', 'Procurement Manager', 'Scrum Master', 'IT Support Specialist',
    'UX Researcher', 'Technical Recruiter', 'Associate Marketing Manager', 'Help Desk Technician',
    'Senior Consultant', 'SaaS Product Manager', 'Human Resources Coordinator',
    'Junior Marketing Analyst', 'Social Media Manager (Entry-Level)', 'Senior Account Executive',
    'Director of Marketing', 'Human Resources Manager', 'Chemist', 'Geologist',
    'Operations Manager (Retail)', 'Senior Software Engineer (Front-End)', 'Biologist',
    'Civil Engineer', 'Construction Manager', 'Account Executive', 'HR Generalist',
    'Event Coordinator', 'Mechanical Engineer', 'Financial Advisor (Entry-Level)',
    'Recruitment Manager', 'Project Coordinator', 'Senior Data Analyst', 'Estate Agent',
    'Photographer', 'Sales Associate (Retail)', 'Software Engineer (Entry-Level)',
    'Human Resources Director', 'Full Stack Engineer', 'Customer Service Representative',
    'Digital Marketing Manager', 'Data Entry Clerk (Part-Time)', 'Public Relations Specialist (Entry-Level)'
]

# This list of model_columns MUST EXACTLY match the columns your model was trained on
model_columns = [
    'Age', 'Years of Experience', 'Gender_Male', 'Gender_Other',
    "Education Level_Bachelor's Degree", 'Education Level_High School', "Education Level_Master's",
    "Education Level_Master's Degree", 'Education Level_PhD', 'Education Level_phD',
    'Job Title_Account Executive', 'Job Title_Account Manager', 'Job Title_Accountant',
    'Job Title_Administrative Assistant', 'Job Title_Analyst', 'Job Title_Applications Engineer',
    'Job Title_Area Sales Manager', 'Job Title_Associate Marketing Manager', 'Job Title_Automation Engineer',
    'Job Title_Back end Developer', 'Job Title_Biologist', 'Job Title_Business Analyst',
    'Job Title_Business Consultant', 'Job Title_Business Development Manager',
    'Job Title_Business Intelligence Analyst', 'Job Title_CEO', 'Job Title_CFO', 'Job Title_Chemist',
    'Job Title_Chief Data Officer', 'Job Title_Chief Financial Officer', 'Job Title_Chief Marketing Officer',
    'Job Title_Chief Operating Officer', 'Job Title_Chief Technology Officer',
    'Job Title_Chief Technology Officer (CTO)', 'Job Title_Civil Engineer',
    'Job Title_Cloud Engineer', 'Job Title_Construction Manager', 'Job Title_Content Creator',
    'Job Title_Copy Editor', 'Job Title_Copywriter', 'Job Title_Creative Director',
    'Job Title_Customer Service Manager', 'Job Title_Customer Service Manager (Retail)',
    'Job Title_Customer Service Representative', 'Job Title_Customer Success Manager',
    'Job Title_Customer Success Specialist', 'Job Title_Data Analyst',
    'Job Title_Data Analyst (Entry-Level)', 'Job Title_Data Entry Clerk',
    'Job Title_Data Entry Clerk (Part-Time)', 'Job Title_Data Entry Specialist',
    'Job Title_Data Scientist', 'Job Title_Delivery Driver', 'Job Title_DevOps Engineer',
    'Job Title_Digital Content Creator', 'Job Title_Digital Content Specialist',
    'Job Title_Digital Marketing Manager', 'Job Title_Digital Marketing Specialist',
    'Job Title_Director', 'Job Title_Director of Engineering', 'Job Title_Director of Finance',
    'Job Title_Director of HR', 'Job Title_Director of HR Technology',
    'Job Title_Director of IT', 'Job Title_Director of Marketing',
    'Job Title_Director of Product Management', 'Job Title_Director of Sales',
    'Job Title_E-commerce Manager', 'Job Title_Electrical Engineer', 'Job Title_Estate Agent',
    'Job Title_Event Coordinator', 'Job Title_Event Manager', 'Job Title_Executive Assistant',
    'Job Title_Facilities Manager', 'Job Title_Financial Analyst',
    'Job Title_Financial Advisor', 'Job Title_Financial Advisor (Entry-Level)',
    'Job Title_Financial Manager', 'Job Title_Financial Planner', 'Job Title_Front End Developer',
    'Job Title_Full Stack Developer', 'Job Title_Full Stack Engineer', 'Job Title_Game Developer',
    'Job Title_Geologist', 'Job Title_Graphic Designer', 'Job Title_HR Business Partner',
    'Job Title_HR Generalist', 'Job Title_HR Manager', 'Job Title_Hardware Engineer',
    'Job Title_Head of Marketing', 'Job Title_Help Desk Analyst',
    'Job Title_Help Desk Technician', 'Job Title_Human Resources Coordinator',
    'Job Title_Human Resources Director', 'Job Title_Human Resources Manager',
    'Job Title_IT Administrator', 'Job Title_IT Manager', 'Job Title_IT Support',
    'Job Title_IT Support Specialist', 'Job Title_IT Technician', 'Job Title_Junior Accountant',
    'Job Title_Junior Data Analyst', 'Job Title_Junior Designer',
    'Job Title_Junior Developer', 'Job Title_Junior HR Business Partner',
    'Job Title_Junior HR Generalist', 'Job Title_Junior Marketing Analyst',
    'Job Title_Junior Software Engineer', 'Job Title_Legal Counsel',
    'Job Title_Management Consultant', 'Job Title_Marketing Analyst',
    'Job Title_Marketing Coordinator', 'Job Title_Marketing Director',
    'Job Title_Marketing Manager', 'Job Title_Marketing Specialist',
    'Job Title_Medical Doctor', 'Job Title_Mechanical Engineer',
    'Job Title_Network Engineer', 'Job Title_Office Manager', 'Job Title_Operations Analyst',
    'Job Title_Operations Coordinator', 'Job Title_Operations Director',
    'Job Title_Operations Manager', 'Job Title_Operations Manager (Retail)', 'Job Title_Photographer',
    'Job Title_Principal Consultant', 'Job Title_Principal Engineer',
    'Job Title_Principal Software Engineer', 'Job Title_Procurement Manager',
    'Job Title_Product Designer', 'Job Title_Product Manager',
    'Job Title_Product Marketing Manager', 'Job Title_Program Manager',
    'Job Title_Project Engineer', 'Job Title_Project Manager',
    'Job Title_Project Coordinator', 'Job Title_Public Relations Manager',
    'Job Title_Public Relations Specialist',
    'Job Title_Public Relations Specialist (Entry-Level)', 'Job Title_Quality Assurance Engineer',
    'Job Title_Quality Control Inspector', 'Job Title_Recruiter',
    'Job Title_Recruitment Coordinator', 'Job Title_Recruitment Manager',
    'Job Title_Research Assistant', 'Job Title_Research Director',
    'Job Title_Research Scientist', 'Job Title_SaaS Product Manager',
    'Job Title_Sales Associate', 'Job Title_Sales Associate (Retail)', 'Job Title_Sales Director',
    'Job Title_Sales Executive', 'Job Title_Sales Manager', 'Job Title_Sales Manager (Retail)',
    'Job Title_Sales Operations Manager', 'Job Title_Sales Representative', 'Job Title_Scrum Master',
    'Job Title_Security Analyst', 'Job Title_Senior Account Executive',
    'Job Title_Senior Consultant', 'Job Title_Senior Data Analyst',
    'Job Title_Senior Data Scientist', 'Job Title_Senior Engineer',
    'Job Title_Senior Financial Analyst', 'Job Title_Senior HR Manager',
    'Job Title_Senior Manager', 'Job Title_Senior Marketing Manager',
    'Job Title_Senior Product Manager', 'Job Title_Senior Project Manager',
    'Job Title_Senior Scientist', 'Job Title_Senior Software Engineer',
    'Job Title_Senior Software Engineer (Front-End)', 'Job Title_Service Manager',
    'Job Title_SEO Specialist', 'Job Title_Social Media Manager',
    'Job Title_Social Media Manager (Entry-Level)', 'Job Title_Social Media Specialist',
    'Job Title_Software Architect', 'Job Title_Software Developer',
    'Job Title_Software Engineer', 'Job Title_Software Engineer (Entry-Level)',
    'Job Title_Software Engineer Manager', 'Job Title_Software Project Manager',
    'Job Title_Software Trainer', 'Job Title_Strategy Consultant', 'Job Title_Student',
    'Job Title_Supply Chain Analyst', 'Job Title_Supply Chain Coordinator',
    'Job Title_Supply Chain Manager', 'Job Title_Technical Recruiter',
    'Job Title_Technical Writer', 'Job Title_Training Manager',
    'Job Title_Training Specialist', 'Job Title_UI/UX Designer', 'Job Title_User Experience Designer',
    'Job Title_UX Designer', 'Job Title_UX Researcher', 'Job Title_VP of Marketing',
    'Job Title_VP of Operations', 'Job Title_Vice President of Sales', 'Job Title_Warehouse Manager',
    'Job Title_Web Developer', 'Job Title_Web Designer'
]

st.title('Salary Prediction App')
st.write('Enter employee details to predict their salary.')

# User Inputs
age = st.slider('Age', 18, 65, 30)
years_of_experience = st.slider('Years of Experience', 0, 40, 5)

gender = st.selectbox('Gender', unique_genders)
education_level = st.selectbox('Education Level', unique_education_levels)
job_title = st.selectbox('Job Title', unique_job_titles)

# Prepare input for the model
input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

input_data['Age'] = age
input_data['Years of Experience'] = years_of_experience

# One-hot encode categorical features
if f'Gender_{gender}' in model_columns:
    input_data[f'Gender_{gender}'] = 1
# Handle variations in 'Education Level' columns
education_col_name = f"Education Level_{education_level}"
if education_col_name not in model_columns:
    # Try alternative names if original not found (e.g., Bachelor's vs Bachelor's Degree)
    if education_level == "Bachelor's":
        if "Education Level_Bachelor's Degree" in model_columns: education_col_name = "Education Level_Bachelor's Degree"
    elif education_level == "Master's":
        if "Education Level_Master's Degree" in model_columns: education_col_name = "Education Level_Master's Degree"
    elif education_level == "PhD":
        if "Education Level_phD" in model_columns: education_col_name = "Education Level_phD"

if education_col_name in model_columns:
    input_data[education_col_name] = 1


if f'Job Title_{job_title}' in model_columns:
    input_data[f'Job Title_{job_title}'] = 1

# Prediction
if st.button('Predict Salary'):
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Salary: ${prediction:,.2f}')

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import openai

# Set page config
st.set_page_config(page_title="HealthGuard Pro", page_icon="👩‍⚕️")

page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://www.shutterstock.com/image-vector/abstract…les-medical-background-vector-260nw-131117981.jpg");
  background-size: cover;
}
</style>
"""

st.markdown(page_element, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-family: Arial, sans-serif;'>HealthGuard Pro</h1>", unsafe_allow_html=True)

# Loading the saved models
diabetes_model = pickle.load(open('C:\\Users\\VIDHI\\Desktop\\HealthGuard Pro\\Disease\\diabetes_model_new.sav', 'rb'))
heart_disease_model = pickle.load(open('C:\\Users\\VIDHI\\Desktop\\HealthGuard Pro\\Disease\\heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('C:\\Users\\VIDHI\\Desktop\\HealthGuard Pro\\Disease\\parkinsons_model.sav', 'rb'))
breast_cancer_model = pickle.load(open('C:\\Users\\VIDHI\\Desktop\\HealthGuard Pro\\Disease\\breast_cancer_model.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'HealthGuard Pro',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction','Breast Cancer Prediction'],
        icons=['capsule-pill', 'heart-pulse', 'graph-up-arrow', 'hospital'],
        default_index=0
    )
    
# Load Images
diabetes_image_path = "C:/Users/VIDHI/Desktop/Project/Multiple-Disease-Prediction-System/Multiple Disease Prediction System/pics/diabetes_image.png"
heart_image_path = "C:/Users/VIDHI/Desktop/Project/Multiple-Disease-Prediction-System/Multiple Disease Prediction System/pics/heart_image.png"
parkinsons_image_path = "C:/Users/VIDHI/Desktop/Project/Multiple-Disease-Prediction-System/Multiple Disease Prediction System/pics/parkinsons_image.png"
breast_cancer_path = "C:/Users/VIDHI/Desktop/HealthGuard Pro/Images/breast_cancer.png"

# Display Images
def display_image(image_path):
    try:
        img = Image.open(image_path)
        st.image(img, use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")

# Diabetes Prediction Page
# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # Page title with custom styling
    st.markdown("<h1 style='text-align: center; font-family:Lucida Sans;'>Diabetes Prediction using ML</h1>", unsafe_allow_html=True)
    display_image(diabetes_image_path)  # Assuming display_image is a function to show images

    # Creating tabs for different categories of inputs
    tab1, tab2 = st.tabs(["Basic Information", "Medical Information"])

    # Basic Information Tab for non-medical user inputs
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')
            Age = st.text_input('Age of the Person')

        with col2:
            BMI = st.text_input('BMI value')
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    # Medical Information Tab for medical-specific user inputs
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            Glucose = st.text_input('Glucose Level')
            BloodPressure = st.text_input('Blood Pressure value')

        with col2:
            SkinThickness = st.text_input('Skin Thickness value')
            Insulin = st.text_input('Insulin Level')

    # Prediction and result display remains common and outside the tabs
    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        # Assuming diabetes_model is a pre-trained model available here
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_diagnosis = 'The person is Diabetic' if diab_prediction[0] == 1 else 'The person is Not Diabetic'
    st.success(diab_diagnosis)


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    # Page title with custom styling
    st.markdown("<h1 style='text-align: center; font-family:Lucida Sans;'>Heart Disease Prediction using ML</h1>", unsafe_allow_html=True)
    display_image(heart_image_path)  # Assuming display_image is a function to show images

    # Creating tabs for different categories of inputs
    tab1, tab2 = st.tabs(["Personal Information", "Medical Information"])

    # Personal Information Tab for general user inputs
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input('Age of the Person')
            sex = st.number_input('Sex of the Person')

        with col2:
            cp = st.number_input('Chest pain types')
            fbs = st.number_input('Fasting blood sugar > 120 mg/dl')

    # Medical Information Tab for medical-specific user inputs
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            trestbps = st.number_input('Resting Blood Pressure')
            chol = st.number_input('Serum Cholestoral in mg/dl')
            restecg = st.number_input('Resting Electrocardiographic results')

        with col2:
            thalach = st.number_input('Maximum Heart Rate achieved')
            exang = st.number_input('Exercise Induced Angina')
            oldpeak = st.number_input('ST depression induced by exercise')

    # Additional inputs
    slope = st.number_input('Slope of the peak exercise ST segment')
    ca = st.number_input('Major vessels colored by flourosopy')
    thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # Prediction and result display remains common and outside the tabs
    heart_diagnosis = ''
    if st.button('Heart Test Result'):
        # Assuming heart_disease_model is a pre-trained model available here
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        heart_diagnosis = 'The person is suffering from Heart disease' if heart_prediction[0] == 1 else 'The person is Not suffering from Heart disease'
    st.success(heart_diagnosis)


# Parkinsons Prediction Page
if selected == 'Parkinsons Prediction':
    # Page title with custom styling
    st.markdown("<h1 style='text-align: center; font-family:Lucida Sans;'>Parkinsons Prediction using ML</h1>", unsafe_allow_html=True)
    display_image(parkinsons_image_path)  # Assuming display_image is a function to show images

    # Creating tabs for different categories of inputs
    tab1, tab2, tab3 = st.tabs(["MDVP Features", "Jitter & Shimmer", "Other Features"])

    # Tab 1: MDVP Features
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fo = st.text_input('MDVP: Fo(Hz)')
            fhi = st.text_input('MDVP: Fhi(Hz)')
            flo = st.text_input('MDVP: Flo(Hz)')

        with col2:
            RPDE = st.text_input('RPDE')
            DFA = st.text_input('DFA')
            spread1 = st.text_input('spread1')
            spread2 = st.text_input('spread2')

    # Tab 2: Jitter & Shimmer
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Jitter_percent = st.text_input('MDVP: Jitter(%)')
            Jitter_Abs = st.text_input('MDVP: Jitter(Abs)')
            RAP = st.text_input('MDVP: RAP')
            APQ = st.text_input('MDVP: APQ')
            
            
        with col2:
            PPQ = st.text_input('MDVP: PPQ')
            DDP = st.text_input('Jitter: DDP')
            Shimmer = st.text_input('MDVP: Shimmer')
            
            
        with col3:
            Shimmer_dB = st.text_input('MDVP: Shimmer(dB)')
        APQ3 = st.text_input('Shimmer: APQ3')
        APQ5 = st.text_input('Shimmer: APQ5')
        DDA = st.text_input('Shimmer: DDA')

    # Tab 3: Other Features
    with tab3:
        NHR = st.text_input('NHR')
        HNR = st.text_input('HNR')
        D2 = st.text_input('D2')
        PPE = st.text_input('PPE')

    # Prediction and result display remains common and outside the tabs
    parkinsons_diagnosis = ''
    if st.button('Parkinsons Test Result'):
        # Assuming parkinsons_model is a pre-trained model available here
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
        parkinsons_diagnosis = 'The person is suffering from Parkinsons disease' if parkinsons_prediction[0] == 1 else 'The person is Not suffering from Parkinsons disease'
    st.success(parkinsons_diagnosis)

    
# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    # Page title
    # Page title with custom styling
    st.markdown("<h1 style='text-align: center; font-family:Lucida Sans;'>Breast Cancer Prediciton using ML</h1>", unsafe_allow_html=True)
    display_image(breast_cancer_path)

    # Creating tabs for different categories of inputs
    tab_basic, tab_error, tab_worst = st.tabs(["Basic Features", "Error Features", "Worst Features"])
    
    # Basic Features Tab
    with tab_basic:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            mean_radius = st.text_input('Mean Radius')
            mean_smoothness = st.text_input('Mean Smoothness')
            mean_symmetry = st.text_input('Mean Symmetry')

        with col2:
            mean_texture = st.text_input('Mean Texture')
            mean_compactness = st.text_input('Mean Compactness')
            mean_fractal_dimension = st.text_input('Mean Fractal Dimension')

        with col3:
            mean_perimeter = st.text_input('Mean Perimeter')
            mean_concavity = st.text_input('Mean Concavity')

        with col4:
            mean_area = st.text_input('Mean Area')
            mean_concave_points = st.text_input('Mean Concave Points')

    # Error Features Tab
    with tab_error:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            perimeter_error = st.text_input('Perimeter Error')
            concavity_error = st.text_input('Concavity Error')

        with col2:
            area_error = st.text_input('Area Error')
            concave_points_error = st.text_input('Concave Points Error')

        with col3:
            radius_error = st.text_input('Radius Error')
            symmetry_error = st.text_input('Symmetry Error')

        with col4:
            smoothness_error = st.text_input('Smoothness Error')
            fractal_dimension_error = st.text_input('Fractal Dimension Error')

    # Worst Features Tab
    with tab_worst:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            worst_radius = st.text_input('Worst Radius')
            worst_smoothness = st.text_input('Worst Smoothness')
            worst_symmetry = st.text_input('Worst Symmetry')

        with col2:
            worst_texture = st.text_input('Worst Texture')
            worst_compactness = st.text_input('Worst Compactness')
            worst_fractal_dimension = st.text_input('Worst Fractal Dimension')

        with col3:
            worst_perimeter = st.text_input('Worst Perimeter')
            worst_concavity = st.text_input('Worst Concavity')

        with col4:
            worst_area = st.text_input('Worst Area')
            worst_concave_points = st.text_input('Worst Concave Points')
        
    # Code for prediction
    cancer_diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Breast Cancer Test Result'):
        if not all([mean_radius, mean_texture, mean_perimeter, mean_area,
                                                       mean_smoothness, mean_compactness, mean_concavity,
                                                       mean_concave_points, mean_symmetry, mean_fractal_dimension,
                                                       radius_error, perimeter_error, area_error,
                                                       smoothness_error,concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
                                                       worst_radius, worst_texture, worst_perimeter, worst_area,
                                                       worst_smoothness, worst_compactness, worst_concavity,
                                                       worst_concave_points, worst_symmetry, worst_fractal_dimension]):
            st.warning("Please fill in all the fields.")
        else:
            cancer_prediction = breast_cancer_model.predict([[mean_radius, mean_texture, mean_perimeter, mean_area,
                                                       mean_smoothness, mean_compactness, mean_concavity,
                                                       mean_concave_points, mean_symmetry, mean_fractal_dimension,
                                                       radius_error, perimeter_error, area_error,
                                                       smoothness_error,concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
                                                       worst_radius, worst_texture, worst_perimeter, worst_area,
                                                       worst_smoothness, worst_compactness, worst_concavity,
                                                       worst_concave_points, worst_symmetry, worst_fractal_dimension]])
            
            if cancer_prediction[0] == 1:
                cancer_diagnosis = 'The person is diagnosed with breast cancer.'
            else:
                cancer_diagnosis = 'The person is not diagnosed with breast cancer.'
        
    st.success(cancer_diagnosis)

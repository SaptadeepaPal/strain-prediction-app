import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import sys
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="ANN %Strain Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .button-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .result-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .input-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

def load_model_and_scaler():
    """Load the pre-trained model and scaler"""
    try:
        # Load the trained model
        if os.path.exists('ann_model.h5'):
            model = tf.keras.models.load_model('ann_model.h5', compile=False)
        else:
            st.error("❌ Model file 'ann_model.h5' not found! Please ensure the file is in the same directory.")
            return None, None
        
        # Load the scaler
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
        else:
            st.error("❌ Scaler file 'scaler.pkl' not found! Please ensure the file is in the same directory.")
            return None, None
            
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model or scaler: {str(e)}")
        return None, None

def predict_strain(expose_duration, output_voltage, model, scaler):
    """Make prediction for single input"""
    try:
        # Scale only the voltage
        scaled_voltage = scaler.transform(np.array([[output_voltage]]))

        # Use constant temperature and pressure
        temperature = 25
        pressure = 1

        # Combine all inputs into one array of shape (1, 4)
        input_scaled = np.array([[expose_duration, scaled_voltage[0][0], temperature, pressure]])

        # Make prediction
        prediction = model.predict(input_scaled, verbose=0)

        return prediction[0][0]
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None

def process_excel_file(file_path_or_buffer, model, scaler):
    """Process Excel file for batch predictions"""
    try:
        # Read Excel file
        if isinstance(file_path_or_buffer, str):
            df = pd.read_excel(file_path_or_buffer)
        else:
            df = pd.read_excel(file_path_or_buffer)
        
        # Check if required columns exist
        required_columns = ['Expose Duration', 'Output Voltage']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("📋 Required columns: 'Expose Duration', 'Output Voltage'")
            return None
        
        # Only scale the 'Output Voltage' column
        scaled_voltage = scaler.transform(df[['Output Voltage']].values)

        # Combine with unscaled 'Expose Duration'
        input_scaled = np.hstack((df[['Expose Duration']].values, scaled_voltage))
        
        # Make predictions
        predictions = model.predict(input_scaled, verbose=0)
        
        # Add predictions to dataframe
        df['Predicted %Strain'] = predictions.flatten()
        df['Temperature (°C)'] = 25  # Constant temperature
        df['Pressure (atm)'] = 1    # Constant pressure
        
        return df

    except Exception as e:
        st.error(f"❌ Error processing Excel file: {str(e)}")
        return None

def main_page():
    """Main page with navigation buttons"""
    st.markdown("""
    <div class="main-header">
        <h1>🧠 ANN %Strain Prediction System</h1>
        <p>Artificial Neural Network Model for Strain Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎛️ Choose an Option:")
    
    # Create three columns for buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔢 Single Manual Input", use_container_width=True, type="primary"):
            st.session_state.page = 'single_input'
            st.rerun()
    
    with col2:
        if st.button("📁 Batch Input from Excel", use_container_width=True, type="secondary"):
            st.session_state.page = 'batch_input'
            st.rerun()
    
    with col3:
        if st.button("❌ Exit", use_container_width=True, type="secondary"):
            st.session_state.page = 'exit'
            st.rerun()

def single_input_page():
    """Single manual input page"""
    st.markdown("### 🔢 Single Manual Input")
    
    # Back button
    if st.button("← Back to Main", type="secondary"):
        st.session_state.page = 'main'
        st.session_state.prediction_result = None
        st.rerun()
    
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        expose_duration = st.number_input(
            "🕒 Expose Duration",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Enter the exposure duration value"
        )
    
    with col2:
        output_voltage = st.number_input(
            "⚡ Output Voltage",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Enter the output voltage value"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Predict button
    if st.button("🔮 Predict %Strain", type="primary", use_container_width=True):
        if expose_duration == 0.0 and output_voltage == 0.0:
            st.warning("⚠️ Please enter valid values for both parameters.")
        else:
            with st.spinner("Loading model and making prediction..."):
                model, scaler = load_model_and_scaler()
                
                if model is not None and scaler is not None:
                    prediction = predict_strain(expose_duration, output_voltage, model, scaler)
                    
                    if prediction is not None:
                        st.session_state.prediction_result = {
                            'expose_duration': expose_duration,
                            'output_voltage': output_voltage,
                            'prediction': prediction
                        }
    
    # Display results
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        
        st.markdown("### 📊 Prediction Results")
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Input Parameters:**")
            st.write(f"🕒 Expose Duration: {result['expose_duration']}")
            st.write(f"⚡ Output Voltage: {result['output_voltage']}")
            st.write(f"🌡️ Temperature: 25°C (constant)")
            st.write(f"🔘 Pressure: 1 atm (constant)")
        
        with col2:
            st.markdown("**🎯 Prediction:**")
            st.markdown(f"### {result['prediction']:.4f}% Strain")
            st.success("✅ Prediction completed successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)

def batch_input_page():
    """Batch input from Excel file page"""
    st.markdown("### 📁 Batch Input from Excel File")
    
    # Back button
    if st.button("← Back to Main", type="secondary"):
        st.session_state.page = 'main'
        st.rerun()
    
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # File upload method selection
    upload_method = st.radio(
        "📂 Choose input method:",
        ["Upload Excel File", "Enter File Path"],
        horizontal=True
    )
    
    uploaded_file = None
    file_path = None
    
    if upload_method == "Upload Excel File":
        uploaded_file = st.file_uploader(
            "📤 Upload Excel file",
            type=['xlsx', 'xls'],
            help="Upload an Excel file containing 'Expose Duration' and 'Output Voltage' columns"
        )
    else:
        file_path = st.text_input(
            "📍 Enter file path:",
            placeholder="C:/path/to/your/file.xlsx",
            help="Enter the full path to your Excel file"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process button
    if st.button("🔮 Predict %Strain for File", type="primary", use_container_width=True):
        if uploaded_file is not None or (file_path and os.path.exists(file_path)):
            with st.spinner("Processing Excel file and making predictions..."):
                model, scaler = load_model_and_scaler()
                
                if model is not None and scaler is not None:
                    # Process the file
                    file_to_process = uploaded_file if uploaded_file else file_path
                    results_df = process_excel_file(file_to_process, model, scaler)
                    
                    if results_df is not None:
                        st.success("✅ Batch prediction completed successfully!")
                        
                        # Display results
                        st.markdown("### 📊 Prediction Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download button
                        output_buffer = BytesIO()
                        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                            results_df.to_excel(writer, index=False, sheet_name='Predictions')
                        
                        st.download_button(
                            label="📥 Download Results",
                            data=output_buffer.getvalue(),
                            file_name="strain_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary"
                        )
        else:
            if upload_method == "Enter File Path":
                st.error("❌ File path does not exist or is invalid.")
            else:
                st.warning("⚠️ Please upload an Excel file.")

def exit_page():
    """Exit page"""
    st.markdown("""
    <div class="main-header">
        <h1>👋 Thank You!</h1>
        <p>Thank you for using the ANN %Strain Prediction System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎉 Session Summary")
    st.info("The application has been successfully used for strain prediction analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Restart Application", type="primary", use_container_width=True):
            st.session_state.page = 'main'
            st.session_state.prediction_result = None
            st.rerun()
    
    with col2:
        if st.button("🚪 Close Application", type="secondary", use_container_width=True):
            st.markdown("### 🛑 Application Closed")
            st.info("You can close this browser tab now.")
            st.stop()

def main():
    """Main application function"""
    # Page routing
    if st.session_state.page == 'main':
        main_page()
    elif st.session_state.page == 'single_input':
        single_input_page()
    elif st.session_state.page == 'batch_input':
        batch_input_page()
    elif st.session_state.page == 'exit':
        exit_page()

if __name__ == "__main__":
    main()
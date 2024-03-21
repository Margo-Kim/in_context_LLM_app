# Import necessary libraries
import streamlit as st  # Streamlit library for creating web apps
import pandas as pd  # Pandas library for data manipulation
import numpy as np  # NumPy library for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for plotting graphs
import warnings  # Module to control warning messages
import base64  # Module for encoding/decoding data
from datasets import Dataset  # Import Dataset class from datasets library
# from transformers import pipeline  # Uncomment to use transformers pipeline (if needed)

import sys
# Add a specific directory to the Python path for additional modules
sys.path.append('/workspaces/ai-assisted-coding_panther/ai_assisted_coding_panther')
from inLoop import inLoop  # Custom module import
from askDataset import AskDataset  # Custom module import
from labeling import train_model, evaluate_model_accuracy, iterative_training, predict_and_update_labels  # Custom labeling functions

# Ignore all warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page layout
st.set_page_config(layout="wide")

# Sidebar setup for navigation
st.sidebar.title('Navigation')
# Radio button for page navigation
page = st.sidebar.radio("Go to", ['Home', 'AI-Assisted Labeling', 'Ask the Dataset', 'Advanced AI-Assisted Labeling'])


# Class containing helpful static methods for various functionalities
class Helpful:
    
    @staticmethod
    def load_csv(uploaded_file):
        # Static method to load CSV file and handle exceptions
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
            return None

    @staticmethod
    def get_table_download_link(df, filename):
        # Static method to create a download link for a DataFrame
        try:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV File", data=csv, file_name=filename, mime='text/csv')
        except Exception as e:
            st.error(f"An error occurred while generating the download link: {e}")

    @staticmethod
    def ask_question_about_dataset(df):
        # Static method for asking a question about the dataset
        question = st.text_input("Ask a question about the dataset")
        if df is None:
            st.write("The answer to your question would appear here.")

    @staticmethod
    def handle_file_upload(single_file=True):
        # Static method to handle file upload, either single or multiple files
        if single_file:
            # For uploading a single file
            uploaded_file = st.file_uploader("Upload CSV file", type="csv")
            if uploaded_file is not None:
                df = Helpful.load_csv(uploaded_file)
                # Display DataFrame with highlighting null values
                st.dataframe(df.style.applymap(lambda x: 'background-color : yellow' if pd.isnull(x) else ''))
                # Markdown to download the processed CSV file
                st.markdown(Helpful.get_table_download_link(df, f"processed_{uploaded_file.name}"), unsafe_allow_html=True)
                return df
        else:
            # For uploading multiple files
            uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type="csv")
            if uploaded_files:
                df_list = [Helpful.load_csv(file) for file in uploaded_files]
                combined_df = pd.concat(df_list, ignore_index=True)
                st.dataframe(combined_df)
                return combined_df

    @staticmethod
    def upload_csv_files():
        # Static method for uploading multiple CSV files
        uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type="csv")
        if uploaded_files:
            df_list = [Helpful.load_csv(file) for file in uploaded_files]
            combined_df = pd.concat(df_list, ignore_index=True)
            return combined_df
        return None

    @staticmethod
    def display_lines_for_labeling(df):
        # Static method to display specific lines from a DataFrame for labeling
        start_line = st.number_input("Select the starting line number", min_value=1, max_value=len(df), value=1)
        num_lines = st.number_input("Select the number of lines to display", min_value=1, max_value=20, value=5)
        displayed_lines = df.iloc[start_line:start_line+num_lines]
        st.dataframe(displayed_lines)
        return start_line, num_lines


    





class HomePage:
    @staticmethod
    def builder():
        st.title("Welcome to ReTeach AI-Assisted Classroom Interaction Analysis Platform")
        st.header("Overview:")
        st.write("This project, developed by Aryaan Upadhyay, Margo Kim, Sarah Auch, and Tessora Stefan, introduces an innovative solution for analyzing classroom interactions. The AI-Assisted Classroom Interaction Analysis Platform is a groundbreaking tool designed to enhance educational research and practice by providing a detailed, AI-powered analysis of classroom transcripts.")
        st.subheader("Features:")
        st.write("The ReTeach AI-Assisted Classroom Interaction Analysis Platform offers a variety of features, including:")
        st.write("**AI-Assisted Labeling:** This feature allows users to upload a CSV file, which will be processed using a Large Language Model (LLM). The processed file, now updated with all the necessary labels, can then be downloaded, allowing users to review the labeled data.")
        st.write("**Ask the Dataset:** This feature allows users to pose questions to the AI about the uploaded classroom data, turning CSV files into a rich source of insights. Here, users receive AI-generated answers, enabling a deeper understanding of the classroom dynamics captured in their data.")
        st.write("**Advanced AI-Assisted Labeling:** This feature enables an advanced, interactive approach where users can refine AI predictions by labeling a set of transcript statements, which the AI then uses to enhance its accuracy on subsequent data batches. It displays the evolving accuracy of the AI model in real-time, allowing users to track improvements and download the progressively refined dataset.")




class AiAssistedLabeling:
    @staticmethod
    def builder():
        st.title("AI-Assisted Labeling")
        st.header("Overview:")
        st.write("AI-Assisted Labeling is designed to facilitate the labeling of ReTeach classroom interactions data. Users can upload a CSV file, which will be processed using a Large Language Model (LLM). The processed file, now updated with all the necessary labels, can then be downloaded, allowing users to review the labeled data.")
        st.header("Instructions:")
        st.write("""Please upload a csv file. After the csv is uploaded the file will begin being processed once the "Start AI Labeler" button is pressed. Once the file is processed the updated file will be displayed below. The updated file can be downloaded by clicking the "Download CSV File" button.
                 """)
        if 'best_accuracy' not in st.session_state:
            st.session_state['best_accuracy'] = 0
        if 'best_df' not in st.session_state:
            st.session_state['best_df'] = None
        if 'initial_df' not in st.session_state:
            st.session_state['initial_df'] = None  # This will store the initial dataframe

        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            # Save the name of the uploaded file
            uploaded_file_name = uploaded_file.name
            st.session_state['uploaded_file_name'] = uploaded_file_name

            if st.session_state['initial_df'] is None:
                st.session_state['initial_df'] = Helpful.load_csv(uploaded_file)

            df = st.session_state['initial_df']

            if 'Label' not in df.columns:
                df['Label'] = 'UNLABELED'
            df['Label'].fillna('UNLABELED', inplace=True)

            # Display the initial DataFrame with labels
            st.write("Initial Data with Labels:")
            st.dataframe(df)

        if st.button('Start AI Labeler') and st.session_state['initial_df'] is not None:
            df = st.session_state['initial_df']
            with st.spinner('Labeling in progress...'):
                model, tokenizer = None, None

                # Perform iterative training and track the best accuracy
                for _ in range(1): 
                    model, tokenizer = iterative_training(df, model, tokenizer, max_iterations=1)
                    df = predict_and_update_labels(df, model, tokenizer)
                    current_accuracy = evaluate_model_accuracy(df, model, tokenizer)

                if current_accuracy > st.session_state['best_accuracy']:
                    st.session_state['best_accuracy'] = current_accuracy
                    st.session_state['best_df'] = df.copy()

            st.success('Labeler completed.')

            # Remove the initial dataframe from session state to stop displaying it
            st.session_state['initial_df'] = None

        # Display the best DataFrame with labels after training
        if st.session_state['best_df'] is not None:
            st.write("Updated Data with Labels:")
            st.dataframe(st.session_state['best_df'])

        # Download link for the best dataset
        if st.session_state['best_df'] is not None:
            download_file_name = "AI_labeled_" + st.session_state.get('uploaded_file_name', 'default_filename.csv')
            st.markdown(Helpful.get_table_download_link(st.session_state['best_df'], download_file_name), unsafe_allow_html=True)
            
            #st.markdown(Helpful.get_table_download_link(st.session_state['best_df'], "best_accuracy_labels.csv"), unsafe_allow_html=True)


class AskTheDataset:
    @staticmethod
    def builder():
        st.title("Ask the Dataset")
        st.header("Overview:")
        st.write("Ask the Dataset is a tool specifically tailored for querying ReTeach classroom interaction data. Users begin by uploading a CSV file, which triggers the appearance of an input box. They can then pose questions about the data directly to our Chat GPT API. The responses to these inquiries are conveniently displayed below the input box.")
        st.header("Instructions:")
        st.write("""Please upload a CSV file. Once the CSV file is uploaded, an input box will appear, allowing the user to type their question. After the user clicks the “Ask” button, the question and file are sent to the Chat Bot. The answer will be displayed below the “Ask” button as soon as it is available.
                 """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                ask_dataset = AskDataset(uploaded_file)
            except Exception as e:
                st.error(f"An error occurred during initialization: {e}")
            

            # UI for asking a question
            user_question = st.text_input('Enter your question:', '')

            # Button to submit the question
            if st.button('Ask'):
                if user_question:
                    try:
                        # Getting the response from AskDataset
                        response = ask_dataset.ask_chatgpt(user_question)
                        st.write(response)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.warning("Please enter a question.")
         

        
        

class AdvancedAiLabeling:
    @staticmethod
    def builder():
        st.title("Advanced AI-Assisted Labeling")
        st.header("Overview:")
        st.write("Advanced AI-Assisted Labeling is designed to streamline the labeling process of ReTeach classroom interaction data. On this page, users can upload their data and train an AI assistant. After uploading, the user is prompted to select labels for the first five rows, aiding in the training of the chatbot. Subsequently, five new label suggestions are presented, with the option for the user to correct any mislabeled items by the chatbot. This process continues until all labels are returned and validated as correct by the user. Finally, users have the option to download the processed CSV file. ")
        st.header("Instructions:")
        st.write("""Please upload a CSV file. Once the CSV is uploaded, the user will be prompted to label the first five rows. These rows are educated guesses made by the chatbot. The user can view the original dataset in the displayed box. After validating and correcting the labels, the user should press 'Confirm Labels'. The chatbot will then present the next five rows with its predicted labels. This process continues until all rows have been validated or confirmed. Upon completion, the user will be able to download the processed CSV file.
        """)

        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    # Initialize inLoop once when file is uploaded
        if uploaded_file is not None and "inLoop" not in st.session_state:
            data = pd.read_csv(uploaded_file)
            st.session_state['inLoop'] = inLoop(data)
            st.session_state['index'] = 0
            st.session_state['data'] = data  # Store the data in session state

        if 'inLoop' in st.session_state:
            ai_labeler = st.session_state['inLoop']
            data = st.session_state['data']
            chunk_size = 5
            start = st.session_state['index']
            end = min(start + chunk_size, len(data))

            data_chunk = data.iloc[start:end]

            if not data_chunk.empty:
                st.write("Data for Labeling:", data_chunk)
                label_options = ["OTR", "PRS", "REP", "NEU"]  # Replace with your actual labels

                # Get predictions and process them
                raw_predictions = ai_labeler.get_prediction_for_text(data_chunk)
                processed_predictions = ai_labeler.turn_dataframe(raw_predictions, data_chunk)

                # Displaying predictions and allowing label changes
                user_labels = []
                for i, row in enumerate(data_chunk.iterrows()):
                    predicted_label = processed_predictions.iloc[i]['Label']
                    print(predicted_label)
                    if predicted_label not in label_options:
                        default_index = 0  # Fallback to the first option if prediction is not in label options
                    else:
                        default_index = label_options.index(predicted_label)
                    
                    label = st.selectbox(f"Label for row {start + i}", options=label_options, index=default_index, key=f"label_{start + i}")
                    user_labels.append(label)

                if st.button("Confirm Labels", key="confirm_labels"):
                    # Update the labels in the dataframe
                    for i, label in enumerate(user_labels):
                        data.at[start + i, 'Label'] = label 

                    # Save the updated dataframe to a new CSV
                    # data.to_csv('updated_data.csv', index=False)
                    
                    st.success("Labels Updated and Saved")
                    st.download_button(label="Download Labeled Data",
                                data= data.to_csv(index=False).encode('utf-8'),
                                file_name='labeled_data.csv',
                                mime='text/csv')
                    # Move to the next chunk
                    st.session_state['index'] += chunk_size

                    if st.session_state['index'] >= len(data):
                        st.write("You have labeled all the data.")
                    else:
                        st.experimental_rerun()


  
if page == "Home":
     HomePage.builder()
elif page == "AI-Assisted Labeling":
    AiAssistedLabeling.builder()
elif page == "Ask the Dataset":
    AskTheDataset.builder()
elif page == "Advanced AI-Assisted Labeling":
    AdvancedAiLabeling.builder()

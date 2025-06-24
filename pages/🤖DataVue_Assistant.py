import streamlit as st
from groq import Groq
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        color: #1e90ff;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        padding-top: 20px;
    }
    .subheader {
        color: #4169e1;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .ai-response {
        background-color: #e6f2ff;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .example-question {
        background-color: #d1ecff;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .example-question:hover {
        background-color: #a8d4ff;
        transform: translateY(-2px);
    }
    .sidebar .sidebar-content {
        background-color: #e6f2ff;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .feedback-btn {
        margin: 5px;
        padding: 8px 16px;
        border-radius: 5px;
        font-weight: bold;
    }
    .positive-feedback {
        background-color: #4CAF50;
        color: white;
    }
    .negative-feedback {
        background-color: #f44336;
        color: white;
    }
    .resources-list {
        padding-left: 20px;
    }
    .resources-list li {
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def display_example_questions(questions):
    st.sidebar.markdown("### Example Questions")
    for q in questions:
        if st.sidebar.button(q, key=f"btn_{q}"):
            st.session_state.question = q
            st.session_state.run_query = True

def display_data_science_datasets():
    st.sidebar.markdown("### Practice Datasets")
    datasets = ["Iris", "Titanic", "Boston Housing", "Wine Quality", "Diabetes"]
    for dataset in datasets:
        if st.sidebar.button(f"Load {dataset} Dataset", key=f"dataset_{dataset}"):
            if dataset == "Iris":
                df = sns.load_dataset('iris')
                st.session_state.current_dataset = df
                st.session_state.dataset_name = "Iris"
            elif dataset == "Titanic":
                df = sns.load_dataset('titanic')
                st.session_state.current_dataset = df
                st.session_state.dataset_name = "Titanic"
            elif dataset == "Boston Housing":
                from sklearn.datasets import load_boston
                boston = load_boston()
                df = pd.DataFrame(boston.data, columns=boston.feature_names)
                df['target'] = boston.target
                st.session_state.current_dataset = df
                st.session_state.dataset_name = "Boston Housing"
            elif dataset == "Wine Quality":
                df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
                st.session_state.current_dataset = df
                st.session_state.dataset_name = "Wine Quality"
            elif dataset == "Diabetes":
                from sklearn.datasets import load_diabetes
                diabetes = load_diabetes()
                df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
                df['target'] = diabetes.target
                st.session_state.current_dataset = df
                st.session_state.dataset_name = "Diabetes"

def display_ai_assistant():
    load_css()
    
    st.markdown("<h1 class='main-header'>DataVue</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; margin-bottom: 30px;'>Your AI Assistant for Data Science and EDA</p>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'question' not in st.session_state:
        st.session_state.question = ""
    if 'run_query' not in st.session_state:
        st.session_state.run_query = False
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = ""
    
    # Initialize Groq client
    try:
        api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not found. Please set it in environment variables or Streamlit secrets.")
            return
        client = Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        return
    
    # Example questions
    example_questions = [
        "How do I perform data cleaning in Python?",
        "What are the best visualization libraries for EDA?",
        "Can you explain the concept of feature engineering?",
        "How do I handle missing data in a dataset?",
        "What statistical tests should I use for hypothesis testing?",
        "Explain the difference between classification and regression",
        "How do I evaluate a machine learning model?",
        "What is cross-validation and why is it important?",
        "How do I handle imbalanced datasets?",
        "What are the assumptions of linear regression?"
    ]
    
    # Display sidebar content
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1998/1998664.png", width=100)
        st.markdown("### Get Started")
        display_example_questions(example_questions)
        st.markdown("---")
        display_data_science_datasets()
        st.markdown("---")
        st.markdown("### Useful Resources")
        resources = {
            "Pandas Documentation": "https://pandas.pydata.org/docs/",
            "Scikit-learn User Guide": "https://scikit-learn.org/stable/user_guide.html",
            "Seaborn Tutorial": "https://seaborn.pydata.org/tutorial.html",
            "Kaggle Learn": "https://www.kaggle.com/learn",
            "Matplotlib Tutorial": "https://matplotlib.org/stable/tutorials/index.html",
            "Statsmodels Documentation": "https://www.statsmodels.org/stable/index.html"
        }
        st.markdown("<ul class='resources-list'>", unsafe_allow_html=True)
        for name, url in resources.items():
            st.markdown(f"<li><a href='{url}' target='_blank'>{name}</a></li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)
    
    # User input
    question = st.text_input("Ask a data science or EDA question:", 
                             value=st.session_state.question, 
                             key="question_input",
                             help="Type your question here or select an example from the sidebar")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Get Answer", key="get_answer"):
            st.session_state.run_query = True
    
    # Display current dataset if available
    if st.session_state.current_dataset is not None:
        st.subheader(f"Current Dataset: {st.session_state.dataset_name}")
        st.dataframe(st.session_state.current_dataset.head())
        
        # Basic EDA for the dataset
        with st.expander("Basic Dataset Analysis"):
            st.write(f"**Shape:** {st.session_state.current_dataset.shape}")
            st.write("**Columns:**")
            st.write(st.session_state.current_dataset.columns.tolist())
            
            st.subheader("Data Types")
            st.write(st.session_state.current_dataset.dtypes)
            
            st.subheader("Summary Statistics")
            st.write(st.session_state.current_dataset.describe())
            
            st.subheader("Missing Values")
            missing = st.session_state.current_dataset.isnull().sum()
            st.write(missing[missing > 0])
    
    if st.session_state.run_query and question:
        with st.spinner("DataVue is analyzing your question..."):
            try:
                # Construct a more specific prompt
                prompt = f"""As an AI assistant specializing in data science and exploratory data analysis, please answer the following question:

                {question}

                Please provide a comprehensive explanation, along with:
                1. Step-by-step guidance if applicable
                2. Relevant Python code examples
                3. Suggestions for libraries or tools to use
                4. Best practices and common pitfalls to avoid
                5. Recommended resources for further learning
                
                Structure your response with clear headings and bullet points for readability.
                """

                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are DataVue, an AI assistant specializing in data science and exploratory data analysis. Provide clear, concise, and practical advice with code examples when relevant."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model="llama3-70b-8192",
                    temperature=0.3,
                    max_tokens=2000
                )
                
                response = chat_completion.choices[0].message.content
                
                # Display the response
                st.markdown("<h2 class='subheader'>DataVue's Response</h2>", unsafe_allow_html=True)
                st.markdown(f"<div class='ai-response'>{response}</div>", unsafe_allow_html=True)
                
                # Add a feedback section
                st.markdown("<h3 class='subheader'>Was this response helpful?</h3>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Yes", key="feedback_yes", help="This response was helpful"):
                        st.success("Thank you for your feedback! We're glad we could help.")
                with col2:
                    if st.button("👎 No", key="feedback_no", help="This response wasn't helpful"):
                        st.info("We're sorry the response wasn't helpful. Please try rephrasing your question.")
                
                # Log the interaction
                try:
                    with open("interaction_log.txt", "a") as log_file:
                        log_file.write(f"{datetime.now()} - Question: {question}\n")
                except Exception as e:
                    st.warning(f"Couldn't save interaction log: {str(e)}")
                
                st.session_state.run_query = False
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.run_query = False

def main():
    st.set_page_config(
        page_title="DataVue - AI Data Science Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    display_ai_assistant()

if __name__ == "__main__":
    main()
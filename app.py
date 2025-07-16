import streamlit as st
import pandas as pd
import openai
from io import StringIO
import warnings
import os


# --- Helper Functions ---

def load_data(uploaded_files):
    """Loads multiple csv/xls files into a single DataFrame."""
    all_data = []
    for file in uploaded_files:
        try:
            file_extension = file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                stringio = StringIO(file.getvalue().decode("utf-8"))
                df = pd.read_csv(stringio)
            elif file_extension in ['xls', 'xlsx']:
                df = pd.read_excel(file)
            else:
                st.warning(f"Unsupported file format: {file.name}. Skipping.")
                continue
            all_data.append(df)
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")

    if not all_data:
        return None

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def preprocess_data(df):
    """
    Handles predictable data cleaning steps before handing off to the LLM.
    - Converts object columns that look like dates into datetime objects.
    """
    for col in df.select_dtypes(include=['object']).columns:
        try:
            pd.to_datetime(df[col].dropna().iloc[:5], errors='raise')
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except (ValueError, TypeError):
            pass
    return df


def get_llm_response(prompt, api_key):
    """Generic function to get a response from OpenAI's LLM."""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system",
                 "content": "You are a helpful data analysis assistant that writes clean, efficient, and modern Python pandas code."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred with the OpenAI API: {e}")
        return None


# --- Main App Logic ---

st.set_page_config(layout="wide", page_title="AI Marketing Analyst")
st.title("🤖 AI-Powered Marketing Campaign Analysis")
st.markdown("""
Upload your performance marketing data from different platforms (e.g., Google Ads, Facebook Ads). 
This app will use an AI to understand the data, generate analysis code, and create a unified performance report.
""")

# API Key input
api_key = os.getenv('OPENAI_API_KEY')

uploaded_files = st.file_uploader(
    "Upload your CSV or XLS files",
    type=['csv', 'xls', 'xlsx'],
    accept_multiple_files=True
)

if uploaded_files and api_key:
    df_raw = load_data(uploaded_files)

    if df_raw is not None:
        # Pre-process data silently
        df_processed = preprocess_data(df_raw.copy())

        if st.button("🚀 Run AI Analysis"):
            column_names = df_processed.columns.tolist()

            with st.status("Running AI Analysis...", expanded=True) as status:

                st.write("🧠 **Step 1: Generating Analysis Plan**")
                plan_prompt = f"""
                I have a pandas DataFrame with the following columns: {column_names}.
                My goal is to create a campaign performance report with these final columns:
                'Campaign', 'Spends', 'Revenue', 'ROAS', 'Impressions', 'Clicks', 'CPC', 'CTR'.
                Please create a step-by-step plan to map the source columns to the target columns, handle missing values, and calculate derived metrics like CPC, CTR, and ROAS.
                """
                analysis_plan = get_llm_response(plan_prompt, api_key)
                if not analysis_plan:
                    status.update(label="Failed to generate plan.", state="error")
                    st.stop()

                st.write("🐍 **Step 2: Generating Python Code**")

                code_gen_prompt = f"""
                Based on the following plan:
                --- PLAN ---
                {analysis_plan}
                --- END PLAN ---

                And using a pandas DataFrame named `df` with the columns: {column_names}.

                Write a Python script using the pandas library to perform the analysis.

                **Instructions & Best Practices:**
                1.  The input DataFrame is already loaded and named `df`.
                2.  **IMPORTANT:** Any date-like columns have ALREADY been converted to datetime objects. You DO NOT need to use `pd.to_datetime()`.
                3.  Your script must handle variations in column names for metrics (e.g., 'Cost' vs 'Amount Spent').
                4.  The final output MUST be a pandas DataFrame named `final_df`.
                5.  `final_df` should be grouped by campaign and contain: ['Campaign', 'Spends', 'Revenue', 'ROAS', 'Impressions', 'Clicks', 'CPC', 'CTR'].
                6.  **CRITICAL:** To avoid warnings, DO NOT use chained assignment with `inplace=True`. Use `df['column'] = df['column'].fillna(0)` instead.
                7.  Sort the results by 'Spends' in descending order and limit to top 10 campaigns.
                8.  Provide ONLY the Python code, without any explanations or markdown formatting.
                """

                generated_code = get_llm_response(code_gen_prompt, api_key)
                if not generated_code:
                    status.update(label="Failed to generate code.", state="error")
                    st.stop()

                if generated_code.startswith("```python"):
                    generated_code = generated_code[9:]
                if generated_code.endswith("```"):
                    generated_code = generated_code[:-3]

                st.write("⚙️ **Step 3: Executing Code**")

                try:
                    exec_scope = {"df": df_processed.copy(), "pd": pd}
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        exec(generated_code, exec_scope)

                    result_df = exec_scope['final_df']

                    # Ensure we only show top 10 campaigns by spend
                    if len(result_df) > 10:
                        result_df = result_df.nlargest(10, 'Spends')

                    status.update(label="Analysis Complete!", state="complete")

                except Exception as e:
                    status.update(label=f"Code Execution Error: {e}", state="error")
                    st.error(f"An error occurred while executing the generated code: {e}")
                    st.stop()

            st.balloons()

            # Show only the campaigns table
            st.header("📊 Top 10 Campaigns by Spend")
            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

elif uploaded_files and not api_key:
    st.warning("Please enter your OpenAI API key to proceed with the analysis.")
elif not uploaded_files:
    st.info("Please upload your CSV or XLS files to get started.")
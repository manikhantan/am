import streamlit as st
import pandas as pd
import json
import traceback
import io
from typing import Dict, List, Any, Optional

class CSVHandler:
    """Handles loading, cleaning, and summarizing CSV data."""

    def load_csv(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Loads a CSV file, performs validation and basic cleaning."""
        if uploaded_file.size > 100 * 1024 * 1024:  # 100MB limit
            st.error("File size exceeds the 100MB limit.")
            return None
        try:
            import chardet
            raw_data = uploaded_file.read()
            uploaded_file.seek(0)
            encoding = chardet.detect(raw_data)['encoding']
            df = pd.read_csv(io.BytesIO(raw_data), encoding=encoding)
        except Exception:
            st.error("Could not decode file. Trying standard encodings...")
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                return None

        if df.empty:
            st.error("The uploaded CSV file is empty.")
            return None

        df.columns = (df.columns.str.strip()
                      .str.replace(r'[^\w\s]', '_', regex=True)
                      .str.replace(r'\s+', '_', regex=True))
        return df

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Creates a comprehensive summary of the dataframe for the LLM."""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'sample_data': df.head(3).to_dict(orient='records'),
            'numeric_stats': df.select_dtypes(include='number').describe().to_dict()
        }


class CodeExecutor:
    """Executes Python code in a controlled environment."""

    def execute_code(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Executes the provided Python code, returning results or an error."""
        try:
            exec_env = {
                'df': df.copy(),
                'pd': pd,
                'px': __import__('plotly.express'),
                'go': __import__('plotly.graph_objects'),
                'np': __import__('numpy')
            }
            exec(code, exec_env)
            return exec_env.get('result', {})
        except Exception:
            return {'error': 'Code execution failed.', 'traceback': traceback.format_exc()}


class LLMService:
    """A unified service to interact with different LLM providers."""

    def __init__(self, provider: str, model_name: str, api_key: str):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.client = self._initialize_client()

    def _initialize_client(self):
        if not self.api_key: return None
        try:
            if self.provider == "openai":
                from openai import OpenAI
                return OpenAI(api_key=self.api_key)
            elif self.provider == "anthropic":
                from anthropic import Anthropic
                return Anthropic(api_key=self.api_key)
        except Exception as e:
            st.error(f"Failed to initialize LLM client: {e}")
            return None

    def _call_llm(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> Optional[str]:
        if not self.client:
            st.warning("LLM client not initialized. Please check your API key.")
            return None
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    response_format={"type": "json_object"} if json_mode else None,
                    temperature=0.1, max_tokens=4000
                )
                return response.choices[0].message.content
            elif self.provider == "anthropic":
                if json_mode: user_prompt += "\n\nIMPORTANT: Respond with only a valid JSON object."
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}],
                    temperature=0.1, max_tokens=4000
                )
                return response.content[0].text
        except Exception as e:
            st.error(f"LLM API call failed: {e}")
        return None

    def _clean_response(self, response_text: str, is_json: bool = False) -> Any:
        if not response_text: return None
        cleaned_text = response_text.strip()
        if is_json:
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                try:
                    return json.loads(cleaned_text[json_start:json_end])
                except json.JSONDecodeError:
                    st.error("Failed to parse JSON from LLM.")
                    return None
            return None
        else:
            if cleaned_text.startswith("```python"): cleaned_text = cleaned_text[9:]
            if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3]
            return cleaned_text.strip()

    def generate_analysis_plan(self, data_summary: Dict, user_prompt: str) -> Optional[Dict]:
        system_prompt = """
        You are an expert data analyst. Generate a sequential, step-by-step analysis plan.
        - When planning aggregations, focus on columns that represent meaningful business metrics (e.g., sales, revenue, price, quantity) rather than IDs.
        - If asked to find 'top' items (e.g., top products), plan to identify the top 5.
        - Do NOT create a separate step for data cleaning. Incorporate cleaning (like handling missing values) directly into the relevant analysis steps.
        - Return a JSON object with a 'steps' key, which is a list of dictionaries. Each step must have 'id', 'title', and 'description'.
        Example: {"steps": [{"id": "step_1", "title": "Top 5 Product Sales Analysis", "description": "Identify the top 5 products by total sales, handling any missing sales data."}]}
        """
        user_prompt = f"User Request: {user_prompt}\n\nData Summary: {json.dumps(data_summary)}"
        response = self._call_llm(system_prompt, user_prompt, json_mode=True)
        return self._clean_response(response, is_json=True)

    def generate_analysis_code(self, step: Dict, data_summary: Dict, error_feedback: Optional[str] = None) -> Optional[
        str]:
        system_prompt = """
        You are an expert Python data analyst. Write Python code for the requested analysis step.
        - The dataframe is `df`. Use pandas and plotly.
        - When identifying 'top' or 'best' items, limit the result to the top 5.
        - The final output must be a dictionary assigned to a `result` variable.
        - The `result` dictionary can contain 'summary' (str), 'data' (DataFrame), or 'visualization' (Plotly figure).
        - Return ONLY raw Python code.
        """
        user_prompt = f"Analysis Step: {step['title']} - {step['description']}\n\nData Summary: {json.dumps(data_summary)}"
        if error_feedback:
            user_prompt += f"\n\nThe previous code failed. Please fix it. Error: {error_feedback}"
        response = self._call_llm(system_prompt, user_prompt)
        return self._clean_response(response, is_json=False)

    def generate_report(self, original_prompt: str, results: List[Dict]) -> Optional[str]:
        system_prompt = "You are an expert data analyst. Synthesize the following analysis results into a concise, final report in Markdown format. Focus on summarizing the key findings."
        user_prompt = f"Original Request: {original_prompt}\n\nAnalysis Results:\n{json.dumps(results, indent=2)}"
        return self._call_llm(system_prompt, user_prompt)


# --------------------------------------------------------------------------
# 2. MAIN APPLICATION CLASS
# --------------------------------------------------------------------------

class AIAnalysisApp:
    """The main class for the Streamlit application."""

    def __init__(self):
        st.set_page_config(page_title="AI Data Analysis", layout="wide", page_icon="📊")
        self._initialize_state()
        self.csv_handler = CSVHandler()
        self.code_executor = CodeExecutor()

    def _initialize_state(self):
        defaults = {
            'df': None, 'current_plan': None, 'execution_results': [], 'visualizations': [],
            'api_key': "", 'model_provider': "OpenAI", 'model_name': "gpt-4o",
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def run(self):
        st.title("🤖 AI-Powered Data Analysis")
        self.render_sidebar()

        # UI is no longer conditional, allowing prompt entry before upload
        self.handle_data_input()

        # Display results if they exist from a previous run
        if st.session_state.execution_results:
            self._display_final_outputs()

    def render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Configuration")
            provider = st.selectbox("LLM Provider", ["OpenAI", "Anthropic"], key='model_provider')
            if provider == "OpenAI":
                st.selectbox("Model", ["gpt-4.1", "gpt-4o"], key='model_name')
            else:
                st.selectbox("Model", ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest"], key='model_name')
            st.text_input("API Key", type="password", key='api_key')
            if st.button("🔄 Clear Session"):
                for key in list(st.session_state.keys()): del st.session_state[key]
                st.rerun()

    def handle_data_input(self):
        """Handles the data upload and analysis prompt area."""
        analysis_prompt = st.text_area(
            "What would you like to analyze?",
            placeholder="e.g., 'Analyze sales trends and identify top-performing products.'",
            height=200
        )
        uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
        if uploaded_file and st.session_state.df is None:
            st.session_state.df = self.csv_handler.load_csv(uploaded_file)
        if st.session_state.df is not None:
            st.success("✅ CSV Loaded Successfully!")
            st.dataframe(st.session_state.df.head(), use_container_width=True)

        if st.button("🚀 Generate Analysis", type="primary", use_container_width=True):
            if not analysis_prompt.strip():
                st.error("Please enter your analysis goal.")
            elif st.session_state.df is None:
                st.error("Please upload a CSV file.")
            elif not st.session_state.api_key:
                st.warning("Please enter your API key in the sidebar.")
            else:
                self.run_full_analysis(analysis_prompt)
                st.rerun()  # Rerun to display the results cleanly

    def run_full_analysis(self, prompt: str):
        """Orchestrates the entire analysis pipeline from planning to execution."""
        llm = LLMService(st.session_state.model_provider, st.session_state.model_name, st.session_state.api_key)

        with st.spinner("Running full analysis... This may take a few minutes."):
            # 1. Generate Plan
            data_summary = self.csv_handler.get_data_summary(st.session_state.df)
            plan = llm.generate_analysis_plan(data_summary, prompt)
            if not plan or 'steps' not in plan:
                st.error("Failed to generate a valid analysis plan.")
                return

            # 2. Execute Plan Silently
            success = self._execute_plan_silently(llm, plan, data_summary)
            if not success:
                return  # Error is shown inside the execution method

            # 3. Generate Final Report
            report = llm.generate_report(prompt, st.session_state.execution_results)
            if report:
                st.session_state.final_report = report
            else:
                st.error("Failed to generate the final report.")

    def _execute_plan_silently(self, llm: LLMService, plan: Dict, data_summary: Dict) -> bool:
        """Executes the analysis plan, collecting results without displaying them."""
        st.session_state.execution_results = []
        st.session_state.visualizations = []

        for step in plan['steps']:
            result, _ = self._execute_single_step_with_retry(llm, step, data_summary)

            if result.get('error'):
                st.error(f"Analysis stopped due to an error in step: '{step['title']}'")
                with st.expander("Error Details"):
                    st.code(result.get('traceback', 'No traceback available.'), language='text')
                return False  # Stop execution

            # Collect results for the final report
            step_result = {'step': step['title'], 'summary': result.get('summary', 'No summary.')}
            st.session_state.execution_results.append(step_result)
            if result.get('visualization'):
                st.session_state.visualizations.append((result['visualization'], step['title']))
        return True

    def _execute_single_step_with_retry(self, llm: LLMService, step: Dict, data_summary: Dict) -> (Dict, str):
        """Generates and executes code for a step, with one retry on failure."""
        error_feedback = None
        for attempt in range(2):
            code = llm.generate_analysis_code(step, data_summary, error_feedback)
            if not code: return {'error': 'LLM failed to generate code.'}, ""

            result = self.code_executor.execute_code(code, st.session_state.df)
            if not result.get('error'): return result, code

            error_feedback = f"Error: {result['error']}\nTraceback: {result.get('traceback', '')}"
        return result, code

    def _display_final_outputs(self):
        """Renders the final report and all collected visualizations."""
        st.markdown("---")
        if st.session_state.get('final_report'):
            st.header("📋 Final Analysis Report")
            st.markdown(st.session_state.final_report)

        if st.session_state.visualizations:
            st.markdown("---")
            st.header("📊 Visualizations")
            for viz in st.session_state.visualizations:
                st.plotly_chart(viz[0], use_container_width=True, unique_key=viz[1])


# --------------------------------------------------------------------------
# 3. SCRIPT ENTRYPOINT
# --------------------------------------------------------------------------

if __name__ == "__main__":
    app = AIAnalysisApp()
    app.run()

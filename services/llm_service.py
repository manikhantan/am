import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional


class LLMService:
    def __init__(self, provider: str, model_name: str, api_key: str):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
        self._initialize_client()

    def _get_temperature(self):
        if self.model_name == 'o4-mini':
            return 1
        else:
            return 0

    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_analysis_plan(self, data_info: Dict, user_prompt: str) -> Dict:
        """Generate a comprehensive analysis plan based on data and user requirements"""

        system_prompt = """You are an expert data analyst. Generate a comprehensive analysis plan based on the provided dataset information and user requirements.

Return a JSON object with the following structure:
{
    "plan_summary": "Brief description of the overall analysis strategy",
    "steps": [
        {
            "id": "step_1",
            "title": "Step title",
            "description": "Detailed description of what this step does",
            "dependencies": ["step_id1", "step_id2"] or [],
            "estimated_time": "estimated time to complete",
            "analysis_type": "exploratory|statistical|visualization|modeling",
            "code_preview": "Brief code preview or approach",
            "expected_output": "Brief summary of expected output"
        }
    ]
}

Guidelines:
1. Create logical, sequential steps that build upon each other
2. Include data exploration, cleaning, analysis, and visualization steps
3. Consider the data types and structure when planning
4. Make steps granular enough to be executed independently
5. Include dependency information for parallel execution
6. Focus on actionable insights and recommendations"""

        user_message = f"""
Dataset Information:
- Shape: {data_info['shape']}
- Columns: {data_info['columns']}
- Data Types: {data_info['dtypes']}
- Missing Values: {data_info['missing_values']}
- Sample Data: {data_info['sample_data']}

User Requirements: {user_prompt}

Generate a comprehensive analysis plan that addresses the user's requirements while considering the dataset characteristics.
"""

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    response_format={"type": "json_object"},
                    temperature=self._get_temperature()
                )
                return json.loads(response.choices[0].message.content)

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4000,
                    temperature=self._get_temperature(),
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\n{user_message}"}
                    ]
                )
                return json.loads(response.content[0].text)

        except Exception as e:
            raise Exception(f"Failed to generate analysis plan: {str(e)}")

    def generate_analysis_code(self, step: Dict, data_sample: pd.DataFrame) -> str:
        """Generate Python code for a specific analysis step"""

        # Get data information
        data_info = {
            'columns': list(data_sample.columns),
            'dtypes': data_sample.dtypes.to_dict(),
            'shape': data_sample.shape,
            'sample': data_sample.head(3).to_dict()
        }

        system_prompt = """You are an expert Python data analyst. Generate clean, executable Python code for the given analysis step.

CRITICAL REQUIREMENTS:
1. Return ONLY executable Python code - no markdown, no explanations, no code blocks
2. FIX the specific error that occurred in the previous code
3. Use pandas, numpy, matplotlib, seaborn, and plotly for analysis
4. Assume the dataframe is available as 'df'
5. Return results in a structured format (dictionary with keys: 'summary', 'data', 'visualization', 'insights')
6. Handle missing values and data quality issues with proper error handling
7. Include comprehensive try/except blocks to prevent similar errors
8. Generate visualizations when appropriate using ONLY plotly (px or go)

VISUALIZATION REQUIREMENTS:
- Use ONLY plotly.express (px) or plotly.graph_objects (go) for visualizations
- DO NOT use matplotlib.pyplot or seaborn for final visualizations
- Ensure all visualizations are plotly Figure objects
- If visualization creation fails, set visualization to None

Code must follow this exact structure:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

try:
    # Your analysis code here
    result = {
        'summary': 'Text summary of findings',
        'data': processed_data_if_applicable,
        'visualization': plotly_figure_if_created,
        'insights': 'Key insights and recommendations'
    }
except Exception as e:
    result = {
        'summary': f'Analysis failed: {str(e)}',
        'error': str(e)
    }

DO NOT include any text before or after the code. DO NOT use markdown code blocks."""

        user_message = f"""
Analysis Step: {step['title']}
Description: {step['description']}
Analysis Type: {step['analysis_type']}

Data Information:
- Columns: {data_info['columns']}
- Shape: {data_info['shape']}
- Data Types: {data_info['dtypes']}
- Sample Data: {data_info['sample']}

Generate Python code that performs this analysis step and returns results in the specified format.
"""

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self._get_temperature(),
                    max_tokens=2000
                )
                code = response.choices[0].message.content
                return self._clean_generated_code(code)

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2000,
                    temperature=self._get_temperature(),
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\n{user_message}"}
                    ]
                )
                code = response.content[0].text
                return self._clean_generated_code(code)

        except Exception as e:
            raise Exception(f"Failed to generate analysis code: {str(e)}")

    def _clean_generated_code(self, code: str) -> str:
        """Clean the generated code to remove markdown formatting and ensure it's executable"""

        # Remove markdown code blocks
        code = code.strip()

        # Remove ```python and ``` markers
        if code.startswith('```python'):
            code = code[9:]
        elif code.startswith('```'):
            code = code[3:]

        if code.endswith('```'):
            code = code[:-3]

        # Remove any leading/trailing whitespace
        code = code.strip()

        # Ensure the code ends with the result assignment
        if 'result = ' not in code:
            code += '\nresult = {"summary": "Analysis completed", "insights": "No specific insights generated"}'

        return code

    def regenerate_code_with_error(self, step: Dict, data_sample: pd.DataFrame, previous_code: str,
                                   error_result: Dict) -> str:
        """Regenerate code after an execution error, providing error feedback to the LLM"""

        data_info = {
            'columns': data_sample.columns.tolist(),
            'dtypes': data_sample.dtypes.to_dict(),
            'shape': data_sample.shape,
            'sample': data_sample.head(3).to_dict()
        }

        system_prompt = """You are an expert Python data analyst. The previously generated code failed to execute. 

        ANALYZE THE ERROR AND GENERATE CORRECTED CODE:

        CRITICAL REQUIREMENTS:
        1. Return ONLY executable Python code - no markdown, no explanations, no code blocks
        2. FIX the specific error that occurred in the previous code
        3. Use pandas, numpy, matplotlib, seaborn, and plotly for analysis
        4. Assume the dataframe is available as 'df'
        5. Return results in a structured format (dictionary with keys: 'summary', 'data', 'visualization', 'insights')
        6. Handle missing values and data quality issues with proper error handling
        7. Include comprehensive try/except blocks to prevent similar errors
        8. Generate visualizations when appropriate using plotly

        Code must follow this exact structure:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import plotly.graph_objects as go

        try:
            # Your CORRECTED analysis code here
            # Address the specific error that occurred
            result = {
                'summary': 'Text summary of findings',
                'data': processed_data_if_applicable,
                'visualization': plotly_figure_if_created,
                'insights': 'Key insights and recommendations'
            }
        except Exception as e:
            result = {
                'summary': f'Analysis failed: {str(e)}',
                'error': str(e)
            }

        IMPORTANT: Learn from the error and implement proper error handling to prevent it from happening again."""

        error_info = error_result.get('error', 'Unknown error')
        traceback_info = error_result.get('traceback', 'No traceback available')

        user_message = f"""
        Analysis Step: {step['title']}
        Description: {step['description']}
        Expected Output: {step['expected_output']}

        Data Information:
        - Columns: {data_info['columns']}
        - Data Types: {data_info['dtypes']}
        - Shape: {data_info['shape']}
        - Sample: {data_info['sample']}

        PREVIOUS CODE THAT FAILED:
        {previous_code}

        ERROR THAT OCCURRED:
        {error_info}

        ERROR TRACEBACK:
        {traceback_info}

        Please analyze the error and generate corrected Python code that addresses the specific issue.
        Focus on fixing the root cause of the error while maintaining the original analysis objective.
        """

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self._get_temperature(),
                    max_tokens=2000
                )
                code = response.choices[0].message.content
                return self._clean_generated_code(code)

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2000,
                    temperature=self._get_temperature(),
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\n{user_message}"}
                    ]
                )
                code = response.content[0].text
                return self._clean_generated_code(code)

        except Exception as e:
            raise Exception(f"Failed to regenerate analysis code: {str(e)}")

    def generate_comprehensive_report(self, analysis_results: List[Dict], original_prompt: str) -> Dict:
        """Generate a comprehensive final report by analyzing all execution results"""

        system_prompt = """You are an expert data analyst creating a comprehensive final report. 
        Analyze the provided analysis results and create a professional, actionable report.

        Return a JSON object with the following structure:
        {
            "executive_summary": "Brief overview of key findings and recommendations",
            "sections": [
                {
                    "title": "Section Name",
                    "content": "Detailed analysis content",
                    "key_findings": ["Finding 1", "Finding 2", ...]
                }
            ],
            "recommendations": [
                {
                    "title": "Recommendation Title",
                    "description": "Detailed recommendation with actionable steps",
                    "priority": "high|medium|low"
                }
            ],
            "conclusion": "Overall conclusion and next steps"
        }

        Guidelines:
        1. Focus on actionable insights and business implications
        2. Synthesize findings across all analysis steps
        3. Provide specific, measurable recommendations
        4. Use clear, professional language
        5. Structure the report logically from findings to recommendations
        6. Include quantitative results where available"""

        # Prepare results summary for analysis
        results_text = ""
        for result in analysis_results:
            results_text += f"\n--- {result['step']} ---\n"
            results_text += f"Description: {result['description']}\n"

            if result['findings']:
                for key, value in result['findings'].items():
                    results_text += f"{key}: {value}\n"

        user_message = f"""
        Original Analysis Request: {original_prompt}

        Analysis Results:
        {results_text}

        Please generate a comprehensive final report that:
        1. Synthesizes all the analysis results
        2. Provides clear findings and insights
        3. Includes actionable recommendations
        4. Addresses the original analysis request
        5. Presents conclusions in a professional format
        """

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    response_format={"type": "json_object"},
                    temperature=self._get_temperature(),
                    max_tokens=3000
                )
                return json.loads(response.choices[0].message.content)

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=3000,
                    temperature=self._get_temperature(),
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\n{user_message}"}
                    ]
                )
                return json.loads(response.content[0].text)

        except Exception as e:
            raise Exception(f"Failed to generate comprehensive report: {str(e)}")

    def analyze_step_results(self, step: Dict, result: Dict, data: pd.DataFrame) -> str:
        """Analyze the results of a single analysis step and provide insights"""

        system_prompt = """You are an expert data analyst. Analyze the provided analysis results and provide clear, actionable insights.

        Focus on:
        1. What the results tell us about the data
        2. Key patterns, trends, or anomalies discovered
        3. Business implications of the findings
        4. Specific, quantitative insights where possible
        5. How this relates to the overall analysis goals

        Provide a clear, concise analysis in 2-3 paragraphs."""

        # Prepare result summary
        result_text = f"Step: {step['title']}\n"
        result_text += f"Description: {step['description']}\n"
        result_text += f"Analysis Type: {step['analysis_type']}\n\n"

        if 'summary' in result:
            result_text += f"Summary: {result['summary']}\n"

        if 'data' in result:
            if hasattr(result['data'], 'describe'):
                result_text += f"Data Statistics: {result['data'].describe()}\n"
            elif isinstance(result['data'], dict):
                result_text += f"Data Results: {str(result['data'])[:500]}...\n"

        if 'insights' in result:
            result_text += f"Initial Insights: {result['insights']}\n"

        # Add basic data context
        result_text += f"\nDataset Context:\n"
        result_text += f"- Total rows: {len(data)}\n"
        result_text += f"- Columns: {list(data.columns)}\n"
        result_text += f"- Data types: {data.dtypes.to_dict()}\n"

        user_message = f"""
        Analysis Results to Analyze:
        {result_text}

        Please provide a detailed analysis of these results, focusing on:
        1. What key insights can be drawn from this analysis
        2. What patterns or trends are evident
        3. What business implications these findings have
        4. How significant are these findings
        5. What questions do these results raise for further investigation
        """

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self._get_temperature(),
                    max_tokens=800
                )
                return response.choices[0].message.content

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=800,
                    temperature=self._get_temperature(),
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\n{user_message}"}
                    ]
                )
                return response.content[0].text

        except Exception as e:
            return f"Could not analyze results: {str(e)}"

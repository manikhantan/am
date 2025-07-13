import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io
import sys
import traceback
import warnings
from typing import Dict, Any
import contextlib


class CodeExecutionError(Exception):
    """Custom exception for code execution failures that carries error details"""
    def __init__(self, message, error_dict):
        super().__init__(message)
        self.error_dict = error_dict


class CodeExecutor:
    def __init__(self):
        self.execution_context = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'px': px,
            'go': go
        }

        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore')

    def execute_code(self, code: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Safely execute analysis code with the provided data"""

        # Prepare execution environment
        execution_env = self.execution_context.copy()
        execution_env['df'] = data.copy()

        # Capture stdout for any print statements
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            # Execute the code
            exec(code, execution_env)

            # Get the result
            result = execution_env.get('result', {})

            # Capture any print output
            output = captured_output.getvalue()
            if output.strip():
                result['console_output'] = output.strip()

            # Validate result structure
            if not isinstance(result, dict):
                result = {'data': result}

            # Process visualization if present
            # if 'visualization' in result and result['visualization']:
            #     result['visualization'] = self._process_visualization(result['visualization'])

            return result

        except SyntaxError as e:
            error_msg = f"Code syntax error: {str(e)}"
            error_dict = {
                'error': error_msg,
                'traceback': f"Syntax error at line {e.lineno}: {e.text}",
                'summary': f"Syntax error: {str(e)}",
                'code_preview': code[:500] + '...' if len(code) > 500 else code
            }
            # Create a custom exception that carries the error details
            raise CodeExecutionError(error_msg, error_dict)

        except Exception as e:
            error_msg = f"Code execution failed: {str(e)}"
            traceback_str = traceback.format_exc()
            error_dict = {
                'error': error_msg,
                'traceback': traceback_str,
                'summary': f"Execution failed: {str(e)}",
                'code_preview': code[:500] + '...' if len(code) > 500 else code
            }
            # Create a custom exception that carries the error details
            raise CodeExecutionError(error_msg, error_dict)

        finally:
            sys.stdout = old_stdout

    # def _process_visualization(self, viz):
    #     """Process visualization objects to ensure they're displayable"""
    #
    #     try:
    #         import plotly.graph_objects as go
    #
    #         # Check if it's already a proper plotly Figure
    #         if isinstance(viz, go.Figure):
    #             return viz
    #
    #         # Handle Matplotlib figures - convert to plotly
    #         elif hasattr(viz, 'savefig'):
    #             return self._convert_matplotlib_to_plotly(viz)
    #
    #         # Handle seaborn plots (which are matplotlib underneath)
    #         elif hasattr(viz, 'figure'):
    #             return self._convert_matplotlib_to_plotly(viz.figure)
    #
    #         # Handle dictionary representations of plotly figures
    #         elif isinstance(viz, dict):
    #             if 'data' in viz and 'layout' in viz:
    #                 try:
    #                     return go.Figure(data=viz.get('data', []), layout=viz.get('layout', {}))
    #                 except Exception:
    #                     return None
    #             else:
    #                 return None
    #
    #         # Handle list-like objects (might be plotly data)
    #         elif isinstance(viz, list):
    #             try:
    #                 return go.Figure(data=viz)
    #             except Exception:
    #                 return None
    #
    #         # Handle other plotly-like objects
    #         elif hasattr(viz, 'show') and hasattr(viz, 'data') and hasattr(viz, 'layout'):
    #             try:
    #                 # Try to convert to proper Figure by extracting data and layout
    #                 data = getattr(viz, 'data', [])
    #                 layout = getattr(viz, 'layout', {})
    #                 return go.Figure(data=data, layout=layout)
    #             except Exception:
    #                 return None
    #
    #         # Handle string representations or other invalid objects
    #         else:
    #             return None
    #
    #     except Exception as e:
    #         # Log the error but don't fail the entire analysis
    #         return None

    # def _convert_matplotlib_to_plotly(self, fig):
    #     """Convert matplotlib figure to plotly figure"""
    #
    #     try:
    #         # Try plotly's mpl_to_plotly converter first
    #         import plotly.tools as tls
    #         plotly_fig = tls.mpl_to_plotly(fig)
    #         return plotly_fig
    #     except Exception:
    #         try:
    #             # Alternative: create a simple plotly figure with the same data
    #             import plotly.graph_objects as go
    #
    #             # Create a basic plotly figure
    #             plotly_fig = go.Figure()
    #
    #             # Try to extract data from matplotlib axes
    #             if hasattr(fig, 'axes') and fig.axes:
    #                 ax = fig.axes[0]
    #                 for line in ax.get_lines():
    #                     plotly_fig.add_trace(go.Scatter(
    #                         x=line.get_xdata(),
    #                         y=line.get_ydata(),
    #                         mode='lines',
    #                         name=line.get_label() if line.get_label() else None
    #                     ))
    #
    #             return plotly_fig
    #         except Exception:
    #             # If all conversion attempts fail, return None
    #             return None

    def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate code for potential security issues and syntax errors"""

        # List of potentially dangerous operations
        dangerous_patterns = [
            'import os',
            'import subprocess',
            'import sys',
            'exec(',
            'eval(',
            'open(',
            'file(',
            '__import__',
            'globals()',
            'locals()',
            'vars()',
            'dir()',
            'getattr(',
            'setattr(',
            'delattr(',
            'hasattr(',
        ]

        # Check for dangerous patterns
        security_issues = []
        for pattern in dangerous_patterns:
            if pattern in code:
                security_issues.append(f"Potentially dangerous operation: {pattern}")

        # Check syntax
        syntax_error = None
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            syntax_error = str(e)

        return {
            'is_valid': len(security_issues) == 0 and syntax_error is None,
            'security_issues': security_issues,
            'syntax_error': syntax_error
        }

    def execute_with_timeout(self, code: str, data: pd.DataFrame, timeout: int = 300) -> Dict[str, Any]:
        """Execute code with timeout protection"""

        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            result = self.execute_code(code, data)
            signal.alarm(0)  # Cancel timeout
            return result

        except TimeoutError:
            signal.alarm(0)  # Cancel timeout
            return {
                'error': f"Code execution timed out after {timeout} seconds",
                'summary': "Execution timed out"
            }
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            return {
                'error': f"Execution failed: {str(e)}",
                'summary': f"Execution failed: {str(e)}"
            }

    def optimize_for_large_data(self, code: str, data_size: int) -> str:
        """Optimize code for large datasets"""

        if data_size < 10000:
            return code

        # Add memory optimization techniques
        optimizations = [
            "# Memory optimization for large dataset",
            "import gc",
            "gc.collect()",
            "",
            "# Use efficient data types",
            "df = df.copy()",
            "for col in df.select_dtypes(include=['object']).columns:",
            "    if df[col].nunique() < 100:",
            "        df[col] = df[col].astype('category')",
            "",
            "# Process in chunks if needed",
            "chunk_size = 10000",
            "if len(df) > chunk_size:",
            "    # Process in chunks",
            "    pass",
            "",
        ]

        # Add optimizations at the beginning of the code
        optimized_code = "\n".join(optimizations) + "\n" + code

        return optimized_code

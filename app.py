import streamlit as st
import pandas as pd
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.llm_service import LLMService
from services.analysis_engine import AnalysisEngine
from services.code_executor import CodeExecutor
from utils.csv_handler import CSVHandler
from utils.dependency_resolver import DependencyResolver
import plotly.express as px
import plotly.graph_objects as go

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'current_plan' not in st.session_state:
    st.session_state.current_plan = None
if 'execution_status' not in st.session_state:
    st.session_state.execution_status = {}
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'max_retries' not in st.session_state:
    st.session_state.max_retries = 1

st.set_page_config(
    page_title="AI Data Analysis Platform",
    page_icon="📊",
    layout="wide"
)


def main():
    st.title("🤖 AI-Powered Data Analysis Platform")
    st.markdown("Upload CSV files and let AI generate and execute comprehensive analysis plans")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        model_provider = st.selectbox(
            "Select LLM Provider:",
            ["OpenAI", "Anthropic"],
            help="Choose the AI model provider for analysis planning"
        )

        if model_provider == "OpenAI":
            model_name = st.selectbox(
                "Select OpenAI Model:",
                ["gpt-4.1", "gpt-4o", "o4-mini"],
                help="gpt-4o is the newest OpenAI model"
            )
        else:
            model_name = st.selectbox(
                "Select Anthropic Model:",
                ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-7-sonnet-latest", "claude-3-5-sonnet-20241022"],
                help="claude-sonnet-4-20250514 is the newest Anthropic model"
            )

        # API Key configuration
        st.subheader("🔑 API Keys")
        if model_provider == "OpenAI":
            api_key = st.text_input(
                "OpenAI API Key:",
                type="password",
                help="Enter your OpenAI API key"
            )
        else:
            api_key = st.text_input(
                "Anthropic API Key:",
                type="password",
                help="Enter your Anthropic API key"
            )

        if st.button("🔄 Clear Session"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    # Main content area
    st.header("📁 Data Upload & Analysis")

    handle_data_upload()

    # Show analysis planning in the same tab if data is uploaded
    if st.session_state.uploaded_data is not None:
        st.markdown("---")
        handle_analysis_planning(model_provider, model_name, api_key)


def handle_data_upload():
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file for analysis"
    )

    if uploaded_file is not None:
        try:
            # Initialize CSV handler
            csv_handler = CSVHandler()

            # Load and validate CSV
            df = csv_handler.load_csv(uploaded_file)

            if df is not None:
                st.session_state.uploaded_data = df

                # Display basic information
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")

                # Display data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Display data types and basic statistics
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Data Types")
                    dtype_df = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-null': df.count(),
                        'Null': df.isnull().sum()
                    })
                    st.dataframe(dtype_df, use_container_width=True)

                with col2:
                    st.subheader("📈 Basic Statistics")
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                    else:
                        st.info("No numeric columns found for statistics.")

                st.success("✅ CSV file uploaded and validated successfully!")

        except Exception as e:
            st.error(f"❌ Error loading CSV file: {str(e)}")


def handle_analysis_planning(model_provider, model_name, api_key):
    st.header("🧠 AI Analysis Planning")

    if not api_key:
        st.warning("⚠️ Please enter your API key in the sidebar to proceed.")
        return

    # High-level prompt input
    analysis_prompt = st.text_area(
        "Analysis Instructions:",
        placeholder="e.g., 'Analyze customer behavior patterns, identify trends, and provide recommendations for improving retention rates'",
        height=100,
        help="Describe what kind of analysis you want to perform on your data"
    )

    if st.button("🚀 Generate Analysis", type="primary"):
        if not analysis_prompt.strip():
            st.error("Please enter analysis instructions.")
            return

        try:
            # Initialize services
            llm_service = LLMService(model_provider, model_name, api_key)
            analysis_engine = AnalysisEngine(llm_service)

            # Generate analysis plan
            with st.spinner("🤖 Generating analysis plan..."):
                plan = analysis_engine.generate_analysis_plan(
                    st.session_state.uploaded_data,
                    analysis_prompt
                )

            if plan:
                st.session_state.current_plan = plan
                st.success("✅ Analysis plan generated successfully!")

                # Display the plan
                st.subheader("📋 Generated Analysis Plan")

                for i, step in enumerate(plan['steps'], 1):
                    with st.expander(f"Step {i}: {step['title']}", expanded=False):
                        st.write(f"**Description:** {step['description']}")
                        st.write(
                            f"**Dependencies:** {', '.join(step['dependencies']) if step['dependencies'] else 'None'}")
                        st.write(f"**Estimated Time:** {step['estimated_time']}")

                        if step.get('code_preview'):
                            st.code(step['code_preview'], language='python')

                # Automatically execute the plan
                st.info("🚀 Starting automated analysis execution...")
                execute_analysis_plan(llm_service)

                # Generate comprehensive report after execution
                if st.session_state.execution_status:
                    generate_final_report(llm_service, analysis_prompt)
            else:
                st.error("❌ Failed to generate analysis plan. Please check your API key and try again.")

        except Exception as e:
            st.error(f"❌ Error generating analysis plan: {str(e)}")


def execute_analysis_plan(llm_service):
    if not st.session_state.current_plan:
        st.error("No analysis plan available.")
        return

    try:
        # Initialize services
        code_executor = CodeExecutor()
        dependency_resolver = DependencyResolver()

        # Resolve dependencies and create execution order
        execution_groups = dependency_resolver.resolve_dependencies(
            st.session_state.current_plan['steps']
        )

        # Initialize execution status
        st.session_state.execution_status = {
            step['id']: {'status': 'pending', 'result': None, 'error': None}
            for step in st.session_state.current_plan['steps']
        }

        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

        total_steps = len(st.session_state.current_plan['steps'])
        completed_steps = 0

        # Execute steps in dependency groups (sequential groups, parallel within groups)
        for group_idx, group in enumerate(execution_groups):
            status_text.text(
                f"🤖 AI generating and executing group {group_idx + 1}/{len(execution_groups)} ({len(group)} steps)")

            # For thread safety with Streamlit, we'll collect results first, then update UI
            group_results = {}

            # Execute steps in current group in parallel
            with ThreadPoolExecutor(max_workers=min(len(group), 4)) as executor:
                # Submit all tasks
                future_to_step = {
                    executor.submit(
                        execute_single_step,
                        step,
                        llm_service,
                        code_executor,
                        st.session_state.uploaded_data,
                        st.session_state.get('max_retries', 1)
                    ): step
                    for step in group
                }

                # Collect results as they complete
                for future in as_completed(future_to_step):
                    step = future_to_step[future]
                    try:
                        result = future.result()

                        # Check if result indicates success or failure
                        if result.get('error'):
                            group_results[step['id']] = {
                                'status': 'failed',
                                'result': None,
                                'error': result.get('error', 'Unknown error')
                            }
                        else:
                            group_results[step['id']] = {
                                'status': 'completed',
                                'result': result,
                                'error': None
                            }

                    except Exception as e:
                        group_results[step['id']] = {
                            'status': 'failed',
                            'result': None,
                            'error': str(e)
                        }

            # Update session state and UI after all threads complete
            for step in group:
                step_id = step['id']
                if step_id in group_results:
                    result_data = group_results[step_id]

                    # Update session state
                    st.session_state.execution_status[step_id]['status'] = result_data['status']
                    st.session_state.execution_status[step_id]['result'] = result_data['result']
                    st.session_state.execution_status[step_id]['error'] = result_data['error']

                    completed_steps += 1

                    # Update progress
                    progress_bar.progress(completed_steps / total_steps)

                    # Display result
                    with results_container:
                        if result_data['status'] == 'completed':
                            display_step_result(step, result_data['result'])
                        else:
                            st.error(f"❌ Step '{step['title']}' failed: {result_data['error']}")

        status_text.text("✅ AI-powered analysis execution completed! All steps generated and executed automatically.")

    except Exception as e:
        st.error(f"❌ Error executing analysis plan: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")


def execute_single_step(step, llm_service, code_executor, data, max_retries=1):
    """Execute a single analysis step using AI code generation + automated execution with retry mechanism"""

    # Get max_retries from session state or use default
    if 'max_retries' in st.session_state:
        max_retries = st.session_state.max_retries

    generated_code = None
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # Step 1: AI generates Python code for this analysis step
            if attempt == 0:
                generated_code = llm_service.generate_analysis_code(step, data)
            else:
                # Retry with error feedback - show retry message
                st.warning(f"🔄 Retry {attempt}/{max_retries} - Regenerating code after error...")
                generated_code = llm_service.regenerate_code_with_error(
                    step, data, generated_code, last_error
                )

            # Step 2: Code executor safely runs the AI-generated code
            result = code_executor.execute_code(generated_code, data)

        except Exception as e:
            # Exception during code generation or execution
            last_error = {
                'error': f"Code generation/execution failed: {str(e)}",
                'summary': f"Failed on attempt {attempt + 1}: {str(e)}",
                'traceback': str(e)
            }
            if attempt < max_retries:
                st.warning(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
            else:
                st.error(f"❌ All {max_retries + 1} attempts failed")
            continue

        # Check if execution succeeded (no error in result) - NOW OUTSIDE try-catch
        if not result.get('error'):
            # Success! Step 3: Have LLM analyze the execution results for insights
            try:
                analysis = llm_service.analyze_step_results(step, result, data)
                result['ai_analysis'] = analysis
            except Exception as e:
                result['ai_analysis'] = f"Could not analyze results: {str(e)}"

            # Add metadata about the automated execution
            result['_metadata'] = {
                'step_id': step['id'],
                'execution_method': 'AI Code Generation + Automated Execution + AI Analysis',
                'generated_code': generated_code[:200] + '...' if len(generated_code) > 200 else generated_code,
                'attempts': attempt + 1
            }

            if attempt > 0:
                st.success(f"✅ Success after {attempt + 1} attempts!")

            return result
        else:
            # Code execution failed - store error for next retry
            last_error = result
            if attempt < max_retries:
                st.warning(f"⚠️ Attempt {attempt + 1} failed: {result.get('error', 'Unknown error')}")
            else:
                st.error(f"❌ All {max_retries + 1} attempts failed")
            continue

    # If we get here, all attempts failed
    if last_error:
        if isinstance(last_error, dict):
            last_error['_metadata'] = {
                'step_id': step['id'],
                'execution_method': 'AI Code Generation + Automated Execution (Failed)',
                'generated_code': generated_code[:200] + '...' if generated_code and len(
                    generated_code) > 200 else generated_code,
                'attempts': max_retries + 1,
                'final_failure': True
            }
        return last_error
    else:
        return {
            'error': f"Failed after {max_retries + 1} attempts",
            'summary': f"All {max_retries + 1} attempts failed for step: {step['title']}",
            '_metadata': {
                'step_id': step['id'],
                'attempts': max_retries + 1,
                'final_failure': True
            }
        }


def display_step_result(step, result):
    """Display the result of an AI-executed analysis step"""
    st.subheader(f"🤖 {step['title']} (AI-Generated & Executed)")

    if isinstance(result, dict):
        # Handle errors first
        if 'error' in result:
            st.error(f"❌ {result['error']}")
            if 'traceback' in result:
                with st.expander("🔍 Error Details", expanded=False):
                    st.code(result['traceback'], language='text')
            if 'code_preview' in result:
                with st.expander("🔍 Generated Code (with error)", expanded=False):
                    st.code(result['code_preview'], language='python')
            return

        # Handle successful results
        if 'summary' in result:
            st.write(result['summary'])

        if 'data' in result:
            if isinstance(result['data'], pd.DataFrame):
                st.dataframe(result['data'], use_container_width=True)
            else:
                st.write(result['data'])

        if 'visualization' in result:
            # Create unique key using session state counter
            if 'viz_counter' not in st.session_state:
                st.session_state.viz_counter = 0
            st.session_state.viz_counter += 1
            unique_key = f"viz_{step['id']}_{st.session_state.viz_counter}"
            if isinstance(result['visualization'], dict):
                for key, chart in result['visualization'].items():
                    st.plotly_chart(chart, use_container_width=False, key=f"{unique_key}_{key}")
            elif isinstance(result['visualization'], list):
                for i, chart in enumerate(result['visualization']):
                    st.plotly_chart(chart, use_container_width=False, key=f"{unique_key}_{i}")

            else:
                st.plotly_chart(result['visualization'], use_container_width=True, key=unique_key)

        if 'insights' in result:
            st.info(f"💡 **Initial Insights:** {result['insights']}")

        # Show AI analysis of results
        if 'ai_analysis' in result:
            st.success("🤖 **AI Analysis of Results:**")
            st.write(result['ai_analysis'])

        # Show code generation metadata if available
        if '_metadata' in result:
            with st.expander("🔍 View AI-Generated Code", expanded=False):
                st.code(result['_metadata']['generated_code'], language='python')

                # Show retry information
                attempts = result['_metadata'].get('attempts', 1)
                if attempts > 1:
                    st.caption(f"✅ Code generated and executed successfully after {attempts} attempts")
                else:
                    st.caption("✅ Code generated and executed successfully on first attempt")

                if result['_metadata'].get('final_failure'):
                    st.error(f"❌ Failed after {attempts} attempts")

    else:
        st.write(result)


def generate_final_report(llm_service, original_prompt):
    """Generate a comprehensive final report by analyzing all execution results"""
    st.markdown("---")
    st.header("📋 Comprehensive Analysis Report")

    # Collect all successful results
    successful_results = []
    for step_id, status in st.session_state.execution_status.items():
        if status['status'] == 'completed' and status['result']:
            step = next((s for s in st.session_state.current_plan['steps'] if s['id'] == step_id), None)
            if step:
                successful_results.append({
                    'step_title': step['title'],
                    'step_description': step['description'],
                    'result': status['result']
                })

    if not successful_results:
        st.warning("No successful analysis results to generate report from.")
        return

    # Prepare results summary for LLM
    results_summary = []
    for result in successful_results:
        summary = {
            'step': result['step_title'],
            'description': result['step_description'],
            'findings': {}
        }

        if isinstance(result['result'], dict):
            if 'summary' in result['result']:
                summary['findings']['summary'] = result['result']['summary']
            if 'insights' in result['result']:
                summary['findings']['insights'] = result['result']['insights']
            if 'ai_analysis' in result['result']:
                summary['findings']['ai_analysis'] = result['result']['ai_analysis']
            if 'data' in result['result'] and hasattr(result['result']['data'], 'describe'):
                summary['findings']['data_stats'] = str(result['result']['data'].describe())

        results_summary.append(summary)

    # Generate comprehensive report using LLM
    try:
        with st.spinner("🤖 Generating comprehensive analysis report..."):
            report = llm_service.generate_comprehensive_report(results_summary, original_prompt)

        if report:
            st.success("✅ Comprehensive analysis report generated!")

            # Display executive summary prominently
            if report.get('executive_summary'):
                st.subheader("📋 Executive Summary")
                st.info(report['executive_summary'])

            # Display report sections
            for section in report.get('sections', []):
                st.subheader(f"📊 {section['title']}")
                st.write(section['content'])

                if section.get('key_findings'):
                    st.info("🔍 Key Findings:")
                    for finding in section['key_findings']:
                        st.write(f"• {finding}")

            # Display recommendations with priority indicators
            if report.get('recommendations'):
                st.subheader("💡 Recommendations")
                for rec in report['recommendations']:
                    priority_emoji = "🔴" if rec.get('priority') == 'high' else "🟡" if rec.get(
                        'priority') == 'medium' else "🟢"
                    st.write(f"{priority_emoji} **{rec['title']}**: {rec['description']}")

            # Display conclusion
            if report.get('conclusion'):
                st.subheader("🎯 Conclusion")
                st.write(report['conclusion'])

    except Exception as e:
        st.error(f"❌ Error generating comprehensive report: {str(e)}")


def handle_results_display():
    st.header("📊 Analysis Results")

    if not st.session_state.execution_status:
        st.info("No analysis results available yet. Please execute an analysis plan first.")
        return

    # Display execution summary
    total_steps = len(st.session_state.execution_status)
    completed_steps = sum(1 for status in st.session_state.execution_status.values() if status['status'] == 'completed')
    failed_steps = sum(1 for status in st.session_state.execution_status.values() if status['status'] == 'failed')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Steps", total_steps)
    with col2:
        st.metric("Completed", completed_steps)
    with col3:
        st.metric("Failed", failed_steps)

    # Display detailed results
    for step_id, status in st.session_state.execution_status.items():
        if status['status'] == 'completed' and status['result']:
            # Find the step details
            step = next((s for s in st.session_state.current_plan['steps'] if s['id'] == step_id), None)
            if step:
                display_step_result(step, status['result'])
        elif status['status'] == 'failed':
            st.error(f"❌ Step failed: {status['error']}")

    # Export results
    if completed_steps > 0:
        if st.button("📥 Export Results"):
            try:
                export_data = {
                    'plan': st.session_state.current_plan,
                    'results': st.session_state.execution_status
                }
                st.download_button(
                    label="Download Results (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"analysis_results_{int(time.time())}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"❌ Error exporting results: {str(e)}")


if __name__ == "__main__":
    main()

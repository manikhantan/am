import streamlit as st
import numpy as np
import pandas as pd
import openai
from io import StringIO
import warnings
import os

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="AI Marketing Analyst")

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
    Handles predictable data cleaning steps.
    - Converts object columns that look like dates into datetime objects.
    - Returns the DataFrame and the name of the primary date column found.
    """
  date_col_found = None
  for col in df.select_dtypes(include=['object']).columns:
    try:
      # Attempt to convert a sample to see if it's a date column
      pd.to_datetime(df[col].dropna().iloc[:5], errors='raise')
      df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
      # If conversion is successful for a significant portion, consider it the date column
      if df[col].notna().sum() > len(df) / 2:
        date_col_found = col
    except (ValueError, TypeError, IndexError):
      # This column is not a date column, pass
      pass
  return df, date_col_found


def get_llm_response(prompt, api_key):
  """Generic function to get a response from OpenAI's LLM."""
  try:
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role":
            "system",
            "content":
            "You are a world-class data analysis assistant. You write clean, efficient, and modern Python pandas code. Most importantly, you're great at following instructions. You also provide expert marketing analysis insights based on data."
        }, {
            "role": "user",
            "content": prompt
        }],
        temperature=0)
    return response.choices[0].message.content.strip()
  except Exception as e:
    st.error(f"An error occurred with the OpenAI API: {e}")
    return None


def prepare_data_info(data: pd.DataFrame):
  info = {
      'shape': data.shape,
      'columns': list(data.columns),
      'dtypes': data.dtypes.astype(str).to_dict(),
      'missing_values': data.isnull().sum().to_dict(),
      'sample_data': data.head(3).to_dict()
  }

  # Statistical information for numeric columns
  numeric_cols = data.select_dtypes(include=['number']).columns
  if len(numeric_cols) > 0:
    info['numeric_stats'] = data[numeric_cols].describe().to_dict()

  # Categorical information
  categorical_cols = data.select_dtypes(include=['object', 'category']).columns
  if len(categorical_cols) > 0:
    info['categorical_info'] = {}
    for col in categorical_cols:
      info['categorical_info'][col] = {
          'unique_values': data[col].nunique(),
          'top_values': data[col].value_counts().head(5).to_dict()
      }

  # Data quality assessment
  info['data_quality'] = {
      'total_missing':
      data.isnull().sum().sum(),
      'missing_percentage':
      (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100,
      'duplicate_rows':
      data.duplicated().sum(),
      'memory_usage':
      data.memory_usage(deep=True).sum()
  }

  return info


# --- Initialize Session State ---
if 'analysis_complete' not in st.session_state:
  st.session_state.analysis_complete = False
if 'campaign_insights' not in st.session_state:
  st.session_state.campaign_insights = {}
if 'adgroup_insights' not in st.session_state:
  st.session_state.adgroup_insights = {}
if 'creative_insights' not in st.session_state:
  st.session_state.creative_insights = {}
if 'result_df' not in st.session_state:
  st.session_state.result_df = None
if 'adgroup_df' not in st.session_state:
  st.session_state.adgroup_df = None
if 'creative_df' not in st.session_state:
  st.session_state.creative_df = None
if 'time_series_df' not in st.session_state:
  st.session_state.time_series_df = None
if 'adgroup_time_series_df' not in st.session_state:
  st.session_state.adgroup_time_series_df = None
if 'creative_time_series_df' not in st.session_state:
  st.session_state.creative_time_series_df = None
if 'df_processed' not in st.session_state:
  st.session_state.df_processed = None
if 'selected_deepdive_tab' not in st.session_state:
  st.session_state.selected_deepdive_tab = 0

# --- Main App UI ---

st.title("🤖 AI-Powered Facebook Marketing Analysis")
st.markdown("""
Upload your performance marketing data. 
The AI will unify the data, create a performance summary, and allow you to perform a deep-dive trend analysis for each campaign, ad group, and creative.
""")

api_key = st.secrets["api"]["key"]
uploaded_files = st.file_uploader("Upload your CSV or XLS files",
                                  type=['csv', 'xls', 'xlsx'],
                                  accept_multiple_files=True)

if uploaded_files:
  st.session_state.analysis_complete = False  # Reset on new run
  df_raw = load_data(uploaded_files)

  if df_raw is not None:
    st.session_state.df_processed, date_col = preprocess_data(df_raw.copy())

    if st.button("🚀 Run AI Analysis"):
      data_info = prepare_data_info(st.session_state.df_processed)

      with st.status("Running AI Analysis...", expanded=True) as status:
        # --- Step 1: Generate Summary Code ---
        status.update(label="🧠 Step 1: Generating Campaign Summary Code...")
        summary_code_prompt = f"""
                Dataframe `df` info: {data_info}.
                Write a Python script to create a campaign performance summary.

                **Instructions:**
                1. The input DataFrame is `df`. Date-like columns are already datetime objects.
                2. Only use the columns provided in the info above, do not attempt to infer alternate names for metrics.
                3. Don't use proxies if the specific column is not available. Use NAN or 0 if unavailable.
                4. The final output MUST be a pandas DataFrame named `final_df`.
                5. `final_df` must be grouped by campaign and contain: ['Campaign', 'Spends', 'Revenue', 'ROAS', 'Impressions', 'Clicks', 'CPC', 'CTR'].
                6. Calculate derived metrics: ROAS (Revenue/Spends), CPC (Spends/Clicks), CTR (Clicks/Impressions * 100). Handle division by zero gracefully (fill with 0).
                7. When using groupby().agg(), only include columns that actually exist in the DataFrame. For missing columns, add them after aggregation with default values (0 or NaN).
                8. Use proper pandas aggregation syntax: either dictionary format {{'new_col': ('source_col', 'function')}} or named aggregation with pd.NamedAgg().
                9. Sort `final_df` by 'Spends' in descending order.
                10. Provide ONLY the Python code, without any explanations or markdown.
                """
        summary_code = get_llm_response(summary_code_prompt, api_key)
        if not summary_code:
          status.update(label="Failed to generate campaign summary code.",
                        state="error",
                        expanded=True)
          st.stop()
        summary_code = summary_code.replace("```python", "").replace("```", "")

        # --- Step 2: Generate Ad Group Summary Code ---
        status.update(label="🧠 Step 2: Generating Ad Group Summary Code...")
        adgroup_code_prompt = f"""
                Dataframe `df` info: {data_info}.
                Write a Python script to create an ad group performance summary.

                **Instructions:**
                1. The input DataFrame is `df`. Date-like columns are already datetime objects.
                2. Only use the columns provided in the info above, do not attempt to infer alternate names for metrics.
                3. Don't use proxies if the specific column is not available. Use NAN or 0 if unavailable.
                4. The final output MUST be a pandas DataFrame named `adgroup_df`.
                5. `adgroup_df` must be grouped by ad group and contain: ['Ad Group', 'Spends', 'Revenue', 'ROAS', 'Impressions', 'Clicks', 'CPC', 'CTR'].
                6. Calculate derived metrics: ROAS (Revenue/Spends), CPC (Spends/Clicks), CTR (Clicks/Impressions * 100). Handle division by zero gracefully (fill with 0).
                7. When using groupby().agg(), only include columns that actually exist in the DataFrame. For missing columns, add them after aggregation with default values (0 or NaN).
                8. Use proper pandas aggregation syntax: either dictionary format {{'new_col': ('source_col', 'function')}} or named aggregation with pd.NamedAgg().
                9. Sort `adgroup_df` by 'Spends' in descending order.
                10. Provide ONLY the Python code, without any explanations or markdown.
                """
        adgroup_code = get_llm_response(adgroup_code_prompt, api_key)
        if not adgroup_code:
          status.update(label="Failed to generate ad group summary code.",
                        state="error",
                        expanded=True)
          st.stop()
        adgroup_code = adgroup_code.replace("```python", "").replace("```", "")

        # --- Step 3: Generate Creative Summary Code ---
        status.update(label="🧠 Step 3: Generating Creative Summary Code...")
        creative_code_prompt = f"""
                Dataframe `df` info: {data_info}.
                Write a Python script to create a creative performance summary.

                **Instructions:**
                1. The input DataFrame is `df`. Date-like columns are already datetime objects.
                2. Only use the columns provided in the info above, do not attempt to infer alternate names for metrics.
                3. Don't use proxies if the specific column is not available. Use NAN or 0 if unavailable.
                4. The final output MUST be a pandas DataFrame named `creative_df`.
                5. `creative_df` must be grouped by creative and contain: ['Creative', 'Spends', 'Revenue', 'ROAS', 'Impressions', 'Clicks', 'CPC', 'CTR'].
                6. Calculate derived metrics: ROAS (Revenue/Spends), CPC (Spends/Clicks), CTR (Clicks/Impressions * 100). Handle division by zero gracefully (fill with 0).
                7. When using groupby().agg(), only include columns that actually exist in the DataFrame. For missing columns, add them after aggregation with default values (0 or NaN).
                8. Use proper pandas aggregation syntax: either dictionary format {{'new_col': ('source_col', 'function')}} or named aggregation with pd.NamedAgg().
                9. Sort `creative_df` by 'Spends' in descending order.
                10. If no creative column is found, set creative_df = None.
                11. Provide ONLY the Python code, without any explanations or markdown.
                """
        creative_code = get_llm_response(creative_code_prompt, api_key)
        if not creative_code:
          status.update(label="Failed to generate creative summary code.",
                        state="error",
                        expanded=True)
          st.stop()
        creative_code = creative_code.replace("```python",
                                              "").replace("```", "")

        # --- Step 4: Generate Time-Series Code ---
        status.update(
            label="🧠 Step 4: Generating Campaign Time-Series Analysis Code...")
        if not date_col:
          st.warning(
              "Could not confidently identify a primary date column for trend analysis. Skipping time-series."
          )
          time_series_code = "time_series_df = None"
          adgroup_time_series_code = "adgroup_time_series_df = None"
          creative_time_series_code = "creative_time_series_df = None"
        else:
          # Decide on Week vs Month analysis
          date_range = st.session_state.df_processed[date_col].max(
          ) - st.session_state.df_processed[date_col].min()
          time_period = 'W' if date_range.days <= 90 else 'M'
          period_name = "Weekly" if time_period == 'W' else "Monthly"
          st.write(
              f"ℹ️ Data spans {date_range.days} days. Using {period_name} trend analysis."
          )

          time_series_code_prompt = f"""
                    Dataframe`df` info: {data_info}.
                    The primary date column is `{date_col}`.
                    Write a Python script to create a time-series performance breakdown by campaign.

                    **Instructions:**
                    1. The input DataFrame is `df`. `{date_col}` is already a datetime object.
                    2. Only use the columns provided in the info above, do not attempt to infer alternate names for metrics.
                    3. Don't use other columns as proxies if the specific column is not available.
                    4. Group the data by Campaign and by time period ('{time_period}' for {period_name}).
                    5. For each group, calculate: Sum of Spends, Sum of Revenue, Sum of Impressions, Sum of Clicks. If Clicks column doesn't exist, set Clicks to None/NaN. Use agg method to aggregate.
                    6. From the aggregated values, calculate: ROAS, CPC, and CTR for each group. If Clicks is None/NaN, set CPC and CTR to None/NaN. Handle division by zero.
                    7. When using groupby().agg(), only include columns that actually exist in the DataFrame. For missing columns, add them after aggregation with default values (0 or NaN).
                    8. Use proper pandas aggregation syntax: either dictionary format {{'new_col': ('source_col', 'function')}} or named aggregation with pd.NamedAgg().
                    9. The final output MUST be a pandas DataFrame named `time_series_df`.
                    10. `time_series_df` must have columns like: 'Date', 'Campaign', 'Spends', 'Revenue', 'ROAS', 'Impressions', 'Clicks', 'CPC', 'CTR'. Set CPC and CTR to None/NaN if Clicks column doesn't exist.
                    11. Provide ONLY the Python code, without any explanations or markdown.
                    """
          time_series_code = get_llm_response(time_series_code_prompt, api_key)
          if not time_series_code:
            status.update(
                label="Failed to generate campaign time-series code.",
                state="error",
                expanded=True)
            st.stop()
          time_series_code = time_series_code.replace("```python",
                                                      "").replace("```", "")

          # --- Step 5: Generate Ad Group Time-Series Code ---
          status.update(
              label="🧠 Step 5: Generating Ad Group Time-Series Analysis Code..."
          )
          adgroup_time_series_code_prompt = f"""
                    Dataframe`df` info: {data_info}.
                    The primary date column is `{date_col}`.
                    Write a Python script to create a time-series performance breakdown by ad group.

                    **Instructions:**
                    1. The input DataFrame is `df`. `{date_col}` is already a datetime object.
                    2. Only use the columns provided in the info above, do not attempt to infer alternate names for metrics.
                    3. Don't use other columns as proxies if the specific column is not available.
                    4. Group the data by Ad Group, and by time period ('{time_period}' for {period_name}).
                    5. For each group, calculate: Sum of Spends, Sum of Revenue, Sum of Impressions, Sum of Clicks. If Clicks column doesn't exist, set Clicks to None/NaN. Use agg method to aggregate.
                    6. From the aggregated values, calculate: ROAS, CPC, and CTR for each group. If Clicks is None/NaN, set CPC and CTR to None/NaN. Handle division by zero.
                    7. When using groupby().agg(), only include columns that actually exist in the DataFrame. For missing columns, add them after aggregation with default values (0 or NaN).
                    8. Use proper pandas aggregation syntax: either dictionary format {{'new_col': ('source_col', 'function')}} or named aggregation with pd.NamedAgg().
                    9. The final output MUST be a pandas DataFrame named `adgroup_time_series_df`.
                    10. must have columns like: 'Date', 'Ad Group', 'Spends', 'Revenue', 'ROAS', 'Impressions', 'Clicks', 'CPC', 'CTR'. Set CPC and CTR to None/NaN if Clicks column doesn't exist.
                    11. Provide ONLY the Python code, without any explanations or markdown.
                    """
          adgroup_time_series_code = get_llm_response(
              adgroup_time_series_code_prompt, api_key)
          if not adgroup_time_series_code:
            status.update(
                label="Failed to generate ad group time-series code.",
                state="error",
                expanded=True)
            st.stop()
          adgroup_time_series_code = adgroup_time_series_code.replace(
              "```python", "").replace("```", "")

          # --- Step 6: Generate Creative Time-Series Code ---
          status.update(
              label="🧠 Step 6: Generating Creative Time-Series Analysis Code..."
          )
          creative_time_series_code_prompt = f"""
                    Dataframe`df` info: {data_info}.
                    The primary date column is `{date_col}`.
                    Write a Python script to create a time-series performance breakdown by creative.

                    **Instructions:**
                    1. The input DataFrame is `df`. `{date_col}` is already a datetime object.
                    2. Only use the columns provided in the info above, do not attempt to infer alternate names for metrics.
                    3. Don't use other columns as proxies if the specific column is not available.
                    4. Group the data by Creative and by time period ('{time_period}' for {period_name}).
                    5. For each group, calculate: Sum of Spends, Sum of Revenue, Sum of Impressions, Sum of Clicks. If Clicks column doesn't exist, set Clicks to None/NaN. Use agg method to aggregate.
                    6. From the aggregated values, calculate: ROAS, CPC, and CTR for each group. If Clicks is None/NaN, set CPC and CTR to None/NaN. Handle division by zero.
                    7. When using groupby().agg(), only include columns that actually exist in the DataFrame. For missing columns, add them after aggregation with default values (0 or NaN).
                    8. Use proper pandas aggregation syntax: either dictionary format {{'new_col': ('source_col', 'function')}} or named aggregation with pd.NamedAgg().
                    9. The final output MUST be a pandas DataFrame named `creative_time_series_df`.
                    10. `creative_time_series_df` must have columns like: 'Date', 'Creative', 'Spends', 'Revenue', 'ROAS', 'Impressions', 'Clicks', 'CPC', 'CTR'. Set CPC and CTR to None/NaN if Clicks column doesn't exist.
                    11. If no creative column is found, set creative_time_series_df = None.
                    12. Provide ONLY the Python code, without any explanations or markdown.
                    """
          creative_time_series_code = get_llm_response(
              creative_time_series_code_prompt, api_key)
          if not creative_time_series_code:
            status.update(
                label="Failed to generate creative time-series code.",
                state="error",
                expanded=True)
            st.stop()
          creative_time_series_code = creative_time_series_code.replace(
              "```python", "").replace("```", "")

        # --- Step 7: Execute Generated Code ---
        status.update(label="⚙️ Step 7: Executing AI-Generated Code...")
        try:
          exec_scope = {
              "df": st.session_state.df_processed.copy(),
              "pd": pd,
              "np": np
          }
          with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(summary_code, exec_scope)
            exec(adgroup_code, exec_scope)
            exec(creative_code, exec_scope)
            exec(time_series_code, exec_scope)
            exec(adgroup_time_series_code, exec_scope)
            exec(creative_time_series_code, exec_scope)

          st.session_state.result_df = exec_scope.get('final_df')
          st.session_state.adgroup_df = exec_scope.get('adgroup_df')
          st.session_state.creative_df = exec_scope.get('creative_df')
          st.session_state.time_series_df = exec_scope.get('time_series_df')
          st.session_state.adgroup_time_series_df = exec_scope.get(
              'adgroup_time_series_df')
          st.session_state.creative_time_series_df = exec_scope.get(
              'creative_time_series_df')
          st.session_state.analysis_complete = True
          status.update(label="Analysis Complete!",
                        state="complete",
                        expanded=False)

        except Exception as e:
          status.update(label=f"Code Execution Error: {e}",
                        state="error",
                        expanded=True)
          st.error(
              f"An error occurred while executing the generated code: {e}")
          st.code(summary_code, language="python")
          st.code(adgroup_code, language="python")
          st.code(creative_code, language="python")
          st.code(time_series_code, language="python")
          st.code(adgroup_time_series_code, language="python")
          st.code(creative_time_series_code, language="python")
          st.stop()

# --- Display Results ---

if st.session_state.result_df is None and not uploaded_files:
  st.info("Please upload your data to get started.")

if st.session_state.result_df is not None and not st.session_state.result_df.empty:
  st.header("📊 Campaign Performance Summary")
  st.dataframe(st.session_state.result_df.head(10).style.format({
      'Spends':
      '{:,.2f}',
      'Revenue':
      '{:,.2f}',
      'ROAS':
      '{:.2f}x',
      'CPC':
      '{:,.2f}',
      'CTR':
      '{:.2f}%'
  }),
               use_container_width=True,
               hide_index=True)
  st.caption("Showing top 10 campaigns by spend.")

  # Display Ad Group Summary if available
  if st.session_state.adgroup_df is not None and not st.session_state.adgroup_df.empty:
    st.header("📊 Ad Group Performance Summary")
    st.dataframe(st.session_state.adgroup_df.head(10).style.format({
        'Spends':
        '{:,.2f}',
        'Revenue':
        '{:,.2f}',
        'ROAS':
        '{:.2f}x',
        'CPC':
        '{:,.2f}',
        'CTR':
        '{:.2f}%'
    }),
                 use_container_width=True,
                 hide_index=True)
    st.caption("Showing top 10 ad groups by spend.")

  # Display Creative Summary if available
  if st.session_state.creative_df is not None and not st.session_state.creative_df.empty:
    st.header("🎨 Creative Performance Summary")
    st.dataframe(st.session_state.creative_df.head(10).style.format({
        'Spends':
        '{:,.2f}',
        'Revenue':
        '{:,.2f}',
        'ROAS':
        '{:.2f}x',
        'CPC':
        '{:,.2f}',
        'CTR':
        '{:.2f}%'
    }),
                 use_container_width=True,
                 hide_index=True)
    st.caption("Showing top 10 creatives by spend.")

  st.divider()

  # --- Deep Dive Analysis Section ---
  st.header("🔎 Deep Dive & Trend Analysis")
  # Create tabs for Campaign, Ad Group, and Creative analysis
  tabs = ["📈 Campaign Analysis", "🎯 Ad Group Analysis", "🎨 Creative Analysis"]

  selected = st.radio("Deep dive!",
                      tabs,
                      horizontal=True,
                      index=st.session_state.selected_deepdive_tab)

  if selected != tabs[st.session_state.selected_deepdive_tab]:
    st.session_state.selected_deepdive_tab = tabs.index(selected)
    st.rerun()

  if st.session_state.selected_deepdive_tab == 0:
    if st.session_state.time_series_df is not None and not st.session_state.time_series_df.empty:
      st.subheader("Campaign Deep Dive")

      campaign_list = st.session_state.result_df.head(10)['Campaign'].unique()
      selected_campaign = st.selectbox("Select a Campaign to Analyze:",
                                       campaign_list,
                                       key="campaign_select")

      if selected_campaign:
        campaign_time_data = st.session_state.time_series_df[
            st.session_state.time_series_df['Campaign'] ==
            selected_campaign].sort_values(by='Date')

        if campaign_time_data.empty:
          st.warning(
              f"No time-series data available for campaign: {selected_campaign}"
          )
        else:
          # --- AI Insights ---
          with st.spinner("🤖 AI is analyzing campaign trends..."):
            if selected_campaign in st.session_state.campaign_insights:
              insights = st.session_state.campaign_insights[selected_campaign]
              st.info("AI-Powered Campaign Insights: (cached)")
              st.write(insights)
            else:
              insight_prompt = f"""
                            I am analyzing the performance of the marketing campaign: "{selected_campaign}".
                            Here is the time-series data for it:
                            {campaign_time_data.to_dict('records')}

                            Please provide a concise analysis based on the following framework:
                            1.  **ROAS Trend:** First, describe the overall trend of ROAS. Is it increasing, decreasing, or volatile?
                            2.  **Diagnostic Analysis:** Explain the 'why' behind the ROAS trend by looking at CTR and CPC.
                                - If ROAS is down, is it because CTR dropped (suggesting creative/audience fatigue) or because CPC increased (suggesting higher competition or CPM)?
                                - If ROAS is up, what is driving it? Higher CTR or lower CPC?
                            3.  **Actionable Recommendations:** Based on your analysis, suggest 1-2 clear, actionable next steps. For example: 'Refresh ad creatives,' 'Analyze audience saturation,' or 'Investigate CPM spikes in specific placements.'

                            Present the output in clear, easy-to-read markdown.
                            """
              insights = get_llm_response(insight_prompt, api_key)
              if insights:
                # Remove the markdown code block fences if they exist
                insights = insights.strip()
                if insights.startswith("```markdown"):
                  insights = insights[11:]
                elif insights.startswith("```"):
                  insights = insights[3:]
                if insights.endswith("```"):
                  insights = insights[:-3]
                insights = insights.strip()
              st.session_state.campaign_insights[selected_campaign] = insights
              st.info("AI-Powered Campaign Insights:")
              st.write(insights)

          st.subheader(
              f"Performance Trends for Campaign '{selected_campaign}'")

          col1, col2 = st.columns(2)
          with col1:
            st.metric("Total Spend",
                      f"{campaign_time_data['Spends'].sum():,.2f}")
            st.metric("Total Revenue",
                      f"{campaign_time_data['Revenue'].sum():,.2f}")
          with col2:
            avg_roas = campaign_time_data['Revenue'].sum() / campaign_time_data['Spends'].sum() if \
                campaign_time_data['Spends'].sum() > 0 else 0
            st.metric("Overall ROAS", f"{avg_roas:.2f}x")

          st.write("##### ROAS Trend")
          st.line_chart(campaign_time_data, x='Date', y='ROAS')

          st.write("##### Diagnostic Metrics: Clicks, CTR & CPC")
          col1, col2, col3 = st.columns(3)
          with col1:
            st.line_chart(campaign_time_data,
                          x='Date',
                          y='Clicks',
                          color="#FF4B4B")
          with col2:
            st.line_chart(campaign_time_data,
                          x='Date',
                          y='CTR',
                          color="#4BFF4B")
          with col3:
            st.line_chart(campaign_time_data,
                          x='Date',
                          y='CPC',
                          color="#4B4BFF")

    else:
      st.warning(
          "Campaign time-series analysis could not be performed. This may be due to a lack of a clear date column in the data or an error during code generation."
      )

  elif st.session_state.selected_deepdive_tab == 1:
    if st.session_state.adgroup_time_series_df is not None and not st.session_state.adgroup_time_series_df.empty:
      st.subheader("Ad Group Deep Dive")

      adgroup_list = st.session_state.adgroup_df.head(10)['Ad Group'].unique()
      selected_adgroup = st.selectbox("Select an Ad Group to Analyze:",
                                      adgroup_list,
                                      key="adgroup_select")

      if selected_adgroup:
        adgroup_time_data = st.session_state.adgroup_time_series_df[
            st.session_state.adgroup_time_series_df['Ad Group'] ==
            selected_adgroup].sort_values(by='Date')

        if adgroup_time_data.empty:
          st.warning(
              f"No time-series data available for ad group: {selected_adgroup}"
          )
        else:
          # --- AI Insights ---
          with st.spinner("🤖 AI is analyzing ad group trends..."):
            if selected_adgroup in st.session_state.adgroup_insights:
              insights = st.session_state.adgroup_insights[selected_adgroup]
              st.info("AI-Powered Ad Group Insights: (cached)")
              st.write(insights)
            else:
              insight_prompt = f"""
                            I am analyzing the performance of the ad group: "{selected_adgroup}".
                            Here is the time-series data for it:
                            {adgroup_time_data.to_dict('records')}

                            Please provide a concise analysis based on the following framework:
                            1.  **ROAS Trend:** First, describe the overall trend of ROAS. Is it increasing, decreasing, or volatile?
                            2.  **Diagnostic Analysis:** Explain the 'why' behind the ROAS trend by looking at CTR and CPC.
                                - If ROAS is down, is it because CTR dropped (suggesting creative/audience fatigue) or because CPC increased (suggesting higher competition or CPM)?
                                - If ROAS is up, what is driving it? Higher CTR or lower CPC?
                            3.  **Ad Group Specific Recommendations:** Based on your analysis, suggest 1-2 clear, actionable next steps specific to ad group optimization. For example: 'Test new ad creatives for this ad group,' 'Adjust bidding strategy,' 'Refine audience targeting,' or 'Pause underperforming keywords.'

                            Present the output in clear, easy-to-read markdown.
                            """
              insights = get_llm_response(insight_prompt, api_key)
              if insights:
                # Remove the markdown code block fences if they exist
                insights = insights.strip()
                if insights.startswith("```markdown"):
                  insights = insights[11:]
                elif insights.startswith("```"):
                  insights = insights[3:]
                if insights.endswith("```"):
                  insights = insights[:-3]
                insights = insights.strip()
              st.session_state.adgroup_insights[selected_adgroup] = insights
              st.info("AI-Powered Ad Group Insights:")
              st.write(insights)

          st.subheader(f"Performance Trends for Ad Group '{selected_adgroup}'")

          col1, col2 = st.columns(2)
          with col1:
            st.metric("Total Spend",
                      f"{adgroup_time_data['Spends'].sum():,.2f}")
            st.metric("Total Revenue",
                      f"{adgroup_time_data['Revenue'].sum():,.2f}")
          with col2:
            avg_roas = adgroup_time_data['Revenue'].sum() / adgroup_time_data['Spends'].sum() if \
                adgroup_time_data['Spends'].sum() > 0 else 0
            st.metric("Overall ROAS", f"{avg_roas:.2f}x")

          st.write("##### ROAS Trend")
          st.line_chart(adgroup_time_data, x='Date', y='ROAS')

          st.write("##### Diagnostic Metrics: Clicks, CTR & CPC")
          col1, col2, col3 = st.columns(3)
          with col1:
            st.line_chart(adgroup_time_data,
                          x='Date',
                          y='Clicks',
                          color="#FF4B4B")
          with col2:
            st.line_chart(adgroup_time_data,
                          x='Date',
                          y='CTR',
                          color="#4BFF4B")
          with col3:
            st.line_chart(adgroup_time_data,
                          x='Date',
                          y='CPC',
                          color="#4B4BFF")

    else:
      st.warning(
          "Ad group time-series analysis could not be performed. This may be due to a lack of ad group data, a clear date column in the data, or an error during code generation."
      )

  elif st.session_state.selected_deepdive_tab == 2:
    if st.session_state.creative_time_series_df is not None and not st.session_state.creative_time_series_df.empty:
      st.subheader("Creative Deep Dive")

      creative_list = st.session_state.creative_df.head(
          10)['Creative'].unique()
      selected_creative = st.selectbox("Select a Creative to Analyze:",
                                       creative_list,
                                       key="creative_select")

      if selected_creative:
        creative_time_data = st.session_state.creative_time_series_df[
            st.session_state.creative_time_series_df['Creative'] ==
            selected_creative].sort_values(by='Date')

        if creative_time_data.empty:
          st.warning(
              f"No time-series data available for creative: {selected_creative}"
          )
        else:
          # --- AI Insights ---
          with st.spinner("🤖 AI is analyzing creative trends..."):
            if selected_creative in st.session_state.creative_insights:
              insights = st.session_state.creative_insights[selected_creative]
              st.info("AI-Powered Creative Insights: (cached)")
              st.write(insights)
            else:
              insight_prompt = f"""
                            I am analyzing the performance of the creative: "{selected_creative}".
                            Here is the time-series data for it:
                            {creative_time_data.to_dict('records')}

                            Please provide a concise analysis based on the following framework:
                            1.  **ROAS Trend:** First, describe the overall trend of ROAS. Is it increasing, decreasing, or volatile?
                            2.  **Diagnostic Analysis:** Explain the 'why' behind the ROAS trend by looking at CTR and CPC.
                                - If ROAS is down, is it because CTR dropped (suggesting creative/audience fatigue) or because CPC increased (suggesting higher competition or CPM)?
                                - If ROAS is up, what is driving it? Higher CTR or lower CPC?
                            3.  **Creative Specific Recommendations:** Based on your analysis, suggest 1-2 clear, actionable next steps specific to creative optimization. For example: 'Rotate in new formats (e.g., video, carousel) to combat ad fatigue.', 'Repurpose high-performing creatives across other campaigns or placements.', 'Test new ad creatives with refreshed visuals or messaging.'

                            Present the output in clear, easy-to-read markdown.
                            """
              insights = get_llm_response(insight_prompt, api_key)
              if insights:
                # Remove the markdown code block fences if they exist
                insights = insights.strip()
                if insights.startswith("```markdown"):
                  insights = insights[11:]
                elif insights.startswith("```"):
                  insights = insights[3:]
                if insights.endswith("```"):
                  insights = insights[:-3]
                insights = insights.strip()
              st.session_state.creative_insights[selected_creative] = insights
              st.info("AI-Powered Creative Insights:")
              st.write(insights)

          st.subheader(
              f"Performance Trends for Creative '{selected_creative}'")

          col1, col2 = st.columns(2)
          with col1:
            st.metric("Total Spend",
                      f"{creative_time_data['Spends'].sum():,.2f}")
            st.metric("Total Revenue",
                      f"{creative_time_data['Revenue'].sum():,.2f}")
          with col2:
            avg_roas = creative_time_data['Revenue'].sum() / creative_time_data['Spends'].sum() if \
                creative_time_data['Spends'].sum() > 0 else 0
            st.metric("Overall ROAS", f"{avg_roas:.2f}x")

          st.write("##### ROAS Trend")
          st.line_chart(creative_time_data, x='Date', y='ROAS')

          st.write("##### Diagnostic Metrics: Clicks, CTR & CPC")
          col1, col2, col3 = st.columns(3)
          with col1:
            st.line_chart(creative_time_data,
                          x='Date',
                          y='Clicks',
                          color="#FF4B4B")
          with col2:
            st.line_chart(creative_time_data,
                          x='Date',
                          y='CTR',
                          color="#4BFF4B")
          with col3:
            st.line_chart(creative_time_data,
                          x='Date',
                          y='CPC',
                          color="#4B4BFF")

    else:
      st.warning(
          "Creative time-series analysis could not be performed. This may be due to a lack of creative data, a clear date column in the data, or an error during code generation."
      )

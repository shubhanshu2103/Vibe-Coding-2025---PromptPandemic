import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser 
import hashlib
import time 

# --- CONFIGURATION ---
METADATA_FILE = "form_metadata.json"
ADMIN_PASS = "hackathon2025" 

# Set page config FIRST - must be the very first Streamlit command
st.set_page_config(page_title="Dynamic AI Form Builder", layout="wide")

# --- UI STYLING & CUSTOM CSS (Keeping the original styling) ---

CUSTOM_CSS = """
<style>
/* 1. Global Streamlit Overrides to match dark theme (New Tailwind Colors) */
.stApp {
    background-color: #101922; /* New Deep Navy Blue/Black */
    color: #F0F8FF; /* Light Text */
}
/* Ensure main content background inherits the dark color */
[data-testid="stDecoration"] {
    background-color: #101922; 
}
/* Adjust markdown and main text color */
.stMarkdown, h1, h2, h3, p {
    color: #F0F8FF !important;
}
/* Adjust text color for specific Streamlit elements that rely on HMTL/CSS (e.g., labels) */
.stTextInput label, .stNumberInput label, .stDateInput label, .stCheckbox label, .stSelectbox label {
    color: #F0F8FF !important;
}

/* 2. Custom Header/Navigation Mock-up Styling */
.header-container {
    padding: 10px 0;
    margin-bottom: 20px;
    border-bottom: 1px solid rgba(17, 115, 212, 0.3); /* Based on primary/30 */
}
.logo-text {
    font-size: 20px; /* Text-xl */
    font-weight: 700;
    color: #F0F8FF;
}
.logo-icon {
    color: #1173d4; /* Primary color */
}
.nav-link {
    color: #94A3B8; /* slate-400 */
    padding: 8px 15px;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
}
.nav-link:hover {
    color: #1173d4; /* Primary color hover */
}

/* 3. Form Creator Prompt Box Styling */
[data-testid="stTextarea"] textarea {
    background-color: transparent !important; 
    border: none !important;
    color: #F0F8FF !important;
    min-height: 144px !important; 
    padding: 1rem !important; 
    box-shadow: none !important;
}

/* 4. Streamlit Button Styling */
[data-testid="stFormSubmitButton"] > button, .stButton > button {
    /* Styles from the HTML template's button class */
    background-color: #1173d4 !important;
    color: white !important;
    height: 3rem !important; /* h-12 */
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    border-radius: 0.5rem !important;
    font-weight: 700 !important;
    transition: transform 0.1s;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); /* shadow-lg */
}
[data-testid="stFormSubmitButton"] > button:hover, .stButton > button:hover {
    background-color: rgba(17, 115, 212, 0.9) !important; /* hover:bg-primary/90 */
}
[data-testid="stFormSubmitButton"] > button:active, .stButton > button:active {
    transform: scale(0.95);
}

/* 5. Dashboard Card and Data Styling (New Styles from the provided HTML) */
.dashboard-section-card {
    background-color: rgba(16, 25, 34, 0.7); /* Custom semi-transparent background */
    padding: 1.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.06);
    margin-bottom: 2rem; 
}
.chart-inner-card {
    background-color: #101922 !important; /* background-dark */
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid #374151; /* gray-700 */
    min-height: 300px; /* Ensure space for charts */
}
.ai-insights-card {
    background-color: rgba(17, 115, 212, 0.2) !important;
    padding: 1.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
[data-testid="stDataFrame"] {
    border-radius: 0.5rem !important; 
    border: 1px solid #374151 !important; /* gray-700 */
    overflow-x: auto;
}
[data-testid="stDataFrame"] .row-header {
    background-color: #1F2937 !important; /* gray-800 for the header row */
    color: #9CA3AF !important; /* gray-400 text */
    font-weight: 500;
    text-transform: uppercase;
    font-size: 0.75rem !important; /* text-xs */
}
[data-testid="stDataFrame"] .data-row {
    background-color: #101922 !important; /* background-dark */
    border-bottom: 1px solid #374151 !important; /* gray-700 divider */
}
[data-testid="stDataFrame"] .data-row > div {
    color: #F0F8FF !important; /* white text */
    font-size: 0.875rem !important; /* text-sm */
}
.stContainer {
    background-color: transparent !important;
    border-radius: 0;
}
</style>
"""

# Apply the custom CSS immediately
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- 1. DYNAMIC DATA PERSISTENCE FUNCTIONS (CSV & METADATA) ---

def get_data_file_path(form_id: int) -> str:
    """Returns the unique CSV file path for a given form ID."""
    return f"form_data_{form_id}.csv"

def load_data(form_id: int) -> pd.DataFrame:
    """Loads submission data from the unique CSV file for a specific form_id, ensuring timestamp is parsed."""
    data_file = get_data_file_path(form_id)
    try:
        if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
            # --- FIX: Explicitly parse 'timestamp' column as datetime on load ---
            df = pd.read_csv(data_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df
    except pd.errors.EmptyDataError:
        pass
    except Exception:
        pass
    return pd.DataFrame()

def append_data(form_id: int, data: dict):
    """Appends a new submission to the unique CSV file for a specific form_id."""
    data_file = get_data_file_path(form_id)
    try:
        df_new = pd.DataFrame([data])
        write_header = not (os.path.exists(data_file) and os.path.getsize(data_file) > 0)
        df_new.to_csv(data_file, mode='a', header=write_header, index=False)
    except Exception as e:
        st.error(f"Failed to save data to {data_file}: {e}")

# --- METADATA FUNCTIONS (To track all created forms) ---

def save_form_metadata(form_id: int, definition: dict, prompt: str):
    """Saves the form definition, prompt, and ID to the metadata file."""
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
    except:
        metadata = {}

    metadata[str(form_id)] = {
        "id": form_id,
        "definition": definition,
        "prompt": prompt,
        "created_at": pd.Timestamp.now().isoformat()
    }

    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
        
def load_all_form_metadata() -> dict:
    """Loads all form metadata (ID, prompt, definition) from the JSON file."""
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
                return {int(k): v for k, v in metadata.items()}
    except:
        pass
    return {}


# --- 2. LLM CORE FUNCTIONS (AI Brains) ---

JSON_SCHEMA = {
    "clarification": "string | null (a message if the request is contradictory, otherwise null)",
    "fields": [
        {
            "name": "string (snake_case name)",
            "type": "string (e.g., text, email, number, date, checkbox)",
            "label": "string (User-friendly label)",
            "validation": "string (e.g., 'required', 'email_format', 'min_length:10' or 'optional')"
        }
    ]
}

def generate_form_json(user_request: str) -> dict:
    """Uses Llama 3 to convert natural language into a structured form JSON."""
    try:
        llm = Ollama(model="llama3")
    except Exception as e:
        st.error(f"Failed to connect to Ollama (llama3). Is the server running? Error: {e}")
        return None
    
    schema_string = json.dumps(JSON_SCHEMA, indent=2)
    
    system_prompt = """
    You are a highly precise AI form builder. Your task is to analyze a user's request and output a form definition in the required JSON format.
    RULES:
    1. Always output a single, valid JSON object following this structure: {schema}.
    2. Field names must be in snake_case.
    3. Use appropriate types: 'text', 'email', 'number', 'date', 'checkbox'.
    4. CRITICAL: If the user request contains a contradiction (e.g., "anonymous form but must include full name"), you MUST set the 'clarification' field to a message and set 'fields' to an empty list [].
    Strictly adhere to the JSON format. Do not include any text before or after the JSON block.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User's Form Request: {request}")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        response_dict = chain.invoke({
            "request": user_request, 
            "schema": schema_string
        })
        return response_dict
    except Exception as e:
        st.error(f"LLM Processing Error during JSON generation: {e}")
        st.info("Check if your Ollama server is running and the 'llama3' model is pulled.")
        return None

def generate_ai_insights(df: pd.DataFrame) -> str:
    """Uses Llama 3 to analyze data text and generate key insights."""
    try:
        llm = Ollama(model="llama3")
    except:
        return "AI Insight generation failed: Could not connect to Ollama."
        
    data_summary = f"Total Submissions: {len(df)}\nColumns: {list(df.columns)}\n\n"
    data_summary += "Value counts (Top 5 columns):\n"
    for col in df.columns[:5]:
        # Only include object or low-cardinality columns for value counts
        if df[col].dtype == 'object' or df[col].nunique() < 10:
            data_summary += f"- {col}: {df[col].value_counts().to_dict()}\n"
    
    system_prompt = """
    You are an expert data analyst. Review the provided summary of form submission data and generate three concise, actionable, and high-level insights for an administrator.
    Present your findings as a numbered list in Markdown format.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Analyze this data summary: \n\n{summary}")
    ])
    chain = prompt | llm
    try:
        # Increase timeout slightly, as large LLM calls can take time
        response_text = chain.invoke({"summary": data_summary}, config={'run_name': 'ai_insights_call'})
        return response_text
    except Exception as e:
        return f"AI Insight generation failed during LLM call: {e}. Check Ollama logs."

# --- 3. UI RENDERING FUNCTIONS ---

def render_custom_header(current_page):
    """Renders the custom header."""
    col_logo, col_links, col_new_form, col_avatar = st.columns([1.5, 3, 1, 0.5])

    with col_logo:
        st.markdown("""
            <div class="flex items-center gap-2">
                <div class="size-6 logo-icon">
                    <svg fill="none" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><path d="M4 4H17.3334V17.3334H30.6666V30.6666H44V44H4V4Z" fill="currentColor"></path></svg>
                </div>
                <h2 class="text-xl font-bold logo-text" style="color:white;">FormForge</h2>
            </div>
            """, unsafe_allow_html=True)

    with col_links:
        st.markdown("""
            <div class="flex items-center gap-6 justify-start">
                <a class="nav-link" href="#">Home</a>
                <a class="nav-link" href="#">Templates</a>
                <a class="nav-link" href="#">Examples</a>
                <a class="nav-link" href="#">Pricing</a>
            </div>
            """, unsafe_allow_html=True)

    with col_new_form:
        if st.button("New Form", key="nav_new_form_stream", type="secondary", use_container_width=True):
            st.session_state['page'] = "Form Creator (Public)"
            st.rerun() 
             
    with col_avatar:
        st.markdown("""
            <div class="bg-center bg-no-repeat aspect-square bg-cover rounded-full size-10" style='background-image: url("https://lh3.googleusercontent.com/aida-public/AB6AXuC3f0PEaY0Oeiae269K4ceGmthuuxozLXVbkxPx06nRL9zEweHJLn7l_vztkByKvdMG8h0HT3Jk998Xs7gH1bY118Amo28ZX9dY1z8cBZo4QqNCHzoKwqNf4en5CE5kqOB2MG7JLDhWFZ823IDkuSdZ3sPeNWyln5u-POIqA4i12R0SmmE7znB_JN-S9Qp8FW4DTSp-BYiR1NCtLbX88ChsvwkoxvkRhYnwWb_WcqUcHDc176jkK3hzPtOgvaPRWx1h3QWP297VFA");'></div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="header-container"></div>', unsafe_allow_html=True)


def render_form(form_id: int, data: dict):
    """Dynamically renders a Streamlit form, handles submission, and stores data."""
    
    if data.get('clarification'):
        st.warning("🚨 Contradiction Detected! Please clarify your request.")
        st.info(data['clarification'])
        return

    st.markdown("""
        <div class="w-full pt-10 border-t border-[#1173d4]/20 dark:border-[#1173d4]/30">
        <h2 class="text-2xl font-bold text-white text-center">Generated Form</h2>
        <p class="text-slate-400 text-center mt-2">Submit your data below. All entries are saved uniquely for this form.</p>
        </div>
        """, unsafe_allow_html=True)

    
    # Store field keys and types for safe post-submission retrieval
    field_keys = {}
    
    with st.container():
        with st.form(key=f"dynamic_form_{form_id}", clear_on_submit=True):
            
            st.markdown(f"**Form ID (Unique Identifier):** `{form_id}`")

            for i, field in enumerate(data.get('fields', [])):
                field_name = field['name']
                label = field['label']
                field_type = field['type']
                
                # Create a hyper-unique key
                key = f"form_{form_id}_{field_name}_{i}_{field_type}" 
                field_keys[field_name] = {'key': key, 'type': field_type} # Store key AND type
                
                # Streamlit component rendering
                if field_type == 'text' or field_type == 'email':
                    st.text_input(label, key=key)
                elif field_type == 'number':
                    st.number_input(label, key=key, step=1, format="%d", value=0)
                elif field_type == 'date':
                    st.date_input(label, key=key)
                elif field_type == 'checkbox':
                    st.checkbox(label, key=key)
            
            submitted = st.form_submit_button("Submit Form", type="primary")
            
            if submitted:
                final_submission = {}
                
                for field_name, info in field_keys.items():
                    key = info['key']
                    field_type = info['type']
                    
                    value = st.session_state.get(key)
                    
                    # Safely handle date conversion and None values
                    if field_type == 'date' and value is not None:
                        value = value.isoformat()
                    elif value is None and (field_type == 'text' or field_type == 'email'):
                        value = ""
                    
                    final_submission[field_name] = value
                
                # Add metadata fields
                final_submission['form_id'] = form_id 
                final_submission['timestamp'] = pd.Timestamp.now().isoformat()
                
                # Use the verified data for submission
                append_data(form_id, final_submission)
                st.success(f"Form Submitted Successfully! Data Saved to file: form_data_{form_id}.csv")

def render_dashboard():
    """Renders the Admin Dashboard with form selection and analytics for the selected form."""
    
    st.markdown("""
        <div class="mb-8">
            <h1 class="text-3xl font-bold text-gray-900 dark:text-white">Data Insights Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)
    
    all_forms_metadata = load_all_form_metadata()
    if not all_forms_metadata:
        st.warning("No forms have been generated yet. Create one in the Form Creator view to see analytics.")
        return
    
    # Create options for the selectbox: Display (Prompt...) -> Internal ID
    form_options = {
        f"ID: {meta['id']} | Prompt: {meta['prompt'][:50]}... (Created: {meta['created_at'][:10]})" : meta['id']
        for meta in all_forms_metadata.values()
    }
    
    # 1. Form Selector
    st.markdown('<div class="dashboard-section-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Select Form to Analyze</h2>', unsafe_allow_html=True)
    
    selected_key = st.selectbox("Choose a generated form:", list(form_options.keys()), key="dashboard_form_select", label_visibility="collapsed")
    
    selected_form_id = form_options[selected_key]
    
    st.markdown(f"Analyzing data for **Form ID: {selected_form_id}**")
    st.markdown('</div>', unsafe_allow_html=True) # Close dashboard-section-card
    
    # 2. Load Data for the SELECTED ID
    df = load_data(selected_form_id)
    
    if df.empty:
        st.info("The selected form has no submissions yet.")
        return

    # --- Data Overview Card ---
    st.markdown('<div class="dashboard-section-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Latest Submissions</h2>', unsafe_allow_html=True)
    st.dataframe(df.tail(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True) 

    # --- Charts Section Card ---
    st.markdown('<div class="dashboard-section-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Charts</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    chart_cols = [col for col in df.columns if col not in ['timestamp', 'date', 'form_id']] 
    
    # Chart 1: Distribution
    with col1:
        st.markdown('<div class="chart-inner-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="text-base font-medium mb-4 text-gray-900 dark:text-white">Field Distribution</h3>', unsafe_allow_html=True)
        if chart_cols:
            chart_choice = st.selectbox("Select a field for distribution:", chart_cols, key="chart_select_1", label_visibility="collapsed")
            
            if df[chart_choice].dtype == 'object' or df[chart_choice].nunique() < 10:
                count_data = df[chart_choice].value_counts().reset_index()
                count_data.columns = [chart_choice, 'Count']
                
                fig = px.pie(
                    count_data, values='Count', names=chart_choice, title=f'Distribution of {chart_choice}',
                    hole=.3, color_discrete_sequence=px.colors.sequential.Agsunset
                )
                fig.update_layout(
                    paper_bgcolor='#101922', plot_bgcolor='#101922', font_color='white',
                    margin=dict(l=10, r=10, t=40, b=10), showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif pd.api.types.is_numeric_dtype(df[chart_choice]):
                fig = px.histogram(df, x=chart_choice, title=f'Distribution of {chart_choice}', 
                                   color_discrete_sequence=['#1173d4'])
                fig.update_layout(
                    paper_bgcolor='#101922', plot_bgcolor='#101922', font_color='white',
                    bargap=0.2, margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No analyzable fields found in this form.")
        st.markdown('</div>', unsafe_allow_html=True) 

    # Chart 2: Time Series
    with col2:
        st.markdown('<div class="chart-inner-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="text-base font-medium mb-4 text-gray-900 dark:text-white">Form Submissions Over Time</h3>', unsafe_allow_html=True)
        
        # --- FIX: df['timestamp'] is now guaranteed to be datetime objects (or NaT) from load_data ---
        if 'timestamp' in df.columns and not df.empty and not df['timestamp'].isnull().all():
            
            # Use .dt.date to strip time component and aggregate submissions per day
            df['date'] = df['timestamp'].dt.date
            daily_submissions = df.groupby('date').size().reset_index(name='Submissions')
            
            fig = px.area(
                daily_submissions, x='date', y='Submissions', 
                title='Daily Submission Trend',
                line_shape='spline',
                color_discrete_sequence=['#1173d4']
            )
            fig.update_layout(
                paper_bgcolor='#101922', plot_bgcolor='#101922', font_color='white',
                margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Date", yaxis_title="Count"
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor='#374151')
            st.plotly_chart(fig, use_container_width=True)
        else:
             st.info("Timestamp data required for trend analysis (or all submissions lack valid timestamps).")

        st.markdown('</div>', unsafe_allow_html=True) 
    st.markdown('</div>', unsafe_allow_html=True) 


    # --- AI Insights Card ---
    st.markdown('<div class="ai-insights-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">AI Insights</h2>', unsafe_allow_html=True)
    with st.spinner('Analyzing data and generating insights with Llama 3...'):
        insights = generate_ai_insights(df)
        st.markdown(insights)
    st.markdown('</div>', unsafe_allow_html=True) 


def check_password_main_body(key_prefix):
    """Simple password check for the admin dashboard access, rendered in the main body."""
    if 'password_correct' not in st.session_state:
        st.session_state['password_correct'] = False
        
    st.markdown("## Admin Login")
    password = st.text_input("Password", type="password", key=f"{key_prefix}_password")
    
    login_button = st.button("Access Dashboard", key=f"{key_prefix}_login_button", type="primary")

    if login_button or (password == ADMIN_PASS and not st.session_state['password_correct']):
        if password == ADMIN_PASS:
            st.session_state['password_correct'] = True
            st.success("Access Granted! Redirecting to dashboard...")
            st.session_state['page'] = "Admin Dashboard (Private)"
            st.rerun()
        elif password:
            st.error("Incorrect Password")
            st.session_state['password_correct'] = False
        
    return st.session_state['password_correct']

# --- 4. MAIN APPLICATION LOGIC (Routing) ---

# Remove default Streamlit header/footer
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Initialize page state
if 'page' not in st.session_state:
    st.session_state['page'] = "Form Creator (Public)"
if 'current_form_id' not in st.session_state:
    st.session_state['current_form_id'] = None

# Render the custom header mock-up
render_custom_header(st.session_state.get('page'))


# Navigation radio button
st.markdown("## Navigation", unsafe_allow_html=True)
page = st.radio(
    "Select View:", 
    ["Form Creator (Public)", "Admin Dashboard (Private)"], 
    key="main_navigation_radio",
    index=0 if st.session_state.get('page') == "Form Creator (Public)" else 1,
    horizontal=True
)

# Update session state based on radio selection
if page != st.session_state.get('page'):
    st.session_state['page'] = page
    st.rerun()

    
# --- Content Rendering ---

if st.session_state['page'] == "Form Creator (Public)":
    
    st.markdown("""
    <div class="flex flex-col items-center gap-10">
        <div class="text-center">
            <h1 class="text-4xl md:text-5xl font-bold text-white tracking-tight">Craft Your Perfect Form</h1>
            <p class="mt-4 text-lg text-slate-400">Describe the form you need, and we'll generate it for you.</p>
        </div>
        <div class="w-full max-w-4xl mx-auto p-2" style="background: linear-gradient(to right, #1173d4, #1173d4, rgba(17, 115, 212, 0.2)); border-radius: 0.75rem; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);">
            <div style="background-color: #101922; border-radius: 0.5rem; padding: 0.5rem;">
    """, unsafe_allow_html=True)
    
    user_prompt = st.text_area(
        "Prompt Area:",
        placeholder="e.g., 'A modern registration form with fields for name, email, and password.'",
        height=100,
        label_visibility="collapsed",
        key="user_prompt_input"
    )
    
    st.markdown("""
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_button_left, col_button_center, col_button_right = st.columns([1, 2.5, 1])

    with col_button_center:
        if st.button("Generate Form", type="secondary", key="generate_form_button_main", use_container_width=True):
            if user_prompt:
                st.session_state.pop('form_definition', None)
                st.session_state.pop('current_form_id', None)
                
                with st.spinner('Generating form definition using Llama 3...'):
                    form_definition = generate_form_json(user_prompt)
                
                if form_definition and not form_definition.get('clarification'):
                    # Generate unique ID based on the JSON structure
                    json_string = json.dumps(form_definition, sort_keys=True).encode('utf-8')
                    form_id = int(hashlib.sha256(json_string).hexdigest(), 16) % (10**10) 
                    
                    st.session_state['current_form_id'] = form_id
                    st.session_state['form_definition'] = form_definition
                    
                    # Save the form metadata
                    save_form_metadata(form_id, form_definition, user_prompt)
                    
                elif form_definition and form_definition.get('clarification'):
                    st.session_state['form_definition'] = form_definition
            else:
                st.error("Please enter a form description.")
    
    # Render the generated form
    if ('form_definition' in st.session_state and st.session_state['form_definition'] and 
        'current_form_id' in st.session_state and st.session_state['current_form_id']):
        render_form(st.session_state['current_form_id'], st.session_state['form_definition'])

elif st.session_state['page'] == "Admin Dashboard (Private)":
    
    if st.session_state.get('password_correct'):
        render_dashboard()
    else:
        st.markdown(
            """
            <div class="mt-8 p-8 max-w-md mx-auto dashboard-section-card">
                <h2 class="text-2xl font-bold mb-4 text-white">Admin Dashboard Access Required</h2>
                <p class="text-slate-400 mb-6">Enter the administrative password to view the data insights.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        with st.container():
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                check_password_main_body("dashboard_login")
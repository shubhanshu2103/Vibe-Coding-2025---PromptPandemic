import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

# --- LLM and LangChain Imports (Ensure Ollama is running: 'ollama serve') ---
# NOTE: These components communicate with the local Ollama server.
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser 

# --- CONFIGURATION ---
DATA_FILE = "form_submissions.csv"
ADMIN_PASS = "hackathon2025" 

# --- UI STYLING & CUSTOM CSS (Implementing the Dark Theme and Cards) ---

# New color scheme based on user's HTML/Tailwind template:
# Background: #101922
# Primary/Accent: #1173d4
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

/* General Card for Data Overview and Charts Section (bg-background-dark/50 p-6 rounded-lg shadow-lg) */
.dashboard-section-card {
    /* bg-background-dark/50 */
    background-color: rgba(16, 25, 34, 0.7); /* Custom semi-transparent background */
    padding: 1.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.06);
    margin-bottom: 2rem; 
}

/* Inner Chart Card (bg-background-dark p-6 rounded-lg border border-gray-700) */
.chart-inner-card {
    background-color: #101922 !important; /* background-dark */
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid #374151; /* gray-700 */
    min-height: 300px; /* Ensure space for charts */
}

/* AI Insights Card (bg-primary/20 p-6 rounded-lg shadow-lg) */
.ai-insights-card {
    background-color: rgba(17, 115, 212, 0.2) !important;
    padding: 1.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* Dataframe Styling (To match the HTML table look) */
[data-testid="stDataFrame"] {
    /* Rounded corners, border and background for the outer table container */
    border-radius: 0.5rem !important; 
    border: 1px solid #374151 !important; /* gray-700 */
    overflow-x: auto;
}
/* Style the table head */
[data-testid="stDataFrame"] .row-header {
    background-color: #1F2937 !important; /* gray-800 for the header row */
    color: #9CA3AF !important; /* gray-400 text */
    font-weight: 500;
    text-transform: uppercase;
    font-size: 0.75rem !important; /* text-xs */
}
/* Style the table body rows */
[data-testid="stDataFrame"] .data-row {
    background-color: #101922 !important; /* background-dark */
    border-bottom: 1px solid #374151 !important; /* gray-700 divider */
}
/* Style the cell text to be white/light */
[data-testid="stDataFrame"] .data-row > div {
    color: #F0F8FF !important; /* white text */
    font-size: 0.875rem !important; /* text-sm */
}

/* Reset Streamlit containers so custom styles take precedence */
.stContainer {
    background-color: transparent !important;
    border-radius: 0;
}
</style>
"""

# Apply the custom CSS immediately
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- 1. DATA PERSISTENCE FUNCTIONS (CSV) ---
# (Functions remain the same for stability)

def load_data():
    """Loads all submission data from the CSV file into a pandas DataFrame."""
    try:
        if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
            return pd.read_csv(DATA_FILE)
    except pd.errors.EmptyDataError:
        pass
    except Exception as e:
        st.error(f"Error loading data: {e}")
    return pd.DataFrame()

def append_data(data: dict):
    """Appends a new submission to the CSV file."""
    try:
        df_new = pd.DataFrame([data])
        write_header = not os.path.exists(DATA_FILE)
        df_new.to_csv(DATA_FILE, mode='a', header=write_header, index=False)
        st.session_state['data_updated'] = True
    except Exception as e:
        st.error(f"Failed to save data: {e}")

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
    llm = Ollama(model="llama3")
    
    # Dump the schema to a string for explicit insertion into the template
    schema_string = json.dumps(JSON_SCHEMA, indent=2)
    
    # Refactored system_prompt to use {schema} placeholder instead of f-string interpolation
    system_prompt = """
    You are a highly precise AI form builder. Your task is to analyze a user's request and output a form definition in the required JSON format.
    RULES:
    1. Always output a single, valid JSON object following this structure: {schema}.
    2. Field names must be in snake_case.
    3. Use appropriate types: 'text', 'email', 'number', 'date', 'checkbox'.
    4. CRITICAL: If the user request contains a contradiction (e.g., "anonymous form but must include full name"), you MUST set the 'clarification' field to a message and set 'fields' to an empty list [].
    Strictly adhere to the JSON format. Do not include any text before or after the JSON block.
    """
    
    # Now the ChatPromptTemplate only expects 'schema' and 'request' variables
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User's Form Request: {request}")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        # Pass the request and the schema string explicitly during invocation
        response_dict = chain.invoke({
            "request": user_request, 
            "schema": schema_string
        })
        return response_dict
    except Exception as e:
        # Note: Added the full traceback details for better debugging in case the user shares the log again
        st.error(f"LLM Processing Error during JSON generation: {e}")
        st.info("Check if your Ollama server is running and the 'llama3' model is pulled.")
        return None

def generate_ai_insights(df: pd.DataFrame) -> str:
    """Uses Llama 3 to analyze data text and generate key insights."""
    llm = Ollama(model="llama3")
    data_summary = f"Columns: {list(df.columns)}\n\n"
    data_summary += "Value counts (Top 5 columns):\n"
    for col in df.columns[:5]:
        if df[col].dtype == 'object' or df[col].nunique() < 10:
            data_summary += f"- {col}: {df[col].value_counts().to_dict()}\n"
    data_summary += f"\nData descriptive stats:\n{df.describe(include='all').to_markdown()}"
    
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
        response_text = chain.invoke({"summary": data_summary})
        return response_text
    except Exception as e:
        return f"AI Insight generation failed: {e}. (Ollama connection issue or prompt length too long?)"

# --- 3. UI RENDERING FUNCTIONS ---

def render_custom_header(current_page):
    """Renders the custom header using the provided HTML/Tailwind structure and a Streamlit button for function."""
    
    # We use st.columns to simulate the flex layout of the header
    col_logo, col_links, col_new_form, col_avatar = st.columns([1.5, 3, 1, 0.5])

    # 1. Logo Section (Logo + Title)
    with col_logo:
        st.markdown("""
            <div class="flex items-center gap-2">
                <div class="size-6 logo-icon">
                    <svg fill="none" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><path d="M4 4H17.3334V17.3334H30.6666V30.6666H44V44H4V4Z" fill="currentColor"></path></svg>
                </div>
                <h2 class="text-xl font-bold logo-text" style="color:white;">FormForge</h2>
            </div>
            """, unsafe_allow_html=True)

    # 2. Navigation Links
    with col_links:
        # Mocking the links using raw HTML for perfect styling
        st.markdown("""
            <div class="flex items-center gap-6 justify-start">
                <a class="nav-link" href="#">Home</a>
                <a class="nav-link" href="#">Templates</a>
                <a class="nav-link" href="#">Examples</a>
                <a class="nav-link" href="#">Pricing</a>
            </div>
            """, unsafe_allow_html=True)

    # 3. New Form Button (Streamlit button styled by CSS)
    with col_new_form:
        # The button is functional and styled by the extensive CSS overrides
        if st.button("New Form", key="nav_new_form_stream", type="secondary", use_container_width=True):
            # Ensure we navigate to the public page
            st.session_state['page'] = "Form Creator (Public)"
            st.rerun() 
             
    # 4. Avatar
    with col_avatar:
        st.markdown("""
            <div class="bg-center bg-no-repeat aspect-square bg-cover rounded-full size-10" style='background-image: url("https://lh3.googleusercontent.com/aida-public/AB6AXuC3f0PEaY0Oeiae269K4ceGmthuuxozLXVbkxPx06nRL9zEweHJLn7l_vztkByKvdMG8h0HT3Jk998Xs7gH1bY118Amo28ZX9dY1z8cBZo4QqNCHzoKwqNf4en5CE5kqOB2MG7JLDhWFZ823IDkuSdZ3sPeNWyln5u-POIqA4i12R0SmmE7znB_JN-S9Qp8FW4DTSp-BYiR1NCtLbX88ChsvwkoxvkRhYnwWb_WcqUcHDc176jkK3hzPtOgvaPRWx1h3QWP297VFA");'></div>
            """, unsafe_allow_html=True)

    # Add the header bottom border
    st.markdown('<div class="header-container"></div>', unsafe_allow_html=True)


def render_form(data: dict):
    """Dynamically renders a Streamlit form, handles submission, and stores data."""
    
    if data.get('clarification'):
        st.warning("🚨 Contradiction Detected! Please clarify your request.")
        st.info(data['clarification'])
        return

    # Generated Form Section Header (matching the template's divider)
    st.markdown("""
        <div class="w-full pt-10 border-t border-[#1173d4]/20 dark:border-[#1173d4]/30">
        <h2 class="text-2xl font-bold text-white text-center">Generated Form</h2>
        <p class="text-slate-400 text-center mt-2">Your generated form will appear here after you submit your prompt.</p>
        </div>
        """, unsafe_allow_html=True)

    # Use a container for the generated form to provide card styling if needed
    with st.container():
        
        with st.form(key="dynamic_form", clear_on_submit=True):
            submission_data = {}
            st.markdown(f"**Form Hash ID:** `{hash(json.dumps(data, sort_keys=True))}`")

            for field in data.get('fields', []):
                field_name = field['name']
                label = field['label']
                field_type = field['type']
                
                # Streamlit components will be styled by the global CSS
                if field_type == 'text' or field_type == 'email':
                    submission_data[field_name] = st.text_input(label, key=f"form_{field_name}")
                elif field_type == 'number':
                    submission_data[field_name] = st.number_input(label, key=f"form_{field_name}", step=1, format="%d")
                elif field_type == 'date':
                    submission_data[field_name] = st.date_input(label, key=f"form_{field_name}").isoformat()
                elif field_type == 'checkbox':
                    submission_data[field_name] = st.checkbox(label, key=f"form_{field_name}")
            
            submitted = st.form_submit_button("Submit Form", type="primary")
            
            if submitted:
                submission_data['timestamp'] = pd.Timestamp.now().isoformat()
                append_data(submission_data)
                st.success("Form Submitted Successfully! Data Saved to CSV.")

def render_dashboard(df: pd.DataFrame):
    """Renders the Admin Dashboard with charts and AI insights using the card style."""
    
    # Dashboard Header
    st.markdown("""
        <div class="mb-8">
            <h1 class="text-3xl font-bold text-gray-900 dark:text-white">Data Insights Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No submission data yet. Fill out the form to see the magic!")
        return

    # --- 1. Data Overview Card ---
    st.markdown('<div class="dashboard-section-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Data Overview</h2>', unsafe_allow_html=True)
    st.dataframe(df.tail(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True) # Close dashboard-section-card

    # --- 2. Charts Section Card ---
    st.markdown('<div class="dashboard-section-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Charts</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    chart_cols = [col for col in df.columns if col not in ['timestamp', 'date']] # Exclude new 'date' column too
    
    # Chart 1: Distribution
    with col1:
        st.markdown('<div class="chart-inner-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="text-base font-medium mb-4 text-gray-900 dark:text-white">Field Distribution</h3>', unsafe_allow_html=True)
        if chart_cols:
            # Use a hidden selectbox since the label is in the HTML H3 tag
            chart_choice = st.selectbox("Select a field for distribution:", chart_cols, key="chart_select_1", label_visibility="collapsed")
            
            if df[chart_choice].dtype == 'object' or df[chart_choice].nunique() < 10:
                count_data = df[chart_choice].value_counts().reset_index()
                count_data.columns = [chart_choice, 'Count']
                
                fig = px.pie(
                    count_data, values='Count', names=chart_choice, title=f'Distribution of {chart_choice}',
                    hole=.3, color_discrete_sequence=px.colors.sequential.Agsunset
                )
                fig.update_layout(
                    paper_bgcolor='#101922',  # background-dark
                    plot_bgcolor='#101922',
                    font_color='white',
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif pd.api.types.is_numeric_dtype(df[chart_choice]):
                fig = px.histogram(df, x=chart_choice, title=f'Distribution of {chart_choice}', 
                                   color_discrete_sequence=['#1173d4'])
                fig.update_layout(
                    paper_bgcolor='#101922', 
                    plot_bgcolor='#101922',
                    font_color='white',
                    bargap=0.2,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No analyzable fields found.")
        st.markdown('</div>', unsafe_allow_html=True) # Close chart-inner-card

    # Chart 2: Time Series
    with col2:
        st.markdown('<div class="chart-inner-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="text-base font-medium mb-4 text-gray-900 dark:text-white">Form Submissions Over Time</h3>', unsafe_allow_html=True)
        
        # Calculate time series data from actual submissions
        if 'timestamp' in df.columns and not df.empty:
            df['date'] = pd.to_datetime(df['timestamp']).dt.normalize()
            daily_submissions = df.groupby('date').size().reset_index(name='Submissions')
            
            fig = px.area(
                daily_submissions, x='date', y='Submissions', 
                title='Daily Submission Trend',
                line_shape='spline',
                color_discrete_sequence=['#1173d4']
            )
            fig.update_layout(
                paper_bgcolor='#101922', 
                plot_bgcolor='#101922',
                font_color='white',
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Date",
                yaxis_title="Count"
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor='#374151')
            st.plotly_chart(fig, use_container_width=True)
        else:
             st.info("Timestamp data required for trend analysis.")

        st.markdown('</div>', unsafe_allow_html=True) # Close chart-inner-card
    st.markdown('</div>', unsafe_allow_html=True) # Close dashboard-section-card


    # --- 3. AI Insights Card ---
    st.markdown('<div class="ai-insights-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">AI Insights</h2>', unsafe_allow_html=True)
    with st.spinner('Analyzing data and generating insights with Llama 3...'):
        insights = generate_ai_insights(df)
        st.markdown(insights)
    st.markdown('</div>', unsafe_allow_html=True) # Close ai-insights-card


def check_password_main_body(key_prefix):
    """Simple password check for the admin dashboard access, rendered in the main body."""
    # Use session state to manage password status
    if 'password_correct' not in st.session_state:
        st.session_state['password_correct'] = False
        
    st.markdown("## Admin Login")
    password = st.text_input("Password", type="password", key=f"{key_prefix}_password")
    
    login_button = st.button("Access Dashboard", key=f"{key_prefix}_login_button", type="primary")

    # If the button is pressed OR the correct password is in the text input on load (for initial page load check)
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

# Set page config BEFORE any Streamlit calls that use it
st.set_page_config(page_title="Dynamic AI Form Builder", layout="wide")

# Remove default Streamlit header/footer
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Initialize page state if not present
if 'page' not in st.session_state:
    st.session_state['page'] = "Form Creator (Public)"

# Render the custom header mock-up
render_custom_header(st.session_state.get('page'))


# --- Handle Main Navigation and View Switching ---

# Navigation radio button placed in the main body for visibility without the sidebar
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
    # Form Creator Page Content
    
    # Use HTML structure for the title and prompt box wrapper
    st.markdown("""
    <div class="flex flex-col items-center gap-10">
        <div class="text-center">
            <h1 class="text-4xl md:text-5xl font-bold text-white tracking-tight">Craft Your Perfect Form</h1>
            <p class="mt-4 text-lg text-slate-400">Describe the form you need, and we'll generate it for you.</p>
        </div>
        <!-- Outer Gradient Wrapper (p-2 bg-gradient-to-r...) -->
        <div class="w-full max-w-4xl mx-auto p-2" style="background: linear-gradient(to right, #1173d4, #1173d4, rgba(17, 115, 212, 0.2)); border-radius: 0.75rem; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);">
            <!-- Inner Background (bg-background-dark p-2) -->
            <div style="background-color: #101922; border-radius: 0.5rem; padding: 0.5rem;">
    """, unsafe_allow_html=True)
    
    # The actual Streamlit text area component
    user_prompt = st.text_area(
        "Prompt Area:",
        placeholder="e.g., 'A modern registration form with fields for name, email, and password.'",
        height=100,
        label_visibility="collapsed",
        key="user_prompt_input"
    )
    
    # Close the HTML wrappers
    st.markdown("""
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the button (using columns)
    col_button_left, col_button_center, col_button_right = st.columns([1, 2.5, 1])

    with col_button_center:
        # Check if prompt is entered before generating
        if st.button("Generate Form", type="secondary", key="generate_form_button_main", use_container_width=True):
            if user_prompt:
                # Clear previous generation
                st.session_state.pop('form_definition', None)
                
                with st.spinner('Generating form definition using Llama 3...'):
                    form_definition = generate_form_json(user_prompt)
                
                if form_definition:
                    st.session_state['form_definition'] = form_definition
            else:
                st.error("Please enter a form description.")
    
    # Render the generated form
    if 'form_definition' in st.session_state and st.session_state['form_definition']:
        render_form(st.session_state['form_definition'])

elif st.session_state['page'] == "Admin Dashboard (Private)":
    # Admin Dashboard Logic
    
    # If password is correct, render the dashboard
    if st.session_state.get('password_correct'):
        df_submissions = load_data()
        render_dashboard(df_submissions)
    else:
        # If password is NOT correct, show login form in the main body
        st.markdown(
            """
            <div class="mt-8 p-8 max-w-md mx-auto dashboard-section-card">
                <h2 class="text-2xl font-bold mb-4 text-white">Admin Dashboard Access Required</h2>
                <p class="text-slate-400 mb-6">Enter the administrative password to view the data insights.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        # Place the login check within a centered container
        with st.container():
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                check_password_main_body("dashboard_login")
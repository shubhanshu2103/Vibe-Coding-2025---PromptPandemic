title: PromptPandemic Dynamic Form Builder emoji: 🧙‍♀️ colorFrom: pink colorTo: purple sdk: streamlit sdk_version: 1.36.0 app_file: app.py pinned: false
PromptPandemic Dynamic Form Builder
# 🧙‍♂️ PromptPandemic Dynamic Form Builder

**Build. Bond. Breakthrough.** (Vibe Coding Hackathon Entry)

This project is an AI-first application built for the PromptPandemic Vibe Coding Hackathon. It demonstrates the ability to translate unstructured natural language requests into a strictly structured, functional web form instantly.

## ✨ Innovation: Natural Language to Structured UI

The core innovation is connecting a local Large Language Model (LLama 3) to a dynamic UI generator (Streamlit) using Pydantic for schema enforcement. This allows the system to:

- **Generate Forms**: Convert descriptions like "A club sign-up form with name, email, and t-shirt size options" into a working web form.

- **Handle Contradictions**: Detect logical flaws in the user's request (e.g., "anonymous, but collect a phone number") and politely ask for clarification instead of failing.

- **Dynamic Validation**: Infer validation rules (required, min_length, email_format) from the prompt and enforce them client-side.

## 🛠 Setup Guide (For Forkers/Collaborators)

This application is designed to run completely locally using your machine's GPU for inference (Apple Silicon M-series recommended).

### Prerequisites

You must have the following installed:

- Python 3.9+
- Ollama: The easiest way to run local LLMs.
  - Installation: Download the installer for macOS or Linux from the Ollama website.

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/PromptPandemic-DynamicFormBuilder.git
   cd PromptPandemic-DynamicFormBuilder
   ```

2. **Install Python Dependencies**
   Create and activate a virtual environment, then install the necessary Python packages:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Download the LLama 3 Model (CRITICAL)**
   This project requires the Llama 3 model to be running locally.

   - Start the Ollama Server: Open a separate terminal window and run this command. Keep this window open for the entire time you are using the app.
     ```bash
     ollama serve
     ```

   - Pull the Model: In a third terminal window, download the model:
     ```bash
     ollama pull llama3
     ```

4. **Run the Application**
   Once the model pull is complete and the Ollama server is running, you can launch the app:
   ```bash
   streamlit run app.py
   ```

   Your browser will automatically open the application at http://localhost:8501.

## 🌟 Features

- **Natural Language Form Generation**: Describe the form you need, and AI creates it
- **Dynamic Form Rendering**: Automatically renders forms with appropriate input fields
- **Data Collection & Storage**: Captures and stores form submissions
- **Interactive Admin Dashboard**: Visualize submission data with charts and graphs
- **AI-Powered Insights**: Get intelligent analysis of your collected data
- **Modern UI/UX**: Clean, responsive dark-themed interface

## 🛠️ Technical Architecture

- **Frontend**: Streamlit with custom CSS styling
- **AI Integration**: LangChain with Ollama (Llama 3)
- **Data Visualization**: Plotly Express
- **Data Handling**: Pandas with CSV storage
- **Authentication**: Simple password protection for admin area

## 📋 Requirements

- Python 3.8+
- Ollama with Llama 3 model installed
- Required Python packages (see Installation)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/formforge.git
   cd formforge
   ```

2. Install required packages:
   ```bash
   pip install streamlit pandas plotly langchain-community langchain-core
   ```

3. Install Ollama and pull the Llama 3 model:
   ```bash
   # Follow instructions at https://ollama.com to install Ollama
   ollama pull llama3
   ```

## 💻 Usage

### Form Creation
1. Enter a natural language description of your desired form
2. Click "Generate Form" to create your form
3. The AI will analyze your request and generate appropriate form fields

### Admin Dashboard
1. Navigate to the Admin Dashboard from the navigation menu
2. Enter the password (default: "hackathon2025")
3. View submission data, charts, and AI-generated insights

## 🔒 Security Note

This application uses a simple password mechanism for the admin area. For production use, implement proper authentication and secure your data appropriately.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

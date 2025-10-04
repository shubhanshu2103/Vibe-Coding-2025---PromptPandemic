# FormForge - AI-Powered Dynamic Form Builder

FormForge is an intelligent form generation application that uses AI to create dynamic forms from natural language descriptions. Built with Streamlit and powered by Llama 3 via Ollama, this application allows users to describe the form they need in plain English and instantly generates a functional form with appropriate validation.

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
   git clone https://github.com/yourusername/Vibe-Coding-2025---PromptPandemic.git
   cd Vibe-Coding-2025---PromptPandemic
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

1. Start the Ollama server:
   ```bash
   ollama serve
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to http://localhost:8501

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

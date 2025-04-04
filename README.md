


# 🚀 Voice AI Assistant with ConversationRelay & LiteLLM

A voice AI assistant leveraging Twilio's Conversation Relay for real-time voice interactions, powered by multiple LLM providers via **LiteLLM**. This WebSocket server enables natural conversations over phone calls while maintaining session history.

## ✨ Features

- 🔄 **Real-time streaming responses** via WebSocket  
- 🤖 **Supports multiple LLM providers** (OpenAI, Anthropic, DeepSeek)  
- 🗂️ **Session-based conversation history**  
- 🎙️ **Optimized for voice interactions**  
- 🔌 **Seamless integration with Twilio's Conversation Relay**  

## 🛠️ Prerequisites

Ensure you have the following installed:

- Python **3.8+**
- **FastAPI**, **LiteLLM**, **Uvicorn**
- API keys for **LLM providers** (OpenAI, Anthropic, DeepSeek)
- A **Twilio account** and phone number

## 📦 Installation

### 1️⃣ Clone the repository  
```sh
git clone <your-repo-url>
cd ConvRelay-LiteLLM
```

### 2️⃣ Install dependencies  
```sh
pip install fastapi uvicorn litellm python-dotenv
```

### 3️⃣ Set up environment variables  
Create a `.env` file and add your API keys:  
```ini
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
DEEPSEEK_API_KEY=your_deepseek_key
```

## 🚀 Usage

### 1️⃣ Start the WebSocket server  
```sh
python server_litellm.py
```
The server will be available at:  
**`ws://localhost:8000/ws`**

### 2️⃣ Expose the local server with ngrok  
```sh
ngrok http 8000
```
Copy the **HTTPS URL** provided by ngrok.

### 3️⃣ Set up a TwiML Bin with the ngrok URL  
Go to **Twilio Console → TwiML Bins** and create a new TwiML Bin.  
Paste the following XML, replacing `<your-ngrok-domain>` with your actual ngrok domain:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <ConversationRelay url="wss://your-ngrok-domain.ngrok.io/ws" welcomeGreeting="Welcome message" />
  </Connect>
</Response>
```

### 4️⃣ Connect your Twilio phone number  
- Link your Twilio phone number to the **TwiML Bin** created in the previous step.

### 5️⃣ Make a call  
Dial your **Twilio number** and start interacting with the AI assistant! 🎉

---

## 🤝 Contributing  
We welcome contributions! Feel free to submit a pull request or open an issue.

## 📄 License  
This project is licensed under the MIT License.  

🚀 Happy coding!  


# Voice AI Assistant with ConversationRelay and LiteLLM

A voice AI assistant that leverages Twilio's Conversation Relay for real-time voice interactions, powered by multiple LLM providers through LiteLLM. This WebSocket server enables natural conversations over phone calls while maintaining session history.

## Features

- Real-time streaming responses via WebSocket
- Support for multiple LLM providers (OpenAI, Anthropic, DeepSeek)
- Session-based conversation history
- Optimized for voice interactions
- Easy integration with Twilio's Conversation Relay

## Prerequisites

- Python 3.8+
- FastAPI
- LiteLLM
- Uvicorn
- API keys for LLM providers
- Twilio account and phone number

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd voxray-py

2. Install dependencies:
pip install fastapi uvicorn litellm python-dotenv

3. Create a .env file with your API keys:
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
DEEPSEEK_API_KEY=your_deepseek_key

## Usage
1. Start the server: 
python server_litellm.py

The WebSocket server will be available at ws://localhost:8000/ws

2. Start ngrok tunnel:
ngrok http 8000

Copy the HTTPS URL provided by ngrok.

3. Set up a TwiML bin with the ngrok URL:
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <ConversationRelay url="wss://your-ngrok-domain.ngrok.io/ws" welcomeGreeting="Welcome message" />
  </Connect>
</Response>

4. Connect your Twilio phone number to this TwiML bin.

5. Place a call to your Twilio phone number.

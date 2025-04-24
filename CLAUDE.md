# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run server: `python server_litellm.py`
- Install dependencies: `pip install -r requirements.txt`
- Create virtual env: `python -m venv venv && source venv/bin/activate`

## Code Style Guidelines
- **Imports**: Use absolute imports, group standard library, third-party, and local imports
- **Formatting**: Follow PEP 8 guidelines with 4-space indentation
- **Types**: Use Python type hints (typing module) for function parameters and return values
- **Error Handling**: Use try/except blocks with specific exceptions
- **Naming**: Use snake_case for functions/variables, CamelCase for classes
- **Async**: Use `async/await` for asynchronous functions
- **Documentation**: Use docstrings for functions explaining parameters and return values
- **MCP Tools**: Follow FastMCP pattern for creating new Model Context Protocol tools

## Useful References
- Twilio ConversationRelay Documentation: https://www.twilio.com/docs/voice/twiml/connect/conversationrelay
- LiteLLM Documentation: https://docs.litellm.ai/docs/
- PipeCat Docs: https://github.com/pipecat-ai/pipecat

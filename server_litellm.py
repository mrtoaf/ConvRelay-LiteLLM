from fastapi import FastAPI, WebSocket
from typing import Dict, List, Optional
import json
import litellm
from litellm import acompletion
from litellm import experimental_mcp_client
from dotenv import load_dotenv
import uvicorn
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os


# Load environment variables
load_dotenv()
app = FastAPI()
sessions: Dict[str, List[dict]] = {}

# Setup MCP sessions and tools
async def setup_mcp():
    # We'll store all tools in this list
    all_tools = []
    
    # Setup weather MCP
    weather_params = StdioServerParameters(
        command="python3",
        args=["./weather.py"],  # Path to the weather MCP server
    )

    # Setup time MCP
    time_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_server_time", "--local-timezone=America/New_York"]
    )

    # First connect to weather MCP
    try:
        print("Connecting to weather MCP server...")
        async with stdio_client(weather_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                weather_tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
                print("Weather MCP TOOLS:", len(weather_tools))
                all_tools.extend(weather_tools)
    except Exception as e:
        print(f"Error initializing weather MCP: {e}")
        import traceback
        traceback.print_exc()
    
    # Then connect to time MCP
    try:
        print("Connecting to time MCP server...")
        async with stdio_client(time_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                time_tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
                print("Time MCP TOOLS:", len(time_tools))
                all_tools.extend(time_tools)
    except Exception as e:
        print(f"Error initializing time MCP: {e}")
        import traceback
        traceback.print_exc()
    
    if all_tools:
        print(f"Total MCP tools available: {len(all_tools)}")
        return True, all_tools
    else:
        print("No MCP tools were loaded successfully")
        return False, []

system_prompt = """
You are a helpful, concise, and reliable voice assistant. Your responses will be converted directly to speech, so always reply in plain, unformatted text that sounds natural when spoken.

When given a transcribed user request:

1. Silently fix likely transcription errors. Focus on intended meaning over literal wording. For example, interpret "buy milk two tomorrow" as "buy milk tomorrow."

2. Keep answers short and direct unless the user asks for more detail.

3. Prioritize clarity and accuracy. Avoid bullet points, formatting, or unnecessary filler.

4. Answer questions directly. Acknowledge or confirm commands.

5. If you don't understand the request, say: "I'm sorry, I didn't understand that."
"""

async def draft_response(model, messages):
    # Get all the combined tools
    success, all_tools = await setup_mcp()
    
    if not success or not all_tools:
        # Fallback to normal response if MCP setup fails
        print("Using normal completion without tools (MCP setup failed)")
        response = await acompletion(model=model, messages=messages, stream=True)
        async for part in response:
            if part.choices[0].delta.content is not None:
                yield {
                    "token": part.choices[0].delta.content,
                    "last": False,
                    "type": "text",
                }
        
        yield {
            "token": "",
            "type": "text",
            "last": True,
        }
        return
    
    try:
        print(f"Making LLM call with {len(all_tools)} MCP tools available...")
        
        # First call to the LLM with all tools available
        llm_response = await litellm.acompletion(
            model=model,
            messages=messages,
            tools=all_tools,
        )
        
        # Check if the LLM wants to use a tool
        if (llm_response.get("choices") and 
            llm_response["choices"][0]["message"].get("tool_calls")):
            
            # Extract the tool call
            openai_tool = llm_response["choices"][0]["message"]["tool_calls"][0]
            tool_name = openai_tool['function']['name']
            print(f"LLM decided to use tool: {tool_name}")
            
            # Determine which tool server to use based on the tool name
            if "time" in tool_name.lower():
                # Time MCP tool
                time_params = StdioServerParameters(
                    command="python",
                    args=["-m", "mcp_server_time", "--local-timezone=America/New_York"]
                )
                server_params = time_params
                print("Using time MCP server")
            else:
                # Default to weather MCP tool
                weather_params = StdioServerParameters(
                    command="python3",
                    args=["./weather.py"]
                )
                server_params = weather_params
                print("Using weather MCP server")
            
            # Connect to the appropriate MCP server and call the tool
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Call the tool using MCP client
                    call_result = await experimental_mcp_client.call_openai_tool(
                        session=session,
                        openai_tool=openai_tool,
                    )
                    
                    tool_result = str(call_result.content[0].text)
                    print(f"Tool result: {tool_result[:100]}...")
                    
                    # Add the tool results to messages
                    updated_messages = messages.copy()
                    updated_messages.append(llm_response["choices"][0]["message"])
                    updated_messages.append(
                        {
                            "role": "tool",
                            "content": tool_result,
                            "tool_call_id": openai_tool["id"],
                        }
                    )
                    
                    # Second call to LLM with tool results
                    final_response = await litellm.acompletion(
                        model=model,
                        messages=updated_messages,
                        stream=True,
                    )
                    
                    # Stream the final response for voice output
                    async for part in final_response:
                        if part.choices[0].delta.content is not None:
                            yield {
                                "token": part.choices[0].delta.content,
                                "last": False,
                                "type": "text",
                            }
        else:
            # If no tool is called, stream the original response
            print("No tool call needed, using regular response")
            response_text = llm_response["choices"][0]["message"]["content"]
            for char in response_text:
                yield {
                    "token": char,
                    "last": False,
                    "type": "text",
                }
                
    except Exception as e:
        print(f"Error in MCP or tool call: {e}")
        # Fallback to regular response without tools
        response = await acompletion(model=model, messages=messages, stream=True)
        async for part in response:
            if part.choices[0].delta.content is not None:
                yield {
                    "token": part.choices[0].delta.content,
                    "last": False,
                    "type": "text",
                }
    
    # Final token to indicate end of stream
    yield {
        "token": "",
        "type": "text",
        "last": True,
    }
          

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    call_sid: Optional[str] = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            # print("Message type: ", message["type"])
            print("ConvRelay Message: ", message)

            if message["type"] == "setup":
                call_sid = message["callSid"]
                print(f"Setup initiated for call {call_sid} from **{message['from'][-4:]}")
                

                # Initialize conversation history and system prompt
                sessions[str(call_sid)] = [
                    {"role": "system", "content": system_prompt}
                ]

            elif message["type"] == "prompt":
                prompt = message["voicePrompt"]
                print("Prompt: ", prompt)

                # Retrieve and update conversation history
                # Make sure call_sid is not None
                conversation = sessions[str(call_sid)] if call_sid is not None else []
                conversation.append({"role": "user", "content": prompt})

                # print(conversation)

                response = ""
                claude = "anthropic/claude-3-7-sonnet-20250219"
                openai = "openai/gpt-4o"

                async for event in draft_response(openai, messages=conversation):
                    await websocket.send_json(event)
                    response += event["token"]

                #save back
                print("AI response: ", response)
                conversation.append({"role": "assistant", "content": response})
                sessions[str(call_sid)] = conversation

            elif message["type"] == "interrupt":
                print("Response interrupted")

            elif message["type"] == "error":
                print("Error")

            else:
                print("Unknown message type")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if call_sid:
            sessions.pop(call_sid, None)
        print("Client has disconnected.")

if __name__ == "__main__":
    import asyncio

    # Run the inspection
    asyncio.run(setup_mcp())
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
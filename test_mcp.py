import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from litellm import experimental_mcp_client

async def test_weather_mcp():
    # Set up MCP server parameters
    server_params = StdioServerParameters(
        command="uv",
        args=["./weather.py"]  # Path to your weather.py MCP server
    )

    server_params2 = StdioServerParameters(
        command="python",
        args=["-m", "mcp_server_time", "--local-timezone=America/New_York"]
    )
    
    try:
        print("Connecting to weather and date MCP server...")
        # Connect to MCP server
        client = stdio_client(server_params, server_params2)
        read, write = await client.__aenter__()
        
        # Create and initialize session
        print("Creating MCP session...")
        session = ClientSession(read, write)
        await session.initialize()
        
        # Load MCP tools
        print("Loading MCP tools...")
        tools = await experimental_mcp_client.load_mcp_tools(session=session, format="mcp")
        
        # Print available tools
        print(f"Successfully loaded {len(tools)} MCP tools:")
        for i, tool in enumerate(tools):
            print(f"Tool {i+1}: {tool.function.name} - {tool.function.description}")
            
        # Test a simple tool call - get weather alerts for CA
        if any(tool.function.name == "get_alerts" for tool in tools):
            print("\nTesting get_alerts tool...")
            tool_result = await session.call_tool(
                tool_id="get_alerts",
                params={"state": "CA"}
            )
            print(f"Tool result: {tool_result}")
        
        print("\nMCP test completed successfully!")
        await client.__aexit__(None, None, None)
        
    except Exception as e:
        print(f"Error testing MCP: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_weather_mcp())
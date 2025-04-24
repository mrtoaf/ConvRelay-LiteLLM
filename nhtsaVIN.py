from typing import Any, Optional
import httpx
import re
from mcp.server.fastmcp import FastMCP

# Init FastMCP server
mcp = FastMCP("nhtsa-vin-number")

# Constants
NHTSA_API_BASE = "https://vpic.nhtsa.dot.gov/api"
USER_AGENT = "vin-decoder/1.0"

async def make_nhtsa_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NHTSA API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


def format_vehicle_info(data: dict) -> str:
    """Format vehicle information into a readable string, filtering out empty fields."""
    if not data or "Results" not in data or not data["Results"]:
        return "No vehicle information found for this VIN."
    
    results = data["Results"][0]
    
    # Define important fields we want to keep, in display order
    # Keep doors, seats, and seat rows as requested
    important_fields = [
        "VIN", "Make", "Model", "ModelYear", "Series", "Trim", 
        "Manufacturer", "BodyClass", "VehicleType", "DriveType",
        "EngineConfiguration", "EngineCylinders", "EngineHP", "Displacement", 
        "FuelTypePrimary", "Doors", "Seats", "SeatRows", "ErrorText"
    ]
    
    # Collect only important non-empty fields
    vehicle_info = []
    for field in important_fields:
        if field in results and results[field] not in (None, "", "/", "Not Applicable"):
            vehicle_info.append(f"{field}: {results[field]}")
    
    # Skip safety features - just use the important fields we defined
    
    return "\n".join(vehicle_info)


@mcp.tool()
async def decode_vin(vin: str = None, modelyear: Optional[str] = None) -> str:
    """Decode a Vehicle Identification Number (VIN) using NHTSA's API.
    
    IMPORTANT: Always use this tool for any potential VIN, even if it appears malformed! 
    This tool automatically handles speech-to-text artifacts including chunking, spaces, 
    commas, periods, and other non-alphanumeric characters.
    
    Args:
        vin: The Vehicle Identification Number to decode (can include STT artifacts like spaces, commas, periods)
        modelyear: Optional model year if known (improves accuracy)
    """
    if not vin:
        return "Please provide a VIN number."
        
    # Enhanced cleaning to handle a wider range of speech-to-text artifacts
    # Remove ALL non-alphanumeric characters (punctuation, spaces, special chars)
    clean_vin = re.sub(r'[^A-Za-z0-9]', '', vin)
    
    # Convert to uppercase as VINs are case-insensitive
    clean_vin = clean_vin.upper()
    
    # Validate basic VIN structure (most VINs are 17 characters)
    if len(clean_vin) != 17:
        # If not 17 characters, still proceed but add a note
        note = f"\nNote: The VIN you provided has {len(clean_vin)} characters instead of the standard 17 characters. Results may be less accurate."
    else:
        note = ""
    
    # Simple confirmation without any fancy formatting
    confirmation = f"Looking up VIN: {clean_vin}{note}"
    
    # Construct the URL with the flat format
    url = f"{NHTSA_API_BASE}/vehicles/DecodeVinValues/{clean_vin}?format=json"
    
    # Add model year if provided
    if modelyear:
        # Clean model year too (in case it has punctuation)
        clean_modelyear = re.sub(r'[,.\s\-_]', '', str(modelyear))
        url += f"&modelyear={clean_modelyear}"
    
    data = await make_nhtsa_request(url)
    
    if not data:
        return "Unable to fetch vehicle information for this VIN."
    
    vehicle_info = format_vehicle_info(data)
    
    # Simple header without model year emphasis
    header = "Here's the vehicle information:"
    
    return f"{confirmation}\n\n{header}\n\n{vehicle_info}"


if __name__ == "__main__":
    # Init and run the server
    mcp.run(transport='stdio')
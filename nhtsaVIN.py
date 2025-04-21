from typing import Any, Optional
import httpx
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
    important_fields = [
        "VIN", "Make", "Model", "ModelYear", "Series", "Trim", 
        "Manufacturer", "PlantCountry", "PlantCity", "PlantState",
        "BodyClass", "VehicleType", "DriveType",
        "EngineConfiguration", "EngineCylinders", "EngineHP", "DisplacementL", 
        "FuelTypePrimary", "Doors", "Seats", "SeatRows",
        "GVWR", "BasePrice", "ErrorText"
    ]
    
    # Collect only important non-empty fields
    vehicle_info = []
    for field in important_fields:
        if field in results and results[field] not in (None, "", "/", "Not Applicable"):
            vehicle_info.append(f"{field}: {results[field]}")
    
    # Add any other non-empty safety features
    safety_features = []
    for key, value in results.items():
        if key not in important_fields and value not in (None, "", "/", "Not Applicable"):
            if "Standard" in str(value) or "Optional" in str(value):
                safety_features.append(f"{key}: {value}")
    
    if safety_features:
        vehicle_info.append("\nSafety Features:")
        vehicle_info.extend(safety_features)
    
    return "\n".join(vehicle_info)


@mcp.tool()
async def decode_vin(vin: str, modelyear: Optional[str] = None) -> str:
    """Decode a Vehicle Identification Number (VIN) using NHTSA's API.
    
    Args:
        vin: Full or partial VIN (use * for unknown characters in partial VINs)
        modelyear: Optional model year (recommended for more accurate results)
    """
    # Construct the URL with the flat format
    url = f"{NHTSA_API_BASE}/vehicles/DecodeVinValues/{vin}?format=json"
    
    # Add model year if provided
    if modelyear:
        url += f"&modelyear={modelyear}"
    
    data = await make_nhtsa_request(url)
    
    if not data:
        return "Unable to fetch vehicle information."
    
    return format_vehicle_info(data)


if __name__ == "__main__":
    # Init and run the server
    mcp.run(transport='stdio')
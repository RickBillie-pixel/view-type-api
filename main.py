"""
View Type API - Detects view type from extracted vector data
Analyzes text content to determine if it's a floor plan, section, detail, etc.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("view_type_api")

app = FastAPI(
    title="View Type Detection API",
    description="Detects view type from extracted vector and text data",
    version="1.0.0",
)

class PageData(BaseModel):
    page_number: int
    drawings: List[Dict[str, Any]]
    texts: List[Dict[str, Any]]

class ViewTypeRequest(BaseModel):
    pages: List[PageData]

@app.post("/detect-view-type/")
async def detect_view_type(request: ViewTypeRequest):
    """
    Detect view type from extracted vector data
    
    Args:
        request: JSON with pages containing drawings and texts
        
    Returns:
        JSON with view type detection results for each page
    """
    try:
        logger.info(f"Detecting view types for {len(request.pages)} pages")
        
        results = []
        
        for page_data in request.pages:
            logger.info(f"Analyzing page {page_data.page_number}")
            
            # Extract text content for analysis
            texts = [t["text"].upper() for t in page_data.texts]
            
            # Rule-based view type detection
            view_type_result = _detect_view_type_from_texts(texts)
            
            results.append({
                "page_number": page_data.page_number,
                "view_type": view_type_result["view_type"],
                "confidence": view_type_result["confidence"],
                "reason": view_type_result["reason"]
            })
        
        logger.info(f"Successfully detected view types for {len(results)} pages")
        return {"pages": results}
        
    except Exception as e:
        logger.error(f"Error detecting view types: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _detect_view_type_from_texts(texts: List[str]) -> Dict[str, Any]:
    """
    Detect view type based on text content using rule-based approach
    
    Args:
        texts: List of uppercase text strings
        
    Returns:
        Dictionary with view_type, confidence, and reason
    """
    # Room names indicate floor plan
    room_keywords = ["WOONKAMER", "SLAAPKAMER", "KITCHEN", "BEDROOM", "BATHROOM", 
                    "TOILET", "WC", "KEUKEN", "BADKAMER", "GANG", "HAL"]
    
    if any(any(room in t for room in room_keywords) for t in texts):
        return {
            "view_type": "floor_plan", 
            "confidence": 1.0, 
            "reason": "Room names detected"
        }
    
    # Section keywords
    section_keywords = ["DOORSNEDE", "SECTION", "SNEDE", "PROFIEL"]
    if any(any(keyword in t for keyword in section_keywords) for t in texts):
        return {
            "view_type": "section", 
            "confidence": 1.0, 
            "reason": "Section keyword detected"
        }
    
    # Detail keywords
    detail_keywords = ["DETAIL", "DETAILLERING", "UITWERKING"]
    if any(any(keyword in t for keyword in detail_keywords) for t in texts):
        return {
            "view_type": "detail", 
            "confidence": 1.0, 
            "reason": "Detail keyword detected"
        }
    
    # Installation symbols indicate installation plan
    installation_keywords = ["WCD", "LICHTPUNT", "MV", "CAI", "SCHAKELAAR", 
                           "STOPCONTACT", "VERLICHTING", "VENTILATIE"]
    if any(any(inst in t for inst in installation_keywords) for t in texts):
        return {
            "view_type": "installation", 
            "confidence": 1.0, 
            "reason": "Installation symbol detected"
        }
    
    # Component table keywords
    table_keywords = ["MERK", "TYPE", "AFMETING", "MATERIAAL", "SPECIFICATIE"]
    if any(all(keyword in t for keyword in ["MERK", "TYPE", "AFMETING"]) for t in texts):
        return {
            "view_type": "component_table", 
            "confidence": 1.0, 
            "reason": "Component table detected"
        }
    
    # Elevation keywords
    elevation_keywords = ["GEVEL", "ELEVATION", "VOORGEVEL", "ZIJGEVEL"]
    if any(any(keyword in t for keyword in elevation_keywords) for t in texts):
        return {
            "view_type": "elevation", 
            "confidence": 1.0, 
            "reason": "Elevation keyword detected"
        }
    
    # Default case
    return {
        "view_type": "unknown", 
        "confidence": 0.0, 
        "reason": "No matching keywords or layout patterns detected"
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "view-type-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
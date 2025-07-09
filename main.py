"""
View Type API - Advanced View Type Detection
Implements complete knowledge base for construction drawing view type detection
Analyzes text content and layout to determine drawing type with high accuracy
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("view_type_api")

# Knowledge Base - View Type Detection Patterns (Rule 1)
VIEW_TYPE_PATTERNS = {
    "floor_plan": [
        r"woonkamer|slaapkamer|keuken|badkamer|toilet|gang|berging|garage|kelder|zolder",
        r"living|bedroom|kitchen|bathroom|hall|storage|garage|basement|attic",
        r"woon|slaap|bad|wc|gang|berg|gar|kel|zol",
        r"kamer|room|suite|studio|appartement|apartment"
    ],
    "section": [
        r"doorsnede|section|doorsnee|sectie|profiel|profile",
        r"a-a|b-b|c-c|d-d|e-e|f-f|g-g|h-h",
        r"doorsnede|section|cut|snede"
    ],
    "detail": [
        r"detail|detaal|uitvergroting|enlargement|detailering",
        r"1:10|1:5|1:2|1:1|1:20|1:50",
        r"detail|detail|uitwerk|enlargement"
    ],
    "installation": [
        r"wcd|cai|mv|schakelaar|thermostaat|lichtpunt",
        r"electrical|plumbing|ventilation|switch|thermostat|light",
        r"elektra|sanitair|ventilatie|verwarming|koeling",
        r"socket|outlet|switch|thermostat|light|ventilation"
    ],
    "component_table": [
        r"merk|type|afmeting|brand|dimension|materiaal",
        r"tabel|table|specificatie|specification|lijst|list",
        r"product|component|element|onderdeel|part"
    ],
    "elevation": [
        r"gevel|elevation|voorgevel|achtergevel|zijgevel",
        r"facade|front|rear|side|elevation",
        r"gevel|elevation|facade|voor|achter|zij"
    ],
    "site_plan": [
        r"terrein|site|perceel|plot|grondplan",
        r"terrain|site|plot|ground|landscape",
        r"terrein|site|perceel|grond|land"
    ],
    "structural": [
        r"constructie|structure|fundering|foundation",
        r"beton|concrete|staal|steel|hout|wood",
        r"constructie|structure|fundering|beton|staal"
    ]
}

# Scale detection patterns
SCALE_PATTERNS = [
    r"1:(\d+)",  # 1:100, 1:50, etc.
    r"schaal\s*(\d+):(\d+)",  # schaal 1:100
    r"scale\s*(\d+):(\d+)"   # scale 1:100
]

app = FastAPI(
    title="Advanced View Type Detection API",
    description="Detects view type from extracted vector and text data using knowledge base",
    version="2.0.0",
)

class PageData(BaseModel):
    page_number: int
    drawings: Dict[str, Any]
    texts: List[Dict[str, Any]]

class ViewTypeRequest(BaseModel):
    pages: List[PageData]

class ViewTypeResponse(BaseModel):
    pages: List[Dict[str, Any]]

@app.post("/detect-view-type/", response_model=ViewTypeResponse)
async def detect_view_type(request: ViewTypeRequest):
    """
    Detect view type from extracted vector data using knowledge base
    
    Args:
        request: JSON with pages containing drawings and texts
        
    Returns:
        JSON with view type detection results for each page
    """
    try:
        logger.info(f"Detecting view types for {len(request.pages)} pages using knowledge base")
        
        results = []
        
        for page_data in request.pages:
            logger.info(f"Analyzing page {page_data.page_number}")
            
            # Extract text content for analysis
            texts = [t["text"] for t in page_data.texts]
            
            # Advanced view type detection using knowledge base
            view_type_result = _detect_view_type_advanced(texts, page_data.drawings)
            
            results.append({
                "page_number": page_data.page_number,
                "view_type": view_type_result["view_type"],
                "confidence": view_type_result["confidence"],
                "reason": view_type_result["reason"],
                "scale": view_type_result["scale"],
                "detected_keywords": view_type_result["detected_keywords"]
            })
        
        logger.info(f"Successfully detected view types for {len(results)} pages")
        return {"pages": results}
        
    except Exception as e:
        logger.error(f"Error detecting view types: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _detect_view_type_advanced(texts: List[str], drawings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Advanced view type detection using knowledge base rules
    
    Args:
        texts: List of text strings
        drawings: Drawing data including lines, rectangles, curves
        
    Returns:
        Dictionary with view_type, confidence, reason, scale, and detected_keywords
    """
    all_text = " ".join([text.lower() for text in texts])
    detected_keywords = []
    max_confidence = 0.0
    best_view_type = "unknown"
    best_reason = "No matching patterns detected"
    
    # Check each view type pattern
    for view_type, patterns in VIEW_TYPE_PATTERNS.items():
        confidence = 0.0
        matched_keywords = []
        
        for pattern in patterns:
            matches = re.findall(pattern, all_text)
            if matches:
                matched_keywords.extend(matches)
                confidence += 0.3  # Base confidence per pattern match
        
        # Additional confidence based on layout analysis
        if view_type == "floor_plan":
            # Check for room-like rectangles
            rectangles = drawings.get("rectangles", [])
            large_rectangles = [r for r in rectangles if r.get("area", 0) > 10000]
            if len(large_rectangles) > 2:
                confidence += 0.2
                matched_keywords.append("room_layout")
        
        elif view_type == "section":
            # Check for vertical lines (typical in sections)
            lines = drawings.get("lines", [])
            vertical_lines = [l for l in lines if abs(l.get("p2", {}).get("x", 0) - l.get("p1", {}).get("x", 0)) < 10]
            if len(vertical_lines) > 5:
                confidence += 0.2
                matched_keywords.append("vertical_lines")
        
        elif view_type == "detail":
            # Check for small scale indicators
            scale_matches = re.findall(r"1:(\d+)", all_text)
            if scale_matches:
                scales = [int(s) for s in scale_matches]
                if any(s <= 20 for s in scales):  # 1:20 or larger scale
                    confidence += 0.3
                    matched_keywords.append(f"large_scale_1:{min(scales)}")
        
        elif view_type == "installation":
            # Check for electrical symbols
            electrical_symbols = ["wcd", "lichtpunt", "schakelaar", "thermostaat"]
            if any(symbol in all_text for symbol in electrical_symbols):
                confidence += 0.2
                matched_keywords.append("electrical_symbols")
        
        # Update best result if confidence is higher
        if confidence > max_confidence:
            max_confidence = confidence
            best_view_type = view_type
            best_reason = f"Pattern matches: {', '.join(matched_keywords)}"
            detected_keywords = matched_keywords
    
    # Extract scale information
    scale = _extract_scale_from_texts(texts)
    
    # Cap confidence at 1.0
    max_confidence = min(max_confidence, 1.0)
    
    return {
        "view_type": best_view_type,
        "confidence": round(max_confidence, 2),
        "reason": best_reason,
        "scale": scale,
        "detected_keywords": detected_keywords
    }

def _extract_scale_from_texts(texts: List[str]) -> Dict[str, Any]:
    """
    Extract scale information from texts
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary with scale information
    """
    all_text = " ".join(texts)
    
    # Look for scale patterns
    for pattern in SCALE_PATTERNS:
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        if matches:
            if isinstance(matches[0], tuple) and len(matches[0]) == 2:  # schaal 1:100 format
                numerator, denominator = int(matches[0][0]), int(matches[0][1])
                return {
                    "scale_ratio": f"{numerator}:{denominator}",
                    "scale_value": numerator / denominator,
                    "source": "explicit_scale"
                }
            else:  # 1:100 format
                denominator = int(matches[0])
                return {
                    "scale_ratio": f"1:{denominator}",
                    "scale_value": 1 / denominator,
                    "source": "explicit_scale"
                }
    
    # Look for dimension-based scale hints
    dimension_patterns = [
        r"(\d+)\s*mm",  # 3600 mm
        r"(\d+)\s*cm",  # 360 cm
        r"(\d+[,.]?\d*)\s*m"  # 3.6 m
    ]
    
    for pattern in dimension_patterns:
        matches = re.findall(pattern, all_text)
        if matches:
            # Assume typical room dimension for scale calculation
            typical_room_size = 3.6  # meters
            detected_size = float(matches[0].replace(',', '.'))
            
            if "mm" in pattern:
                detected_size = detected_size / 1000
            elif "cm" in pattern:
                detected_size = detected_size / 100
            
            # Calculate approximate scale
            if detected_size > 0:
                scale_value = detected_size / typical_room_size
                return {
                    "scale_ratio": f"1:{int(1/scale_value)}",
                    "scale_value": scale_value,
                    "source": "dimension_inference",
                    "detected_dimension": detected_size
                }
    
    return {
        "scale_ratio": "unknown",
        "scale_value": None,
        "source": "not_detected"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Advanced View Type Detection API",
        "version": "2.0.0",
        "endpoints": {
            "/detect-view-type/": "Detect view type using knowledge base",
            "/health/": "Health check"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "view-type-api",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
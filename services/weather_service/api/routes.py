"""
FastAPI routes for weather service.
"""
import logging

from fastapi import APIRouter, HTTPException, status

from services.common.dependencies import AuthenticatedUser

from ..core.weather import get_weather_conditions
from .schemas import (
    WeatherConditionsRequest,
    WeatherConditionsResponse,
    SolarData,
    WeatherData,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/weather", tags=["weather"])


@router.post("/conditions", response_model=WeatherConditionsResponse)
async def get_conditions(
    request: WeatherConditionsRequest,
    current_user: AuthenticatedUser,
):
    """
    Get weather and solar conditions for a given location and datetime.
    
    Determines lighting (lum) and atmospheric (atm) conditions based on:
    - Solar position (Sunrise-Sunset.org API)
    - Weather data (Open-Meteo API with WMO codes)
    
    Args:
        request: WeatherConditionsRequest with coordinates, datetime, and urban indicator
        current_user: Authenticated user (required for access)
    
    Returns:
        WeatherConditionsResponse with lum, atm, and raw weather/solar data
    
    Raises:
        HTTPException: If request validation fails
    """
    try:
        result = get_weather_conditions(
            lat=request.latitude,
            lon=request.longitude,
            current_datetime=request.dt,
            agg_=request.agg_,
        )
        
        # Convert to response models
        solar_data = None
        if result.get("solar_data"):
            solar_data = SolarData(**result["solar_data"])
        
        weather_data = None
        if result.get("weather_data"):
            weather_data = WeatherData(**result["weather_data"])
        
        return WeatherConditionsResponse(
            lum=result["lum"],
            atm=result["atm"],
            solar_data=solar_data,
            weather_data=weather_data,
            timezone=result["timezone"],
            error=result.get("error"),
        )
    
    except Exception as e:
        logger.error(f"Weather conditions error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Weather service error: {str(e)}",
        )

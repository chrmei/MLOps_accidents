"""
Pydantic schemas for weather service API.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class WeatherConditionsRequest(BaseModel):
    """Request payload for getting weather conditions."""

    latitude: float = Field(..., description="Latitude coordinate", ge=-90, le=90)
    longitude: float = Field(..., description="Longitude coordinate", ge=-180, le=180)
    dt: datetime = Field(..., description="Datetime for weather conditions (timezone-aware or naive)", alias="datetime")
    agg_: int = Field(default=2, description="Urban area indicator (1=outside urban, 2=inside urban)", ge=1, le=2)


class SolarData(BaseModel):
    """Solar data from Sunrise-Sunset.org API."""

    sunrise: Optional[str] = Field(None, description="Sunrise time (ISO 8601)")
    sunset: Optional[str] = Field(None, description="Sunset time (ISO 8601)")
    solar_noon: Optional[str] = Field(None, description="Solar noon time (ISO 8601)")
    civil_twilight_begin: Optional[str] = Field(None, description="Civil twilight begin (ISO 8601)")
    civil_twilight_end: Optional[str] = Field(None, description="Civil twilight end (ISO 8601)")


class WeatherData(BaseModel):
    """Weather data from Open-Meteo API."""

    weathercode: Optional[int] = Field(None, description="WMO weather code")
    windspeed_10m: Optional[float] = Field(None, description="Wind speed at 10m (km/h)")
    cloudcover: Optional[float] = Field(None, description="Cloud cover percentage")
    temperature_2m: Optional[float] = Field(None, description="Temperature at 2m (°C)")
    time: Optional[str] = Field(None, description="Time of the weather data (ISO 8601)")


class WeatherConditionsResponse(BaseModel):
    """Response payload for weather conditions."""

    lum: int = Field(..., description="Lighting condition (1-5)")
    atm: int = Field(..., description="Atmospheric condition (-1, 1-9)")
    solar_data: Optional[SolarData] = Field(None, description="Solar data")
    weather_data: Optional[WeatherData] = Field(None, description="Weather data")
    timezone: str = Field(..., description="Timezone used for calculations")
    error: Optional[str] = Field(None, description="Error message if any API failed")


class ErrorResponse(BaseModel):
    """Error response payload."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error detail message")

"""Core weather and solar data fetching and condition determination logic."""
from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

import httpx
from timezonefinder import TimezoneFinder

logger = logging.getLogger(__name__)

# API endpoints
SUNRISE_SUNSET_API = "https://api.sunrise-sunset.org/json"
OPEN_METEO_FORECAST_API = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"

# Wind speed threshold for strong wind (km/h)
STRONG_WIND_THRESHOLD = 50.0

# Initialize timezone finder
_tz_finder = TimezoneFinder()


def get_timezone_from_coords(lat: float, lon: float) -> str:
    """
    Get timezone string for given coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Timezone string (e.g., 'Europe/Paris') or 'UTC' as fallback
    """
    try:
        tz_name = _tz_finder.timezone_at(lat=lat, lng=lon)
        if tz_name:
            return tz_name
        logger.warning(f"Could not determine timezone for coordinates ({lat}, {lon}), using UTC")
        return "UTC"
    except Exception as e:
        logger.error(f"Error determining timezone: {e}", exc_info=True)
        return "UTC"


def fetch_solar_data(lat: float, lon: float, date_obj: date) -> Optional[dict]:
    """
    Fetch solar data from Sunrise-Sunset.org API.
    
    Args:
        lat: Latitude
        lon: Longitude
        date_obj: Date to fetch solar data for
        
    Returns:
        Dict with solar data (sunrise, sunset, civil_twilight_begin, etc.) or None if failed
    """
    try:
        url = SUNRISE_SUNSET_API
        params = {
            "lat": lat,
            "lng": lon,
            "date": date_obj.isoformat(),
            "formatted": 0  # Get ISO 8601 format
        }
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "OK":
                results = data.get("results", {})
                return {
                    "sunrise": results.get("sunrise"),
                    "sunset": results.get("sunset"),
                    "solar_noon": results.get("solar_noon"),
                    "civil_twilight_begin": results.get("civil_twilight_begin"),
                    "civil_twilight_end": results.get("civil_twilight_end"),
                }
            else:
                logger.error(f"Sunrise-Sunset API returned error: {data.get('status')}")
                return None
                
    except httpx.RequestError as e:
        logger.error(f"Error fetching solar data: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching solar data: {e}", exc_info=True)
        return None


def fetch_weather_data(lat: float, lon: float, datetime_obj: datetime, timezone: str) -> Optional[dict]:
    """
    Fetch weather data from Open-Meteo API (forecast or historical).
    
    Automatically selects the appropriate API endpoint based on the date:
    - Historical API for past dates (before today)
    - Forecast API for current/future dates
    
    Args:
        lat: Latitude
        lon: Longitude
        datetime_obj: Datetime to fetch weather for
        timezone: Timezone string (e.g., 'Europe/Paris')
        
    Returns:
        Dict with weather data (weathercode, windspeed_10m, cloudcover, temperature) or None if failed
    """
    try:
        # Ensure datetime_obj is timezone-aware in the target timezone
        if datetime_obj.tzinfo is None:
            datetime_obj = datetime_obj.replace(tzinfo=ZoneInfo(timezone))
        else:
            datetime_obj = datetime_obj.astimezone(ZoneInfo(timezone))
        
        # Determine which API to use based on date
        # Compare the requested date against today's date (in the same timezone)
        # Use historical API for dates more than 1 day in the past
        today_in_tz = datetime.now(ZoneInfo(timezone)).date()
        request_date = datetime_obj.date()
        
        # Use historical API if date is in the past (more than 1 day ago)
        # Forecast API typically covers today and future dates
        if request_date < today_in_tz - timedelta(days=1):
            url = OPEN_METEO_HISTORICAL_API
            logger.debug(f"Using historical API for date {request_date} (today is {today_in_tz})")
        else:
            url = OPEN_METEO_FORECAST_API
            logger.debug(f"Using forecast API for date {request_date} (today is {today_in_tz})")
        
        # Format datetime for API (YYYY-MM-DDTHH:MM format)
        # Use the provided datetime_obj (from dashboard) for the API request
        start_date = datetime_obj.date()
        end_date = start_date + timedelta(days=1)
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "weathercode,windspeed_10m,cloudcover,temperature_2m",
            "timezone": timezone,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, params=params)
            
            # Check for API error responses (like date out of range)
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("reason", "Bad Request")
                    logger.error(f"Open-Meteo API error: {error_msg}")
                    return None
                except Exception:
                    pass
            
            response.raise_for_status()
            data = response.json()
            
            # Check for error in response body
            if data.get("error"):
                error_msg = data.get("reason", "Unknown error")
                logger.error(f"Open-Meteo API returned error: {error_msg}")
                return None
            
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            weathercodes = hourly.get("weathercode", [])
            windspeeds = hourly.get("windspeed_10m", [])
            cloudcovers = hourly.get("cloudcover", [])
            temperatures = hourly.get("temperature_2m", [])
            
            if not times:
                logger.warning("No hourly data returned from Open-Meteo API")
                return None
            
            # Find closest time index
            # Open-Meteo returns times in local timezone (without timezone info) when timezone parameter is set
            # Parse as naive datetime and make timezone-aware in the target timezone
            closest_idx = 0
            min_diff = float('inf')
            for idx, time_str in enumerate(times):
                try:
                    # Parse time string
                    # When timezone parameter is set, Open-Meteo returns times as "YYYY-MM-DDTHH:MM" (no timezone)
                    # When timezone is not set, times are in UTC with 'Z' suffix
                    if time_str.endswith('Z') or ('+' in time_str and len(time_str.split('+')) > 1):
                        # Has UTC timezone info (Z) or explicit offset (+HH:MM)
                        time_dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    else:
                        # No timezone info - it's in local timezone (as specified by timezone parameter)
                        # Format: "YYYY-MM-DDTHH:MM"
                        time_dt = datetime.fromisoformat(time_str)
                        # Make it timezone-aware in the target timezone
                        time_dt = time_dt.replace(tzinfo=ZoneInfo(timezone))
                    
                    diff = abs((time_dt - datetime_obj).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = idx
                except Exception as e:
                    logger.debug(f"Error parsing time string '{time_str}': {e}")
                    continue
            
            # Log the selected time for debugging
            if times and closest_idx < len(times):
                selected_time_str = times[closest_idx]
                # Parse the selected time to show both local and UTC
                if selected_time_str.endswith('Z') or ('+' in selected_time_str and len(selected_time_str.split('+')) > 1):
                    selected_time_dt = datetime.fromisoformat(selected_time_str.replace('Z', '+00:00'))
                else:
                    selected_time_dt = datetime.fromisoformat(selected_time_str).replace(tzinfo=ZoneInfo(timezone))
                
                logger.info(
                    f"Weather data selection: requested={datetime_obj.isoformat()} ({timezone}), "
                    f"selected={selected_time_str} ({timezone} = {selected_time_dt.astimezone(ZoneInfo('UTC')).isoformat()} UTC), "
                    f"diff={min_diff/3600:.2f} hours, temp={temperatures[closest_idx] if closest_idx < len(temperatures) else 'N/A'}°C"
                )
            
            # Extract data for closest hour
            weathercode = weathercodes[closest_idx] if closest_idx < len(weathercodes) else None
            windspeed = windspeeds[closest_idx] if closest_idx < len(windspeeds) else None
            cloudcover = cloudcovers[closest_idx] if closest_idx < len(cloudcovers) else None
            temperature = temperatures[closest_idx] if closest_idx < len(temperatures) else None
            
            # Convert the selected time to UTC for consistency with request format
            selected_time_str = times[closest_idx] if closest_idx < len(times) else None
            time_utc = None
            if selected_time_str:
                try:
                    # Parse the time string (in local timezone when timezone parameter is set)
                    if selected_time_str.endswith('Z') or ('+' in selected_time_str and len(selected_time_str.split('+')) > 1):
                        time_dt = datetime.fromisoformat(selected_time_str.replace('Z', '+00:00'))
                    else:
                        # No timezone info - it's in local timezone
                        time_dt = datetime.fromisoformat(selected_time_str).replace(tzinfo=ZoneInfo(timezone))
                    # Convert to UTC for response
                    time_utc = time_dt.astimezone(ZoneInfo('UTC')).isoformat()
                except Exception as e:
                    logger.debug(f"Error converting time to UTC: {e}")
                    time_utc = selected_time_str  # Fallback to original
            
            return {
                "weathercode": weathercode,
                "windspeed_10m": windspeed,
                "cloudcover": cloudcover,
                "temperature_2m": temperature,
                "time": time_utc or selected_time_str,  # Return UTC time for consistency
            }
            
    except httpx.RequestError as e:
        logger.error(f"Network error fetching weather data: {e}", exc_info=True)
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching weather data: {e.response.status_code} - {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching weather data: {e}", exc_info=True)
        return None


def parse_iso_datetime(iso_str: str, timezone: str) -> Optional[datetime]:
    """
    Parse ISO 8601 datetime string and convert to timezone-aware datetime.
    
    Args:
        iso_str: ISO 8601 datetime string (e.g., "2024-01-01T12:00:00+00:00")
        timezone: Target timezone string
        
    Returns:
        Datetime object in target timezone or None if parsing fails
    """
    try:
        # Parse ISO string (may have +00:00 or Z suffix)
        if iso_str.endswith('Z'):
            dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(iso_str)
        
        # Convert to UTC if not already timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        else:
            dt = dt.astimezone(ZoneInfo("UTC"))
        
        # Convert to target timezone
        target_tz = ZoneInfo(timezone)
        dt = dt.astimezone(target_tz)
        
        return dt
    except Exception as e:
        logger.error(f"Error parsing datetime {iso_str}: {e}", exc_info=True)
        return None


def determine_lighting_condition(
    solar_data: Optional[dict],
    current_datetime: datetime,
    agg_: int,
    timezone: str
) -> int:
    """
    Determine lighting condition (lum) based on solar position.
    
    Args:
        solar_data: Solar data dict from API or None
        current_datetime: Current datetime (timezone-aware)
        agg_: Urban area indicator (1 = outside urban, 2 = inside urban)
        timezone: Timezone string
        
    Returns:
        lum value (1-5): 1=daylight, 2=twilight, 3=night without lighting, 5=night with lighting
    """
    if not solar_data:
        # Fallback: use defaults based on time of day
        hour = current_datetime.hour
        if 6 <= hour < 20:
            return 1  # Assume daylight
        else:
            # Default to urban lighting if agg_ is 2, otherwise rural
            return 5 if agg_ == 2 else 3
    
    try:
        # Parse solar event times
        sunrise_str = solar_data.get("sunrise")
        sunset_str = solar_data.get("sunset")
        civil_twilight_begin_str = solar_data.get("civil_twilight_begin")
        civil_twilight_end_str = solar_data.get("civil_twilight_end")
        
        if not all([sunrise_str, sunset_str, civil_twilight_begin_str, civil_twilight_end_str]):
            logger.warning("Incomplete solar data, using time-based fallback")
            hour = current_datetime.hour
            return 1 if 6 <= hour < 20 else (5 if agg_ == 2 else 3)
        
        # Convert solar times to timezone-aware datetimes
        sunrise = parse_iso_datetime(sunrise_str, timezone)
        sunset = parse_iso_datetime(sunset_str, timezone)
        civil_twilight_begin = parse_iso_datetime(civil_twilight_begin_str, timezone)
        civil_twilight_end = parse_iso_datetime(civil_twilight_end_str, timezone)
        
        if not all([sunrise, sunset, civil_twilight_begin, civil_twilight_end]):
            logger.warning("Failed to parse solar times, using time-based fallback")
            hour = current_datetime.hour
            return 1 if 6 <= hour < 20 else (5 if agg_ == 2 else 3)
        
        # Ensure current_datetime is timezone-aware
        if current_datetime.tzinfo is None:
            current_datetime = current_datetime.replace(tzinfo=ZoneInfo(timezone))
        else:
            current_datetime = current_datetime.astimezone(ZoneInfo(timezone))
        
        # Determine lighting condition
        if sunrise <= current_datetime <= sunset:
            # Sun is above horizon - Full daylight
            return 1
        elif civil_twilight_begin <= current_datetime < sunrise or sunset < current_datetime <= civil_twilight_end:
            # Within civil twilight - Twilight or dawn
            return 2
        else:
            # Nighttime - determine based on urban/rural
            # Default to "public lighting lit" for urban areas
            return 5 if agg_ == 2 else 3
            
    except Exception as e:
        logger.error(f"Error determining lighting condition: {e}", exc_info=True)
        # Fallback based on time
        hour = current_datetime.hour if hasattr(current_datetime, 'hour') else 12
        return 1 if 6 <= hour < 20 else (5 if agg_ == 2 else 3)


def determine_atmospheric_condition(
    weather_data: Optional[dict],
    solar_data: Optional[dict],
    current_datetime: datetime,
    timezone: str
) -> int:
    """
    Determine atmospheric condition (atm) based on WMO weather codes and wind speed.
    
    Args:
        weather_data: Weather data dict from API or None
        solar_data: Solar data dict (for dazzling condition) or None
        current_datetime: Current datetime
        timezone: Timezone string
        
    Returns:
        atm value (-1, 1-9): -1=unknown, 1=normal, 2=light rain, 3=heavy rain, etc.
    """
    if not weather_data:
        return -1  # Unknown
    
    try:
        weathercode = weather_data.get("weathercode")
        windspeed = weather_data.get("windspeed_10m")
        
        # Check for strong wind first (overrides other conditions)
        if windspeed is not None and windspeed > STRONG_WIND_THRESHOLD:
            return 6  # Strong wind (storm)
        
        # Map WMO weather codes to atmospheric conditions
        # WMO codes: https://open-meteo.com/en/docs
        if weathercode is None:
            return -1
        
        # Fog/Smoke (visibility hazards)
        if 40 <= weathercode <= 49:
            return 5  # Fog or smoke
        
        # Frozen/Solid precipitation
        if weathercode in [71, 72, 73, 74, 75, 76, 77, 85, 86]:
            return 4  # Snow or hail
        
        # Rain (drizzle/light vs heavy)
        if weathercode in [51, 52, 53, 56, 57]:  # Drizzle
            return 2  # Light rain
        if weathercode in [61, 63, 65, 66, 67, 80, 81, 82]:  # Rain
            # Distinguish light vs heavy
            if weathercode in [61, 63]:  # Slight/Moderate rain
                return 2  # Light rain
            else:  # Heavy/Violent rain
                return 3  # Heavy rain
        
        # Cloudy weather
        if weathercode in [2, 3]:  # Partly cloudy, overcast
            return 8  # Cloudy weather
        
        # Clear/Normal conditions
        if weathercode in [0, 1]:  # Clear sky, mainly clear
            # Check for dazzling condition: Clear skies during twilight
            if solar_data:
                try:
                    sunrise_str = solar_data.get("sunrise")
                    sunset_str = solar_data.get("sunset")
                    civil_twilight_begin_str = solar_data.get("civil_twilight_begin")
                    civil_twilight_end_str = solar_data.get("civil_twilight_end")
                    
                    if all([sunrise_str, sunset_str, civil_twilight_begin_str, civil_twilight_end_str]):
                        sunrise = parse_iso_datetime(sunrise_str, timezone)
                        sunset = parse_iso_datetime(sunset_str, timezone)
                        civil_twilight_begin = parse_iso_datetime(civil_twilight_begin_str, timezone)
                        civil_twilight_end = parse_iso_datetime(civil_twilight_end_str, timezone)
                        
                        if all([sunrise, sunset, civil_twilight_begin, civil_twilight_end]):
                            if current_datetime.tzinfo is None:
                                current_datetime = current_datetime.replace(tzinfo=ZoneInfo(timezone))
                            else:
                                current_datetime = current_datetime.astimezone(ZoneInfo(timezone))
                            
                            # Check if in twilight period
                            in_twilight = (
                                (civil_twilight_begin <= current_datetime < sunrise) or
                                (sunset < current_datetime <= civil_twilight_end)
                            )
                            
                            if in_twilight:
                                return 7  # Dazzling weather
                except Exception as e:
                    logger.debug(f"Error checking dazzling condition: {e}")
            
            # Normal clear conditions
            return 1  # Normal
        
        # Other/Unknown
        return 9  # Other
        
    except Exception as e:
        logger.error(f"Error determining atmospheric condition: {e}", exc_info=True)
        return -1


def get_weather_conditions(
    lat: float,
    lon: float,
    current_datetime: datetime,
    agg_: int = 2
) -> dict:
    """
    Main function to fetch weather and solar data and determine conditions.
    
    Args:
        lat: Latitude
        lon: Longitude
        current_datetime: Current datetime (may be timezone-naive, will be converted)
        agg_: Urban area indicator (1 = outside urban, 2 = inside urban)
        
    Returns:
        Dict with:
            - lum: Lighting condition (1-5)
            - atm: Atmospheric condition (-1, 1-9)
            - solar_data: Solar data dict or None
            - weather_data: Weather data dict or None
            - timezone: Timezone used
            - error: Error message if any API failed
    """
    # Get timezone for coordinates
    timezone = get_timezone_from_coords(lat, lon)

    # Ensure datetime is timezone-aware (using the datetime provided from dashboard)
    # This datetime will be used consistently for both solar and weather data fetching
    if current_datetime.tzinfo is None:
        current_datetime = current_datetime.replace(tzinfo=ZoneInfo(timezone))
    else:
        current_datetime = current_datetime.astimezone(ZoneInfo(timezone))

    # Fetch solar data using the datetime from dashboard
    solar_data = fetch_solar_data(lat, lon, current_datetime.date())

    # Fetch weather data using the datetime from dashboard
    weather_data = fetch_weather_data(lat, lon, current_datetime, timezone)

    # Determine error message
    error = None
    if not weather_data:
        error = "Weather API unavailable, using solar data only"
    elif not solar_data:
        error = "Solar API unavailable, using weather data only"

    # Determine conditions
    lum = determine_lighting_condition(solar_data, current_datetime, agg_, timezone)
    atm = determine_atmospheric_condition(
        weather_data, solar_data, current_datetime, timezone
    )

    return {
        "lum": lum,
        "atm": atm,
        "solar_data": solar_data,
        "weather_data": weather_data,
        "timezone": timezone,
        "error": error,
    }

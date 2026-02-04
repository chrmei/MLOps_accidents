"""Control Center: prediction form, map, result."""
from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium
import folium
from datetime import datetime

from ..components.api_client import post, check_geocode_health, get_weather_conditions
from ..components.prediction_form import render_prediction_form, render_current_coordinates_display, KEY_PREFIX, LUM_OPTIONS, ATM_OPTIONS
from ..components.address_input import render_address_input


def render():
    st.title("Control Center — Accident Severity")
    st.markdown("Enter incident details and get severity prediction.")

    # Check if geocoding service is available
    geocoding_available = check_geocode_health()

    # Create two-column layout: Location/Coordinates (left) | Map (right)
    col_left, col_right = st.columns([1, 1])

    address_coords = None
    
    # Track coordinates before geocoding to detect changes
    prev_lat = st.session_state.get(f"{KEY_PREFIX}lat", 48.8584)
    prev_long = st.session_state.get(f"{KEY_PREFIX}long", 2.2945)
    geocoding_updated = False
    
    with col_left:
        # Location section
        st.markdown("### Location")
        if geocoding_available:
            address_coords = render_address_input()
        else:
            st.warning("⚠️ Geocoding service is currently unavailable. Click on the map to set coordinates.")
        
        # Current Coordinates Display (moved below address input, smaller)
        current_lat, current_long = render_current_coordinates_display(address_coords, geocoding_available)
        
        # Check if coordinates changed via geocoding
        if address_coords and address_coords.get("latitude") and address_coords.get("longitude"):
            geocoded_lat = float(address_coords["latitude"])
            geocoded_long = float(address_coords["longitude"])
            if abs(geocoded_lat - prev_lat) > 0.0001 or abs(geocoded_long - prev_long) > 0.0001:
                geocoding_updated = True
        
        # Weather conditions display (below GPS coordinates)
        weather_data = st.session_state.get(f"{KEY_PREFIX}weather_data")
        if weather_data:
            render_weather_display(weather_data)

    with col_right:
        # Map display (always shown, spans over two rows)
        map_col1, map_col2 = st.columns([20, 1])
        with map_col1:
            st.markdown("### Map")
        with map_col2:
            st.markdown("💡", help="Click on the map to set coordinates")
        
        # Get current coordinates from session state (updated by geocoding or map click)
        current_lat = st.session_state.get(f"{KEY_PREFIX}lat", 48.8584)
        current_long = st.session_state.get(f"{KEY_PREFIX}long", 2.2945)
        
        # Create map with current coordinates
        m = folium.Map(location=[current_lat, current_long], zoom_start=17)
        
        # Add draggable marker
        marker_popup_text = address_coords.get("address", "Click map to set location") if address_coords else "Click map to set location"
        marker = folium.Marker(
            [current_lat, current_long],
            tooltip="Incident Location (drag marker or click map to move)",
            popup=marker_popup_text,
            draggable=True,
        )
        marker.add_to(m)
        
        # Render map and capture interactions
        map_data = st_folium(
            m,
            width=None,
            height=500,
            key="address_map",
            returned_objects=["last_object_clicked", "last_clicked"],
        )

        # Show selected address and GPS coordinates
        if address_coords and address_coords.get("address"):
            st.caption(f"📍 **Selected:** {address_coords.get('address', 'Unknown address')}")
        
        # Display GPS coordinates
        current_lat = st.session_state.get(f"{KEY_PREFIX}lat", 48.8584)
        current_long = st.session_state.get(f"{KEY_PREFIX}long", 2.2945)
        st.caption(f"**GPS:** {current_lat:.6f}, {current_long:.6f}")

        # Attribution required by Nominatim Usage Policy (below map)
        st.caption(
            "📍 Geocoding data © OpenStreetMap contributors, licensed under ODbL. "
            "See https://www.openstreetmap.org/copyright"
        )
        
        # Handle map interactions: marker drag or map click
        coordinates_updated = False
        new_lat = None
        new_long = None
        
        # Check if marker was dragged/clicked
        if map_data.get("last_object_clicked"):
            clicked = map_data["last_object_clicked"]
            if clicked and isinstance(clicked, dict):
                if "lat" in clicked and "lng" in clicked:
                    new_lat = float(clicked["lat"])
                    new_long = float(clicked["lng"])
                    # Only update if coordinates actually changed
                    if abs(new_lat - current_lat) > 0.0001 or abs(new_long - current_long) > 0.0001:
                        st.session_state[f"{KEY_PREFIX}lat"] = new_lat
                        st.session_state[f"{KEY_PREFIX}long"] = new_long
                        coordinates_updated = True
        
        # Check if map was clicked (not marker)
        if not coordinates_updated and map_data.get("last_clicked"):
            clicked = map_data["last_clicked"]
            if clicked and isinstance(clicked, dict):
                if "lat" in clicked and "lng" in clicked:
                    new_lat = float(clicked["lat"])
                    new_long = float(clicked["lng"])
                    # Only update if coordinates actually changed
                    if abs(new_lat - current_lat) > 0.0001 or abs(new_long - current_long) > 0.0001:
                        st.session_state[f"{KEY_PREFIX}lat"] = new_lat
                        st.session_state[f"{KEY_PREFIX}long"] = new_long
                        coordinates_updated = True
        
        # Fetch weather data when coordinates are updated (via map or geocoding)
        coordinates_changed = coordinates_updated or geocoding_updated
        if coordinates_updated:
            target_lat = new_lat
            target_long = new_long
        elif geocoding_updated:
            target_lat = current_lat
            target_long = current_long
        else:
            target_lat = None
            target_long = None
        
        # Check if we've already fetched weather for these coordinates to avoid infinite loops
        last_weather_lat = st.session_state.get(f"{KEY_PREFIX}last_weather_lat")
        last_weather_long = st.session_state.get(f"{KEY_PREFIX}last_weather_long")
        
        if coordinates_changed and target_lat is not None and target_long is not None:
            # Only fetch if coordinates have actually changed from last fetch
            should_fetch = True
            if last_weather_lat is not None and last_weather_long is not None:
                if abs(target_lat - last_weather_lat) < 0.0001 and abs(target_long - last_weather_long) < 0.0001:
                    should_fetch = False
            
            if should_fetch:
                # Get current datetime from form or use current time
                date_input_key = f"{KEY_PREFIX}date_input"
                time_input_key = f"{KEY_PREFIX}time_input"
                selected_date = st.session_state.get(date_input_key, datetime.now().date())
                selected_time = st.session_state.get(time_input_key, datetime.now().time())
                current_datetime = datetime.combine(selected_date, selected_time)
                
                # Get agg_ value (default to urban area)
                agg_ = st.session_state.get(f"{KEY_PREFIX}agg_", 2)
                
                # Fetch weather conditions
                with st.spinner("Fetching weather conditions..."):
                    weather_result = get_weather_conditions(
                        latitude=target_lat,
                        longitude=target_long,
                        datetime_obj=current_datetime,
                        agg_=agg_
                    )
                    if weather_result:
                        st.session_state[f"{KEY_PREFIX}weather_data"] = weather_result
                        # Store lum and atm in session state for form auto-population
                        if weather_result.get("lum") is not None:
                            st.session_state[f"{KEY_PREFIX}lum"] = weather_result["lum"]
                        if weather_result.get("atm") is not None:
                            st.session_state[f"{KEY_PREFIX}atm"] = weather_result["atm"]
                        # Track the coordinates we fetched weather for
                        st.session_state[f"{KEY_PREFIX}last_weather_lat"] = target_lat
                        st.session_state[f"{KEY_PREFIX}last_weather_long"] = target_long
                
                st.rerun()
        
        # Check if we need to fetch weather on initial load
        weather_data = st.session_state.get(f"{KEY_PREFIX}weather_data")
        if not weather_data and not coordinates_changed:
            # Initial load: fetch weather for current coordinates
            current_lat = st.session_state.get(f"{KEY_PREFIX}lat", 48.8584)
            current_long = st.session_state.get(f"{KEY_PREFIX}long", 2.2945)
            
            # Check if we've already fetched weather for these coordinates
            last_weather_lat = st.session_state.get(f"{KEY_PREFIX}last_weather_lat")
            last_weather_long = st.session_state.get(f"{KEY_PREFIX}last_weather_long")
            should_fetch_initial = True
            if last_weather_lat is not None and last_weather_long is not None:
                if abs(current_lat - last_weather_lat) < 0.0001 and abs(current_long - last_weather_long) < 0.0001:
                    should_fetch_initial = False
            
            if should_fetch_initial:
                date_input_key = f"{KEY_PREFIX}date_input"
                time_input_key = f"{KEY_PREFIX}time_input"
                selected_date = st.session_state.get(date_input_key, datetime.now().date())
                selected_time = st.session_state.get(time_input_key, datetime.now().time())
                current_datetime = datetime.combine(selected_date, selected_time)
                agg_ = st.session_state.get(f"{KEY_PREFIX}agg_", 2)
                
                # Only fetch if we have valid coordinates
                if current_lat and current_long:
                    weather_result = get_weather_conditions(
                        latitude=current_lat,
                        longitude=current_long,
                        datetime_obj=current_datetime,
                        agg_=agg_
                    )
                    if weather_result:
                        st.session_state[f"{KEY_PREFIX}weather_data"] = weather_result
                        if weather_result.get("lum") is not None:
                            st.session_state[f"{KEY_PREFIX}lum"] = weather_result["lum"]
                        if weather_result.get("atm") is not None:
                            st.session_state[f"{KEY_PREFIX}atm"] = weather_result["atm"]
                        # Track the coordinates we fetched weather for
                        st.session_state[f"{KEY_PREFIX}last_weather_lat"] = current_lat
                        st.session_state[f"{KEY_PREFIX}last_weather_long"] = current_long

    st.markdown("---")

    # Prediction form with address coordinates (full width below)
    weather_data_for_form = st.session_state.get(f"{KEY_PREFIX}weather_data")
    features = render_prediction_form(
        address_coords=address_coords,
        geocoding_available=geocoding_available,
        weather_data=weather_data_for_form,
    )

    # Initialize session state for prediction results if not exists
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "prediction_features" not in st.session_state:
        st.session_state.prediction_features = None

    st.markdown("---")
    if st.button("Predict", type="primary"):
        resp = post("/api/v1/predict/", json={"features": features})
        if resp is None:
            st.error("Not authenticated.")
            st.session_state.prediction_result = None
            st.session_state.prediction_features = None
            return
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", "Prediction failed")
            except Exception:
                detail = "Prediction failed"
            st.error(detail)
            st.session_state.prediction_result = None
            st.session_state.prediction_features = None
            return
        data = resp.json()
        pred = data.get("prediction")
        prob = data.get("probability", 0.0)
        model_type = data.get("model_type", "")
        severity_label = "Severe (1)" if pred == 1 else "Not severe (0)"
        # Store results in session state
        st.session_state.prediction_result = {
            "prediction": pred,
            "probability": prob,
            "model_type": model_type,
            "severity_label": severity_label,
        }
        st.session_state.prediction_features = features.copy()

    # Display results from session state (persists across reruns)
    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        st.success(
            f"**Severity:** {result['severity_label']}  |  **Risk probability:** {result['probability']:.2%}  |  Model: {result['model_type']}"
        )
        if (
            st.session_state.prediction_features
            and "lat" in st.session_state.prediction_features
            and "long" in st.session_state.prediction_features
        ):
            # Street-level detail for prediction result map as well
            m = folium.Map(
                location=[
                    float(st.session_state.prediction_features["lat"]),
                    float(st.session_state.prediction_features["long"]),
                ],
                zoom_start=17,
            )
            folium.Marker(
                [
                    float(st.session_state.prediction_features["lat"]),
                    float(st.session_state.prediction_features["long"]),
                ],
                tooltip="Incident",
            ).add_to(m)
            st_folium(m, width=700, height=400)


def get_weathercode_description(weathercode: int | None) -> str:
    """
    Get human-readable description for WMO weather code.
    
    Args:
        weathercode: WMO weather code (0-99)
        
    Returns:
        Description string
    """
    if weathercode is None:
        return "Unknown"
    
    # WMO Weather Code mapping (Open-Meteo uses WMO codes)
    weathercode_map = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    
    # Check exact match first
    if weathercode in weathercode_map:
        return weathercode_map[weathercode]
    
    # Check ranges
    if 40 <= weathercode <= 49:
        return "Fog"
    if 50 <= weathercode <= 59:
        return "Drizzle"
    if 60 <= weathercode <= 69:
        return "Rain"
    if 70 <= weathercode <= 79:
        return "Snow"
    if 90 <= weathercode <= 99:
        return "Thunderstorm"
    
    return f"Code {weathercode}"


def render_weather_display(weather_data: dict) -> None:
    """
    Display weather conditions in a compact format.
    
    Args:
        weather_data: Dict with lum, atm, weather_data, solar_data, error, etc.
    """
    lum = weather_data.get("lum")
    atm = weather_data.get("atm")
    weather_info = weather_data.get("weather_data")
    error = weather_data.get("error")
    
    if error:
        st.warning(f"⚠️ {error}")
    
    # Display lighting condition
    lum_label = LUM_OPTIONS.get(lum, "Unknown") if lum else "Unknown"
    lum_emoji = "☀️" if lum == 1 else "🌅" if lum == 2 else "🌙"
    st.metric("Lighting", f"{lum_emoji} {lum_label}", delta=None)
    
    # Display atmospheric condition
    atm_label = ATM_OPTIONS.get(atm, "Unknown") if atm else "Unknown"
    atm_emoji = "🌤️" if atm == 1 else "🌧️" if atm in [2, 3] else "❄️" if atm == 4 else "🌫️" if atm == 5 else "💨" if atm == 6 else "✨" if atm == 7 else "☁️" if atm == 8 else "❓"
    st.metric("Atmospheric", f"{atm_emoji} {atm_label}", delta=None)
    
    # Display detailed weather info if available (condensed format)
    if weather_info:
        weathercode = weather_info.get("weathercode")
        temp = weather_info.get("temperature_2m")
        wind = weather_info.get("windspeed_10m")
        cloud = weather_info.get("cloudcover")
        
        # Create condensed display with icons
        info_items = []
        
        # Weathercode with description
        if weathercode is not None:
            weathercode_desc = get_weathercode_description(weathercode)
            weathercode_emoji = "☀️" if weathercode in [0, 1] else "⛅" if weathercode in [2, 3] else "🌧️" if weathercode in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82] else "❄️" if weathercode in [71, 73, 75, 77, 85, 86] else "🌫️" if 40 <= weathercode <= 49 else "⛈️" if weathercode in [95, 96, 99] else "🌤️"
            info_items.append((weathercode_emoji, f"Weather: {weathercode_desc}"))
        
        # Temperature
        if temp is not None:
            info_items.append(("🌡️", f"Temp: {temp:.1f}°C"))
        
        # Wind speed
        if wind is not None:
            info_items.append(("💨", f"Wind: {wind:.1f} km/h"))
        
        # Cloud cover
        if cloud is not None:
            info_items.append(("☁️", f"Clouds: {cloud:.0f}%"))
        
        # Display in a compact format (2 columns)
        if info_items:
            cols = st.columns(2)
            for idx, (emoji, text) in enumerate(info_items):
                with cols[idx % 2]:
                    st.caption(f"{emoji} {text}")

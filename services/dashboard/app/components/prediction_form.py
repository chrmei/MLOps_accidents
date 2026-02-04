"""Prediction form: inputs for accident features, returns dict for API."""
from __future__ import annotations

import logging
import streamlit as st
from streamlit_folium import st_folium
import folium
from datetime import datetime, date, time

logger = logging.getLogger(__name__)

# Defaults aligned with src/models/test_features.json
DEFAULTS = {
    "place": 10,
    "catu": 3,
    "sexe": 1,
    "secu1": 0.0,
    "year_acc": 2021,
    "an_nais": 1961,
    "catv": 2,
    "obsm": 1,
    "motor": 1,
    "catr": 3,
    "circ": 2,
    "surf": 1,
    "situ": 1,
    "vma": 50,
    "lum": 5,
    "dep": 77,
    "com": 77317,
    "agg_": 2,
    "int": 1,
    "atm": 0,
    "col": 6,
    "lat": 48.8584,
    "long": 2.2945,
    "nb_victim": 2,
    "nb_vehicules": 1,
    "locp": 0,
    "actp": 0,
    "etatp": 0,
    "obs": 0,
    "v1": 0,
    "vosp": 0,
    "prof": 0,
    "plan": 0,
    "larrout": 0.0,
    "infra": 0,
}

KEY_PREFIX = "pred_"

# Field value mappings for user-friendly display
LUM_OPTIONS = {
    1: "Full daylight",
    2: "Twilight or dawn",
    3: "Night without public lighting",
    4: "Night with public lighting not lit",
    5: "Night with public lighting lit"
}

ATM_OPTIONS = {
    -1: "Unknown",
    1: "Normal",
    2: "Light rain",
    3: "Heavy rain",
    4: "Snow or hail",
    5: "Fog or smoke",
    6: "Strong wind (storm)",
    7: "Dazzling weather",
    8: "Cloudy weather",
    9: "Other"
}

COL_OPTIONS = {
    -1: "Unknown",
    1: "Two vehicles - front to front",
    2: "Two vehicles - from behind",
    3: "Two vehicles - side by side",
    4: "Three or more vehicles - chain collision",
    5: "Three or more vehicles - multiple collisions",
    6: "Other collision",
    7: "No collision"
}

AGG_OPTIONS = {
    1: "Outside urban area",
    2: "Inside urban area"
}

INT_OPTIONS = {
    1: "Outside intersection",
    2: "X intersection",
    3: "T intersection",
    4: "Y intersection",
    5: "Multiple intersection",
    6: "Roundabout",
    7: "Square",
    8: "Level crossing",
    9: "Other intersection"
}

SURF_OPTIONS = {
    -1: "Unknown",
    1: "Normal",
    2: "Wet",
    3: "Puddles",
    4: "Flooded",
    5: "Snow",
    6: "Mud",
    7: "Icy",
    8: "Fat or oil",
    9: "Other"
}

SITU_OPTIONS = {
    -1: "Unknown",
    0: "Not specified",
    1: "On road",
    2: "On emergency stop band",
    3: "On shoulder",
    4: "On sidewalk",
    5: "On cycle path",
    6: "On other special lane",
    7: "On parking lot",
    8: "Other"
}

CIRC_OPTIONS = {
    1: "One way",
    2: "Bidirectional",
    3: "Separated carriageways",
    4: "With variable assignment"
}

CATR_OPTIONS = {
    1: "Highway",
    2: "National road",
    3: "Departmental road",
    4: "Communal road",
    5: "Off public network",
    6: "Parking lot",
    7: "Urban boulevard",
    8: "Other",
    9: "Unknown"
}

CATU_OPTIONS = {
    1: "Driver",
    2: "Passenger",
    3: "Pedestrian",
    4: "Pedestrian with roller skates",
    5: "Other"
}

VOSP_OPTIONS = {
    -1: "Unknown",
    0: "No special lane",
    1: "Cycle path",
    2: "Cycle lane",
    3: "Reserved lane"
}

PROF_OPTIONS = {
    -1: "Unknown",
    1: "Flat",
    2: "Slope up",
    3: "Slope down",
    4: "Summit of slope"
}

PLAN_OPTIONS = {
    -1: "Unknown",
    1: "Straight part",
    2: "Curved to the left",
    3: "Curved to the right",
    4: "S-curve"
}

# Export for use in other modules
__all__ = ["render_prediction_form", "render_current_coordinates_display", "KEY_PREFIX", "DEFAULTS", "LUM_OPTIONS", "ATM_OPTIONS"]


def render_current_coordinates_display(
    address_coords: dict | None = None,
    geocoding_available: bool = True,
) -> tuple[float, float]:
    """
    Display current coordinates (read-only) and initialize session state.
    
    Coordinates can be set via:
    - Address geocoding (if available)
    - Map click interaction
    
    Args:
        address_coords: Optional dict with latitude and longitude from address geocoding
        geocoding_available: Whether geocoding service is available
        
    Returns:
        Tuple of (latitude, longitude) values
    """
    # Determine default lat/long values
    default_lat = float(address_coords.get("latitude", DEFAULTS["lat"])) if address_coords else float(DEFAULTS["lat"])
    default_long = float(address_coords.get("longitude", DEFAULTS["long"])) if address_coords else float(DEFAULTS["long"])

    # Initialize session state if not set
    if f"{KEY_PREFIX}lat" not in st.session_state:
        st.session_state[f"{KEY_PREFIX}lat"] = default_lat
    if f"{KEY_PREFIX}long" not in st.session_state:
        st.session_state[f"{KEY_PREFIX}long"] = default_long

    # Update session state with geocoded coordinates if available
    if address_coords and address_coords.get("latitude") and address_coords.get("longitude"):
        geocoded_lat = float(address_coords["latitude"])
        geocoded_long = float(address_coords["longitude"])
        st.session_state[f"{KEY_PREFIX}lat"] = geocoded_lat
        st.session_state[f"{KEY_PREFIX}long"] = geocoded_long

    # Get current coordinates (coordinates are displayed below the map, not here)
    current_lat = st.session_state.get(f"{KEY_PREFIX}lat", default_lat)
    current_long = st.session_state.get(f"{KEY_PREFIX}long", default_long)
    
    return current_lat, current_long


def render_prediction_form(
    address_coords: dict | None = None,
    geocoding_available: bool = True,
    weather_data: dict | None = None,
) -> dict:
    """
    Render inputs and return current feature dict for POST /api/v1/predict/.
    Call when user clicks Predict; dict is built from current widget values.
    
    Args:
        address_coords: Optional dict with latitude and longitude from address geocoding
        geocoding_available: Whether geocoding service is available (shows/hides fallback)
        weather_data: Optional dict with weather conditions (lum, atm) to auto-populate fields
        
    Returns:
        Dict of feature values for prediction API
    """

    # Get lat/long from session state (updated by manual coordinates fallback or geocoding)
    default_lat = float(address_coords.get("latitude", DEFAULTS["lat"])) if address_coords else float(DEFAULTS["lat"])
    default_long = float(address_coords.get("longitude", DEFAULTS["long"])) if address_coords else float(DEFAULTS["long"])
    lat = float(st.session_state.get(f"{KEY_PREFIX}lat", default_lat))
    long = float(st.session_state.get(f"{KEY_PREFIX}long", default_long))

    # Get current local date and time for auto-setting
    now = datetime.now()
    current_date = now.date()
    current_time = now.time()

    # Initialize datetime fields with current local time if not already set
    date_input_key = f"{KEY_PREFIX}date_input"
    time_input_key = f"{KEY_PREFIX}time_input"
    if date_input_key not in st.session_state:
        st.session_state[date_input_key] = current_date
    if time_input_key not in st.session_state:
        st.session_state[time_input_key] = current_time

    # Helper function to convert INSEE codes to integer format (handles Corsica 2A/2B)
    def convert_department_code(code: str | None) -> int:
        """Convert department code string to integer, handling Corsica codes."""
        if not code:
            return DEFAULTS["dep"]
        # Handle Corsica codes: 2A -> 201, 2B -> 202
        if code == "2A":
            return 201
        if code == "2B":
            return 202
        try:
            return int(code)
        except (ValueError, TypeError):
            return DEFAULTS["dep"]
    
    def convert_commune_code(code: str | None) -> int:
        """Convert commune code string to integer, handling Corsica codes."""
        if not code:
            return DEFAULTS["com"]
        # Handle Corsica codes in commune codes (e.g., "2A123" -> 201123, "2B456" -> 202456)
        if code.startswith("2A"):
            return int("201" + code[2:])
        if code.startswith("2B"):
            return int("202" + code[2:])
        try:
            return int(code)
        except (ValueError, TypeError):
            return DEFAULTS["com"]

    # Auto-populate department and commune codes from geocoding if available
    dep_key = f"{KEY_PREFIX}dep"
    com_key = f"{KEY_PREFIX}com"
    
    # Track last received INSEE codes to detect when new codes arrive
    last_insee_codes_key = f"{KEY_PREFIX}last_insee_codes"
    
    # Check if we have INSEE codes from geocoding
    if address_coords:
        dept_code = address_coords.get("department_code")
        commune_code = address_coords.get("commune_code")
        geocoded_lat = address_coords.get("latitude")
        geocoded_lon = address_coords.get("longitude")
        
        # Get last stored INSEE codes
        last_insee_codes = st.session_state.get(last_insee_codes_key, {})
        last_dept = last_insee_codes.get("department_code")
        last_commune = last_insee_codes.get("commune_code")
        last_lat = last_insee_codes.get("lat")
        last_lon = last_insee_codes.get("lon")
        
        # Check if coordinates changed significantly (new address/location)
        # Use a smaller threshold to catch more address changes
        coordinates_changed = False
        if geocoded_lat and geocoded_lon:
            if last_lat is None or last_lon is None:
                coordinates_changed = True
            else:
                # Check if coordinates changed by more than ~50 meters (roughly 0.0005 degrees)
                lat_diff = abs(float(geocoded_lat) - float(last_lat))
                lon_diff = abs(float(geocoded_lon) - float(last_lon))
                coordinates_changed = lat_diff > 0.0005 or lon_diff > 0.0005
        
        # Check if INSEE codes are new or different from what we last stored
        codes_changed = (
            (dept_code and dept_code != last_dept) or
            (commune_code and commune_code != last_commune)
        )
        
        # Update INSEE codes if they are available and either:
        # 1. Coordinates changed significantly (new location) - always update
        # 2. INSEE codes changed - always update
        # 3. INSEE codes are new (not previously stored) - always update
        # 4. We have INSEE codes but coordinates changed (even slightly) - update to ensure sync
        should_update = (
            coordinates_changed or
            codes_changed or
            (dept_code and not last_dept) or
            (commune_code and not last_commune)
        )
        
        if should_update:
            # Always update if we have new INSEE codes, regardless of current form values
            if dept_code:
                converted_dep = convert_department_code(dept_code)
                st.session_state[dep_key] = converted_dep
                logger.info(f"Auto-populated Department Code: {converted_dep} (from INSEE: {dept_code})")
            if commune_code:
                converted_com = convert_commune_code(commune_code)
                st.session_state[com_key] = converted_com
                logger.info(f"Auto-populated Commune Code: {converted_com} (from INSEE: {commune_code})")
            
            # Store current INSEE codes and coordinates to detect future changes
            if geocoded_lat and geocoded_lon:
                st.session_state[last_insee_codes_key] = {
                    "department_code": dept_code,
                    "commune_code": commune_code,
                    "lat": float(geocoded_lat),
                    "lon": float(geocoded_lon),
                }

    # ========== LOCATION & DATE/TIME ==========
    c1, c2 = st.columns(2)
    with c1:
        loc_col1, loc_col2 = st.columns([20, 1])
        with loc_col1:
            st.markdown("### 📍 Location & Date-Time")
        with loc_col2:
            st.markdown("ℹ️", help="Fields are automatically populated after searching for an address or selecting a location on the map")
        dep = st.number_input(
            "Department Code",
            value=st.session_state.get(dep_key, DEFAULTS["dep"]),
            min_value=1,
            max_value=99,
            key=dep_key,
            help="French department number (1-99, 201/202 for Corsica). Auto-filled from GPS coordinates for French locations."
        )
        com = st.number_input(
            "Commune Code",
            value=st.session_state.get(com_key, DEFAULTS["com"]),
            min_value=1000,
            key=com_key,
            help="Municipality code (5 digits). Auto-filled from GPS coordinates for French locations."
        )
        # Datetime selector in one cell
        dt_col1, dt_col2 = st.columns(2)
        with dt_col1:
            selected_date = st.date_input(
                "Date",
                min_value=date(2015, 1, 1),
                max_value=date(2030, 12, 31),
                key=date_input_key
            )
        with dt_col2:
            selected_time = st.time_input(
                "Time",
                key=time_input_key
            )
        
        # Extract day, month, year, and time from selected datetime
        jour = selected_date.day
        mois = selected_date.month
        an = selected_date.year
        hrmn = selected_time.strftime("%H:%M")
    
    with c2:
        # ========== ENVIRONMENTAL CONDITIONS ==========
        env_col1, env_col2 = st.columns([20, 1])
        with env_col1:
            st.markdown("### 🌤️ Environmental Conditions")
        with env_col2:
            st.markdown("ℹ️", help="Fields are automatically populated after searching for an address or selecting a location on the map")
        
        # Get options lists
        lum_options_list = list(LUM_OPTIONS.keys())
        atm_options_list = list(ATM_OPTIONS.keys())
        
        # Determine default lum value (prioritize session state, then weather_data, then default)
        # Session state is the source of truth after weather data is fetched
        default_lum = DEFAULTS["lum"]
        if f"{KEY_PREFIX}lum" in st.session_state:
            session_lum = st.session_state[f"{KEY_PREFIX}lum"]
            # Validate that session state value is a valid option
            if session_lum in lum_options_list:
                default_lum = session_lum
            else:
                # Invalid value, reset to default
                st.session_state[f"{KEY_PREFIX}lum"] = default_lum
        elif weather_data and weather_data.get("lum") is not None:
            weather_lum = weather_data["lum"]
            # Validate that weather data value is a valid option
            if weather_lum in lum_options_list:
                default_lum = weather_lum
                # Update session state for consistency
                st.session_state[f"{KEY_PREFIX}lum"] = default_lum
        
        # Determine default atm value (prioritize session state, then weather_data, then default)
        default_atm = DEFAULTS["atm"]
        if f"{KEY_PREFIX}atm" in st.session_state:
            session_atm = st.session_state[f"{KEY_PREFIX}atm"]
            # Validate that session state value is a valid option
            if session_atm in atm_options_list:
                default_atm = session_atm
            else:
                # Invalid value, reset to default
                st.session_state[f"{KEY_PREFIX}atm"] = default_atm
        elif weather_data and weather_data.get("atm") is not None:
            weather_atm = weather_data["atm"]
            # Validate that weather data value is a valid option
            if weather_atm in atm_options_list:
                default_atm = weather_atm
                # Update session state for consistency
                st.session_state[f"{KEY_PREFIX}atm"] = default_atm
        
        # Find index for selectbox (only used if session state doesn't have the value)
        lum_index = lum_options_list.index(default_lum) if default_lum in lum_options_list else 0
        atm_index = atm_options_list.index(default_atm) if default_atm in atm_options_list else 0
        
        # Ensure session state is set with valid values (needed for selectbox with key)
        if f"{KEY_PREFIX}lum" not in st.session_state or st.session_state[f"{KEY_PREFIX}lum"] not in lum_options_list:
            st.session_state[f"{KEY_PREFIX}lum"] = default_lum
        if f"{KEY_PREFIX}atm" not in st.session_state or st.session_state[f"{KEY_PREFIX}atm"] not in atm_options_list:
            st.session_state[f"{KEY_PREFIX}atm"] = default_atm
        
        lum = st.selectbox(
            "Lighting Conditions",
            options=lum_options_list,
            format_func=lambda x: LUM_OPTIONS[x],
            index=lum_index,
            key=f"{KEY_PREFIX}lum"
        )
        atm = st.selectbox(
            "Atmospheric Conditions",
            options=atm_options_list,
            format_func=lambda x: ATM_OPTIONS[x],
            index=atm_index,
            key=f"{KEY_PREFIX}atm"
        )
    
    # ========== VICTIM INFORMATION & VEHICLE INFORMATION ==========
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("### 👤 Victim Information")
        catu = st.selectbox(
            "User Category",
            options=list(CATU_OPTIONS.keys()),
            format_func=lambda x: CATU_OPTIONS[x],
            index=list(CATU_OPTIONS.keys()).index(DEFAULTS["catu"]) if DEFAULTS["catu"] in CATU_OPTIONS else 0,
            key=f"{KEY_PREFIX}catu",
            help="Category of the user involved in the accident"
        )
        sexe = st.selectbox(
            "Gender",
            options=[1, 2],
            format_func=lambda x: "Male" if x == 1 else "Female",
            index=0 if DEFAULTS["sexe"] == 1 else 1,
            key=f"{KEY_PREFIX}sexe"
        )
        an_nais = st.number_input(
            "Birth Year",
            value=DEFAULTS["an_nais"],
            min_value=1920,
            max_value=2010,
            key=f"{KEY_PREFIX}an_nais",
            help="Year of birth of the victim"
        )
        place = st.number_input(
            "Position in Vehicle",
            value=DEFAULTS["place"],
            min_value=1,
            max_value=20,
            key=f"{KEY_PREFIX}place",
            help="Position of the user in the vehicle (1-20)"
        )
        secu1 = st.number_input(
            "Safety Equipment",
            value=int(DEFAULTS["secu1"]),
            min_value=-1,
            max_value=9,
            key=f"{KEY_PREFIX}secu1",
            help="Type of safety equipment used (-1: Unknown, 0: None, 1-9: Various equipment)"
        )
        locp = st.number_input(
            "Pedestrian Location",
            value=DEFAULTS["locp"],
            min_value=-1,
            max_value=9,
            key=f"{KEY_PREFIX}locp",
            help="Location of pedestrian if applicable (-1: Unknown, 0: Not pedestrian, 1-9: Various locations)"
        )
        actp_input = st.text_input(
            "Pedestrian Action",
            value=str(DEFAULTS["actp"]),
            key=f"{KEY_PREFIX}actp",
            help="Action of pedestrian (-1, 0, or A/B)"
        )
        etatp = st.number_input(
            "Pedestrian State",
            value=DEFAULTS["etatp"],
            min_value=-1,
            max_value=3,
            key=f"{KEY_PREFIX}etatp",
            help="State of pedestrian (-1: Unknown, 1-3: Various states)"
        )
    
    with v2:
        st.markdown("### 🚗 Vehicle Information")
        catv = st.number_input(
            "Vehicle Category",
            value=DEFAULTS["catv"],
            min_value=0,
            max_value=20,
            key=f"{KEY_PREFIX}catv",
            help="Category of vehicle (0: Unknown, 1-20: Various vehicle types)"
        )
        motor = st.number_input(
            "Motor Type",
            value=DEFAULTS["motor"],
            min_value=-1,
            max_value=6,
            key=f"{KEY_PREFIX}motor",
            help="Type of motor (-1: Unknown, 0: None, 1-6: Various motor types)"
        )
        obsm = st.number_input(
            "Obstacle Marker",
            value=DEFAULTS["obsm"],
            min_value=-1,
            max_value=9,
            key=f"{KEY_PREFIX}obsm",
            help="Obstacle marker type (-1: Unknown, 0: None, 1-9: Various markers)"
        )
        obs = st.number_input(
            "Obstacle",
            value=DEFAULTS["obs"],
            min_value=-1,
            max_value=17,
            key=f"{KEY_PREFIX}obs",
            help="Type of obstacle (-1: Unknown, 0: None, 1-17: Various obstacles)"
        )
        nb_victim = st.number_input(
            "Number of Victims",
            value=DEFAULTS["nb_victim"],
            min_value=0,
            key=f"{KEY_PREFIX}nb_victim"
        )
        nb_vehicules = st.number_input(
            "Number of Vehicles",
            value=DEFAULTS["nb_vehicules"],
            min_value=0,
            key=f"{KEY_PREFIX}nb_vehicules"
        )
        situ = st.selectbox(
            "Situation",
            options=list(SITU_OPTIONS.keys()),
            format_func=lambda x: SITU_OPTIONS[x],
            index=list(SITU_OPTIONS.keys()).index(DEFAULTS["situ"]) if DEFAULTS["situ"] in SITU_OPTIONS else 0,
            key=f"{KEY_PREFIX}situ"
        )
    
    # ========== ROAD/LOCATION CHARACTERISTICS & ROAD SURFACE & INFRASTRUCTURE ==========
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("### 🛣️ Road & Location Characteristics")
        catr = st.selectbox(
            "Road Category",
            options=list(CATR_OPTIONS.keys()),
            format_func=lambda x: CATR_OPTIONS[x],
            index=list(CATR_OPTIONS.keys()).index(DEFAULTS["catr"]) if DEFAULTS["catr"] in CATR_OPTIONS else 0,
            key=f"{KEY_PREFIX}catr"
        )
        circ = st.selectbox(
            "Traffic Direction",
            options=list(CIRC_OPTIONS.keys()),
            format_func=lambda x: CIRC_OPTIONS[x],
            index=list(CIRC_OPTIONS.keys()).index(DEFAULTS["circ"]) if DEFAULTS["circ"] in CIRC_OPTIONS else 0,
            key=f"{KEY_PREFIX}circ"
        )
        vma = st.number_input(
            "Speed Limit (km/h)",
            value=DEFAULTS["vma"],
            min_value=0,
            max_value=150,
            key=f"{KEY_PREFIX}vma",
            help="Maximum authorized speed in km/h"
        )
        vosp = st.selectbox(
            "Special Lane",
            options=list(VOSP_OPTIONS.keys()),
            format_func=lambda x: VOSP_OPTIONS[x],
            index=list(VOSP_OPTIONS.keys()).index(DEFAULTS["vosp"]) if DEFAULTS["vosp"] in VOSP_OPTIONS else 0,
            key=f"{KEY_PREFIX}vosp"
        )
        v1 = st.number_input(
            "Traffic Lane",
            value=DEFAULTS["v1"],
            min_value=-1,
            max_value=9,
            key=f"{KEY_PREFIX}v1",
            help="Number of traffic lanes (-1: Unknown, 0: Not specified, 1-9: Number of lanes)"
        )
        prof = st.selectbox(
            "Road Profile",
            options=list(PROF_OPTIONS.keys()),
            format_func=lambda x: PROF_OPTIONS[x],
            index=list(PROF_OPTIONS.keys()).index(DEFAULTS["prof"]) if DEFAULTS["prof"] in PROF_OPTIONS else 0,
            key=f"{KEY_PREFIX}prof"
        )
        plan = st.selectbox(
            "Road Plan",
            options=list(PLAN_OPTIONS.keys()),
            format_func=lambda x: PLAN_OPTIONS[x],
            index=list(PLAN_OPTIONS.keys()).index(DEFAULTS["plan"]) if DEFAULTS["plan"] in PLAN_OPTIONS else 0,
            key=f"{KEY_PREFIX}plan"
        )
        larrout = st.number_input(
            "Road Width (meters)",
            value=float(DEFAULTS["larrout"]),
            min_value=-1.0,
            max_value=50.0,
            step=0.1,
            key=f"{KEY_PREFIX}larrout",
            help="Width of the road in meters (-1: Unknown)"
        )
    
    with r2:
        st.markdown("### 🏗️ Road Surface & Infrastructure")
        agg_ = st.selectbox(
            "Urban Area",
            options=list(AGG_OPTIONS.keys()),
            format_func=lambda x: AGG_OPTIONS[x],
            index=list(AGG_OPTIONS.keys()).index(DEFAULTS["agg_"]) if DEFAULTS["agg_"] in AGG_OPTIONS else 0,
            key=f"{KEY_PREFIX}agg_"
        )
        int_ = st.selectbox(
            "Intersection Type",
            options=list(INT_OPTIONS.keys()),
            format_func=lambda x: INT_OPTIONS[x],
            index=list(INT_OPTIONS.keys()).index(DEFAULTS["int"]) if DEFAULTS["int"] in INT_OPTIONS else 0,
            key=f"{KEY_PREFIX}int"
        )
        surf = st.selectbox(
            "Road Surface Condition",
            options=list(SURF_OPTIONS.keys()),
            format_func=lambda x: SURF_OPTIONS[x],
            index=list(SURF_OPTIONS.keys()).index(DEFAULTS["surf"]) if DEFAULTS["surf"] in SURF_OPTIONS else 0,
            key=f"{KEY_PREFIX}surf"
        )
        infra = st.number_input(
            "Infrastructure",
            value=DEFAULTS["infra"],
            min_value=-1,
            max_value=9,
            key=f"{KEY_PREFIX}infra",
            help="Type of infrastructure (-1: Unknown, 0: None, 1-9: Various infrastructures)"
        )
        col = st.selectbox(
            "Collision Type",
            options=list(COL_OPTIONS.keys()),
            format_func=lambda x: COL_OPTIONS[x],
            index=list(COL_OPTIONS.keys()).index(DEFAULTS["col"]) if DEFAULTS["col"] in COL_OPTIONS else 0,
            key=f"{KEY_PREFIX}col"
        )
    
    # Convert actp to appropriate type
    actp_value = DEFAULTS["actp"]
    try:
        if actp_input.isdigit():
            actp_value = int(actp_input)
        elif actp_input.upper() in ['A', 'B']:
            actp_value = actp_input.upper()
        elif actp_input == '-1':
            actp_value = -1
    except:
        actp_value = DEFAULTS["actp"]
    
    return {
        "place": place,
        "catu": catu,
        "sexe": sexe,
        "secu1": float(secu1),
        "year_acc": an,
        "an_nais": an_nais,
        "catv": catv,
        "obsm": obsm,
        "motor": motor,
        "catr": catr,
        "circ": circ,
        "surf": surf,
        "situ": situ,
        "vma": vma,
        "jour": jour,
        "mois": mois,
        "an": an,
        "hrmn": hrmn,
        "lum": lum,
        "dep": dep,
        "com": com,
        "agg_": agg_,
        "int": int_,
        "atm": atm,
        "col": col,
        "lat": lat,
        "long": long,
        "nb_victim": nb_victim,
        "nb_vehicules": nb_vehicules,
        "locp": locp,
        "actp": actp_value,
        "etatp": etatp,
        "obs": obs,
        "v1": v1,
        "vosp": vosp,
        "prof": prof,
        "plan": plan,
        "larrout": float(larrout),
        "infra": infra,
    }

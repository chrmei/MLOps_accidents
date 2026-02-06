"""
French Geo API client for reverse geocoding to get INSEE codes.

This module provides reverse geocoding functionality using geo.api.gouv.fr
to retrieve INSEE commune codes and department codes from GPS coordinates.
"""
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class FrenchGeoResult:
    """Result from French Geo API reverse geocoding."""

    commune_code: Optional[str] = None  # INSEE commune code (5 digits)
    department_code: Optional[str] = None  # Department code (2-3 digits)
    commune_name: Optional[str] = None
    region_code: Optional[str] = None
    postal_codes: Optional[list[str]] = None


class FrenchGeoAPIClient:
    """
    Client for French Geo API (geo.api.gouv.fr).
    
    Provides reverse geocoding to get INSEE codes from GPS coordinates.
    Features:
    - Rate limiting: 1 request/second (recommended)
    - Caching: Results cached to reduce API calls
    - Automatic handling of French administrative boundaries
    """

    BASE_URL = "https://geo.api.gouv.fr"
    RATE_LIMIT_PER_SECOND = 1.0

    def __init__(self):
        """Initialize French Geo API client."""
        self.last_request_time = 0.0
        self.session = requests.Session()
        # Cache for reverse geocoding results (coordinate -> INSEE codes)
        # Key format: f"{lat:.6f},{lon:.6f}"
        self._reverse_cache: Dict[str, Optional[FrenchGeoResult]] = {}

    def _rate_limit_wait(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        time_since_last = now - self.last_request_time
        min_interval = 1.0 / self.RATE_LIMIT_PER_SECOND

        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def _is_in_france(self, lat: float, lon: float) -> bool:
        """
        Check if coordinates are within France (approximate bounds).
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            True if coordinates are likely in France
        """
        # Approximate bounds for France (metropolitan + overseas territories)
        # Metropolitan France: ~41.3°N to 51.1°N, ~-5.1°W to 9.6°E
        # Including overseas territories expands these bounds significantly
        # Using broader bounds to be safe
        return (
            -180 <= lon <= 180 and  # Longitude can wrap
            -21 <= lat <= 52  # Latitude range covering all French territories
        )

    def reverse_geocode(
        self, latitude: float, longitude: float
    ) -> Optional[FrenchGeoResult]:
        """
        Reverse geocode coordinates to get INSEE codes.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            FrenchGeoResult with INSEE codes if found, None otherwise
        """
        # Check if coordinates are in France (approximate check)
        if not self._is_in_france(latitude, longitude):
            logger.debug(
                f"Coordinates ({latitude}, {longitude}) are likely not in France, "
                "skipping French Geo API call"
            )
            return None

        # Create cache key (rounded to 6 decimal places for reasonable precision)
        cache_key = f"{latitude:.6f},{longitude:.6f}"

        # Check cache first
        if cache_key in self._reverse_cache:
            logger.debug(f"Returning cached French Geo API result for: {cache_key}")
            cached_result = self._reverse_cache[cache_key]
            if cached_result is None:
                return None
            # Return a copy to avoid modifying cached data
            return FrenchGeoResult(
                commune_code=cached_result.commune_code,
                department_code=cached_result.department_code,
                commune_name=cached_result.commune_name,
                region_code=cached_result.region_code,
                postal_codes=(
                    cached_result.postal_codes.copy()
                    if cached_result.postal_codes
                    else None
                ),
            )

        self._rate_limit_wait()

        try:
            url = f"{self.BASE_URL}/communes"
            params = {
                "lat": latitude,
                "lon": longitude,
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # API returns a list, but should contain at most one result
            if not data or len(data) == 0:
                logger.debug(
                    f"No commune found for coordinates ({latitude}, {longitude})"
                )
                self._reverse_cache[cache_key] = None
                return None

            # Get first result (should be the only one)
            result = data[0]

            french_geo_result = FrenchGeoResult(
                commune_code=result.get("code"),  # INSEE commune code
                department_code=result.get("codeDepartement"),  # Department code
                commune_name=result.get("nom"),
                region_code=result.get("codeRegion"),
                postal_codes=result.get("codesPostaux", []),
            )

            # Cache the result
            self._reverse_cache[cache_key] = french_geo_result
            logger.debug(
                f"French Geo API result: commune={french_geo_result.commune_code}, "
                f"department={french_geo_result.department_code}"
            )
            return french_geo_result

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"French Geo API request failed for ({latitude}, {longitude}): {e}"
            )
            # Don't cache failures - allow retry on next request
            return None
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(
                f"Invalid French Geo API response format for ({latitude}, {longitude}): {e}"
            )
            # Cache None to avoid repeated failed requests
            self._reverse_cache[cache_key] = None
            return None


# Singleton instance
_french_geo_client: Optional[FrenchGeoAPIClient] = None


def get_french_geo_client() -> FrenchGeoAPIClient:
    """
    Get singleton French Geo API client instance.
    
    Returns:
        FrenchGeoAPIClient instance
    """
    global _french_geo_client
    if _french_geo_client is None:
        _french_geo_client = FrenchGeoAPIClient()
    return _french_geo_client

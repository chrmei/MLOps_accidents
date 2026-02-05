"""Dashboard configuration."""
import os

# Internal API base URL (for server-side API calls - uses Docker service name)
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:80").rstrip("/")

# Browser-accessible base URL (for links opened in user's browser - uses localhost or external hostname)
# Defaults to localhost, but can be overridden via BROWSER_BASE_URL env var
BROWSER_BASE_URL = os.environ.get("BROWSER_BASE_URL", "http://localhost").rstrip("/")

SESSION_EXPIRE_MINUTES = int(os.environ.get("SESSION_EXPIRE_MINUTES", "30"))
API_TIMEOUT = 30.0

# API Endpoint Testing

The project includes a comprehensive test suite for all microservices API endpoints. Tests cover successful requests, error handling, authentication, and authorization.

## Quick Start

**Run all tests in Docker (recommended):**

```bash
# Start services and run all API tests
make docker-test

# Or using Docker Compose directly
docker compose --profile test run --rm test
```

**Run tests locally (requires services running):**

```bash
# Set BASE_URL if services are not on localhost
export BASE_URL=http://localhost

# Run all API tests
make test-api

# Run with coverage
make test-api-cov

# Run tests for specific service
make test-api-auth      # Auth service only
make test-api-data      # Data service only
make test-api-train     # Train service only
make test-api-predict   # Predict service only
```

## Docker Compose Usage

The test service runs as a Docker container that can execute tests against all microservices:

```bash
# Run all tests
docker compose --profile test run --rm test

# Run specific test file
docker compose --profile test run --rm test pytest tests/test_auth_service.py -v

# Run tests with coverage
docker compose --profile test run --rm test pytest tests/ -v --cov=services --cov-report=html

# Run with custom BASE_URL
docker compose --profile test run --rm -e BASE_URL=http://192.168.1.100 test

# Interactive shell in test container
docker compose --profile test run --rm test bash
```

## Test Coverage

The test suite covers:
- **Auth Service**: Login, token refresh, user management, authorization
- **Data Service**: Preprocessing jobs, feature engineering, job status, listing
- **Train Service**: Training jobs, job status, metrics retrieval
- **Predict Service**: Single/batch predictions, model listing

## Configuration

Tests can be configured via environment variables:

```bash
export BASE_URL=http://localhost          # Base URL for API endpoints
export ADMIN_USERNAME=admin              # Admin username
export ADMIN_PASSWORD=your_password      # Admin password
export TEST_USER_USERNAME=testuser       # Test user username
export TEST_USER_PASSWORD=TestUser@123   # Test user password
```

For detailed documentation, see:
- [Test Suite README](../tests/README.md) - Complete test documentation
- [Test Service Guide](../tests/TEST_SERVICE.md) - Docker test service usage

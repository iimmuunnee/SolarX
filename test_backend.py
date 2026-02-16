#!/usr/bin/env python
"""
Quick test script to verify backend is working correctly.
Run this from the project root: python test_backend.py
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all backend modules can be imported."""
    print("Testing backend imports...")
    try:
        from backend.app.main import app
        print("  [OK] FastAPI app imported successfully")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed to import: {e}")
        return False


def test_routes():
    """Test that all routes are registered."""
    from backend.app.main import app

    print("\nTesting API routes...")
    routes = [route.path for route in app.routes]

    expected_routes = [
        "/api/health",
        "/api/vendors",
        "/api/simulate/benchmark",
        "/api/simulate/custom",
        "/api/results/benchmark",
        "/api/results/scalability",
    ]

    all_ok = True
    for route in expected_routes:
        if route in routes:
            print(f"  [OK] {route}")
        else:
            print(f"  [FAIL] Missing route: {route}")
            all_ok = False

    return all_ok


def test_health_endpoint():
    """Test the health endpoint using TestClient."""
    from fastapi.testclient import TestClient
    from backend.app.main import app

    print("\nTesting health endpoint...")
    client = TestClient(app)

    try:
        response = client.get("/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"  [OK] Health check passed: {data}")
            return True
        else:
            print(f"  [FAIL] Health check returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  [FAIL] Health check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SolarX Backend Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: Routes
    if results[0][1]:  # Only if imports worked
        results.append(("Routes", test_routes()))

    # Test 3: Health endpoint
    if results[0][1]:  # Only if imports worked
        results.append(("Health Endpoint", test_health_endpoint()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n>>> Backend is ready! Start it with:")
        print(">>> cd backend && uvicorn app.main:app --reload")
        return 0
    else:
        print("\n>>> Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

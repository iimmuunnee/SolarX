#!/usr/bin/env python
"""
Server startup script for development.
Run this from the backend directory: python run_server.py
"""

import sys
import os

# Add parent directory to path (for SolarX modules to be found)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Starting SolarX Backend Server")
    print("=" * 60)
    print(f"Python path includes: {parent_dir}")
    print(f"Server will run on: http://127.0.0.1:8000")
    print(f"API docs available at: http://127.0.0.1:8000/docs")
    print("=" * 60)
    print()

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

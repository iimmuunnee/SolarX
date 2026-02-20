#!/usr/bin/env python
"""
Server startup script for development.
Run this from the backend directory: python run_server.py
"""

import sys
import os
import argparse

# Add parent directory to path (for SolarX modules to be found)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    import uvicorn
    from app.config import settings

    parser = argparse.ArgumentParser(description="SolarX Backend Server")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to listen on (default: settings.port or PORT env var)")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to bind to (default: settings.host or HOST env var)")
    args = parser.parse_args()

    print("=" * 60)
    print("Starting SolarX Backend Server")
    print("=" * 60)
    print(f"Python path includes: {parent_dir}")
    print(f"Server will run on: http://{args.host}:{args.port}")
    print(f"API docs available at: http://{args.host}:{args.port}/docs")
    print("=" * 60)
    print()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=True,
        log_level="info"
    )

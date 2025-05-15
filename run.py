#!/usr/bin/env python
"""
Main runner script for the solubility prediction web app.
This script provides a convenient command-line interface to train models and start the web server.
"""

import os
import sys
import argparse
import subprocess
import uvicorn

def train_models():
    """Train and save all three models"""
    print("Training models (this may take a few minutes)...")
    subprocess.run([sys.executable, "src/train_models.py"], check=True)
    print("Models trained and saved successfully!")

def start_server(host="127.0.0.1", port=8000, reload=True):
    """Start the FastAPI web server"""
    print(f"Starting web server at http://{host}:{port}")
    uvicorn.run("src.app:app", host=host, port=port, reload=reload)

def main():
    parser = argparse.ArgumentParser(description="Solubility Prediction Web App")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train and save the machine learning models")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the web server")
    server_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    server_parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    # All command
    all_parser = subparsers.add_parser("all", help="Train models and start the server")
    all_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
    all_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    all_parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_models()
    elif args.command == "server":
        start_server(args.host, args.port, not args.no_reload)
    elif args.command == "all":
        train_models()
        start_server(args.host, args.port, not args.no_reload)
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main() 
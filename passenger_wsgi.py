"""
Passenger WSGI entry point for hosting providers
(Phusion Passenger, cPanel, etc.)
"""

import sys
import os

# Add project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Change to project directory
os.chdir(project_dir)

# Import the FastAPI app
try:
    from app import app
    
    # For Passenger/WSGI, we need to expose the app
    application = app
except Exception as e:
    print(f"Error importing app: {e}")
    import traceback
    traceback.print_exc()
    raise

# This is required for some hosting providers
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


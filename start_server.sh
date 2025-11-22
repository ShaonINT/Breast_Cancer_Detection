#!/bin/bash

# Start script for hosting deployment
# Keeps the server running in the background

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Get port from environment or default
PORT=${PORT:-8000}

# Start server with nohup
nohup python app.py > app.log 2>&1 &

# Save PID
echo $! > app.pid

echo "Server started with PID: $(cat app.pid)"
echo "Port: $PORT"
echo "Logs: tail -f app.log"
echo "Stop: kill \$(cat app.pid)"


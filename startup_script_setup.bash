#!/bin/bash

# Variables
SERVICE_NAME="duckcam"
SCRIPT_PATH="/home/pi/Duckcam/main.py"
WORKING_DIR="/home/pi/Duckcam"
PYTHON_PATH=$(which python3)
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# Create systemd service file
echo "Creating systemd service file at ${SERVICE_FILE}..."

sudo bash -c "cat > ${SERVICE_FILE}" <<EOF
[Unit]
Description=Duckcam Startup Script
After=network.target

[Service]
ExecStart=${PYTHON_PATH} ${SCRIPT_PATH}
WorkingDirectory=${WORKING_DIR}
StandardOutput=inherit
StandardError=inherit
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable the service
echo "Reloading systemd and enabling the service..."
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}.service

# Optionally start the service now
echo "Starting the service..."
sudo systemctl start ${SERVICE_NAME}.service

echo "✅ Service '${SERVICE_NAME}' is set up and running on startup."

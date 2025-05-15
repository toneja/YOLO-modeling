#!/bin/bash

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing label-studio..."
pip install label-studio

echo "Done."
read -p "Press ENTER to exit."

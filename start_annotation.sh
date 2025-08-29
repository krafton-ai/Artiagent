#!/bin/bash

echo "🚀 Starting Image Annotation System..."
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

# Install requirements if not already installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing requirements..."
    pip3 install -r annotation_requirements.txt
fi

# Set up demo environment
echo "🔧 Setting up demo environment..."
python3 setup_demo.py

echo ""
echo "🌟 Starting the server..."
echo "📱 Access your annotation system at:"
echo "   • Main page: http://localhost:5000"
echo "   • Classification: http://localhost:5000/classify"  
echo "   • Annotation: http://localhost:5000/annotate"
echo "   • Admin Dashboard: http://localhost:5000/admin"
echo ""
echo "📤 To upload images, visit: http://localhost:5000/upload"
echo "🔗 Share the classification and annotation links with your team!"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="

# Start the server
python3 annotation_server.py

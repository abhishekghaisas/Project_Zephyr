#!/bin/bash

# SeaTac Operations Intelligence - Quick Setup Script
# Run this to set up the project from scratch

set -e  # Exit on error

echo "=========================================================================="
echo "✈️  SeaTac Airport Operations Intelligence - Setup"
echo "=========================================================================="

# Check prerequisites
echo ""
echo "🔍 Checking prerequisites..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ Python: $PYTHON_VERSION"
else
    echo "❌ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

# Check MySQL
if command -v mysql &> /dev/null; then
    echo "✅ MySQL installed"
else
    echo "⚠️  MySQL not found. Install with: brew install mysql"
fi

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "✅ Node.js: $NODE_VERSION"
else
    echo "⚠️  Node.js not found (needed for frontend)"
fi

# Backend setup
echo ""
echo "=========================================================================="
echo "🔧 Setting up Backend"
echo "=========================================================================="

cd backend

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔓 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing Python packages..."
pip install -r requirements.txt

# Create .env from example
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  IMPORTANT: Edit backend/.env and add your API keys!"
    echo "   - GEMINI_API_KEY (get from: https://aistudio.google.com/app/apikey)"
    echo "   - DB_PASSWORD (your MySQL password)"
fi

echo ""
echo "=========================================================================="
echo "✅ Backend Setup Complete!"
echo "=========================================================================="
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Edit backend/.env and add your credentials:"
echo "   - GEMINI_API_KEY"
echo "   - DB_PASSWORD"
echo ""
echo "2. Setup MySQL database:"
echo "   mysql -u root -p"
echo "   CREATE DATABASE aiplane;"
echo "   exit"
echo "   mysql -u root -p aiplane < database/AIplane.sql"
echo ""
echo "3. Start the backend:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "4. Visit: http://localhost:8000/docs"
echo ""
echo "=========================================================================="
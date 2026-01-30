#!/bin/bash

# dbt Co-Work - Run Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🤖 dbt Co-Work - Starting...${NC}"
echo ""

# Check if we're in the right directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi

# Check for config file
if [ ! -f "config.env" ]; then
    echo -e "${YELLOW}⚠️  config.env not found. Creating from template...${NC}"
    cp config.example.env config.env
    echo -e "${YELLOW}Please edit config.env with your API keys${NC}"
fi

# Export environment variables
export $(grep -v '^#' config.env | xargs)

echo -e "${GREEN}✅ Environment ready${NC}"
echo ""
echo -e "${GREEN}🚀 Launching Streamlit dashboard...${NC}"
echo -e "${YELLOW}   URL: http://localhost:8501${NC}"
echo ""

# Run Streamlit
streamlit run app/main.py --server.port 8501 --server.headless true


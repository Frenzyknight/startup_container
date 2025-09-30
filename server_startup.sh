#!/bin/bash

# DotsOCR Server Startup Script
# For Ubuntu container with GPU support
# This script sets up and starts the DotsOCR vLLM inference server

set -e

echo "🚀 Starting DotsOCR Server Setup"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_DIR="/workspace"
DOTSOCR_DIR="/workspace/dots-ocr"
MODEL_PATH="/workspace/dots-ocr/weights/DotsOCR"
VLLM_VERSION="0.9.1"
HOST="0.0.0.0"
PORT="8000"
MODEL_NAME="dotsocr-model"

# Function to print colored output
print_status() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU availability
check_gpu() {
    print_status "Checking GPU availability..."
    if command_exists nvidia-smi; then
        nvidia-smi --query-gpu=name --format=csv,noheader,nounits
        print_success "GPU detected"
        return 0
    else
        print_warning "nvidia-smi not found. GPU support may not be available."
        return 1
    fi
}

# Function to install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."

    # Update package list
    apt-get update

    # Install required packages
    apt-get install -y \
        git \
        curl \
        wget \
        unzip \
        python3-pip \
        python3-dev \
        build-essential \
        software-properties-common

    print_success "System dependencies installed"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."

    # Upgrade pip
    pip install --upgrade pip

    # Install vLLM with specific version
    print_status "Installing vLLM ${VLLM_VERSION}..."
    pip install vllm==${VLLM_VERSION}

    print_success "vLLM installed"
}

# Function to clone and setup DotsOCR
setup_dotsocr() {
    print_status "Setting up DotsOCR..."

    # Create workspace directory
    mkdir -p ${WORKSPACE_DIR}
    cd ${WORKSPACE_DIR}

    # Clone repository if it doesn't exist
    if [ ! -d "dots-ocr" ]; then
        print_status "Cloning DotsOCR repository..."
        git clone https://github.com/rednote-hilab/dots.ocr.git dots-ocr
        print_success "Repository cloned"
    else
        print_status "DotsOCR repository already exists"
    fi

    cd ${DOTSOCR_DIR}

    # Install DotsOCR requirements
    print_status "Installing DotsOCR requirements..."
    pip install --no-cache-dir -r requirements.txt

    # Install DotsOCR in development mode
    print_status "Installing DotsOCR package..."
    pip install -e .

    print_success "DotsOCR setup completed"
}

# Function to download model weights
download_model() {
    print_status "Downloading model weights..."

    cd ${DOTSOCR_DIR}

    if [ ! -d "weights/DotsOCR" ]; then
        python3 tools/download_model.py
        print_success "Model weights downloaded"
    else
        print_status "Model weights already exist"
    fi

    # Verify model path
    if [ ! -d "${MODEL_PATH}" ]; then
        print_error "Model path does not exist: ${MODEL_PATH}"
        exit 1
    fi

    print_success "Model weights verified"
}

# Function to configure vLLM for DotsOCR
configure_vllm() {
    print_status "Configuring vLLM for DotsOCR..."

    # Set environment variables
    export HF_MODEL_PATH="${MODEL_PATH}"
    export PYTHONPATH="/workspace/dots-ocr/weights:$PYTHONPATH"

    # Find vLLM executable
    VLLM_PATH=$(which vllm)
    if [ -z "$VLLM_PATH" ]; then
        print_error "vLLM executable not found"
        exit 1
    fi

    # Add DotsOCR import to vLLM
    print_status "Patching vLLM for DotsOCR support..."
    if ! grep -q "from DotsOCR import modeling_dots_ocr_vllm" "$VLLM_PATH"; then
        sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\from DotsOCR import modeling_dots_ocr_vllm' "$VLLM_PATH"
        print_success "vLLM patched successfully"
    else
        print_status "vLLM already patched"
    fi
}

# Function to start vLLM server
start_server() {
    print_status "Starting vLLM server..."

    # Check if port is already in use
    if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null ; then
        print_warning "Port ${PORT} is already in use"
        print_status "Attempting to stop existing server..."
        pkill -f "vllm serve" || true
        sleep 5
    fi

    print_status "Launching vLLM server with DotsOCR model..."
    print_status "Model path: ${MODEL_PATH}"
    print_status "Host: ${HOST}:${PORT}"
    print_status "Model name: ${MODEL_NAME}"

    # Start vLLM server
    CUDA_VISIBLE_DEVICES=0 vllm serve "${MODEL_PATH}" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.95 \
        --chat-template-content-format string \
        --served-model-name "${MODEL_NAME}" \
        --trust-remote-code \
        --host "${HOST}" \
        --port "${PORT}" \
        --max-model-len 20000 &

    SERVER_PID=$!

    # Wait for server to start
    print_status "Waiting for server to start..."
    sleep 10

    # Check if server is running
    for i in {1..30}; do
        if curl -s "http://${HOST}:${PORT}/health" > /dev/null; then
            print_success "vLLM server started successfully!"
            print_success "Server is running at http://${HOST}:${PORT}"
            print_success "Model name: ${MODEL_NAME}"
            print_success "Health endpoint: http://${HOST}:${PORT}/health"
            break
        fi

        if [ $i -eq 30 ]; then
            print_error "Server failed to start within timeout"
            exit 1
        fi

        print_status "Waiting for server... (${i}/30)"
        sleep 2
    done
}

# Function to cleanup on exit
cleanup() {
    print_status "Cleaning up..."
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
    fi
    pkill -f "vllm serve" 2>/dev/null || true
}

# Trap cleanup on exit
trap cleanup EXIT INT TERM

# Main execution
main() {
    echo -e "${GREEN}🔧 DotsOCR Server Setup Starting...${NC}"
    echo ""

    # Check if running as root (required for apt-get)
    if [ "$EUID" -ne 0 ]; then
        print_error "This script must be run as root (use sudo)"
        exit 1
    fi

    # Check GPU
    check_gpu
    echo ""

    # Install dependencies
    install_system_deps
    echo ""

    install_python_deps
    echo ""

    # Setup DotsOCR
    setup_dotsocr
    echo ""

    # Download model
    download_model
    echo ""

    # Configure vLLM
    configure_vllm
    echo ""

    # Start server
    start_server
    echo ""

    print_success "🎉 DotsOCR server is running!"
    echo "================================"
    echo -e "${BLUE}📡 Server URL:${NC}      http://${HOST}:${PORT}"
    echo -e "${BLUE}🏥 Health Check:${NC}   http://${HOST}:${PORT}/health"
    echo -e "${BLUE}📖 Model Name:${NC}     ${MODEL_NAME}"
    echo -e "${BLUE}📁 Model Path:${NC}     ${MODEL_PATH}"
    echo ""
    echo -e "${YELLOW}💡 Tips:${NC}"
    echo "   • Test the server: curl http://${HOST}:${PORT}/health"
    echo "   • View logs in real-time with this script"
    echo "   • Press Ctrl+C to stop the server"
    echo ""

    # Keep script running and monitor
    print_status "🔄 Monitoring server... (Press Ctrl+C to stop)"
    while true; do
        sleep 10
        if ! curl -s "http://${HOST}:${PORT}/health" > /dev/null; then
            print_warning "Server health check failed"
        fi
    done
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
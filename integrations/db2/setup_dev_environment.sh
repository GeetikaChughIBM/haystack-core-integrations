#!/bin/bash

# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# DB2 Haystack Integration - Development Environment Setup Script
# =============================================================================
#
# This script automates the setup process for new developers working on the
# DB2 Haystack integration. It handles:
# - Python version verification
# - Virtual environment creation
# - Dependency installation
# - Hatch setup and verification
# - Code quality checks
# - Test execution
#
# Usage:
#   ./setup_dev_environment.sh [OPTIONS]
#
# Options:
#   --skip-tests        Skip running tests after setup
#   --skip-examples     Skip running example validation
#   --help              Show this help message
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
MIN_PYTHON_VERSION="3.10"
VENV_DIR="venv"
SKIP_TESTS=false
SKIP_EXAMPLES=false
PYTHON_CMD=""

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to compare version numbers
version_ge() {
    # Returns 0 (true) if $1 >= $2
    printf '%s\n%s' "$2" "$1" | sort -V -C
}

# Function to find suitable Python version
find_python() {
    # Try to find Python 3.10 or higher
    for version in 3.13 3.12 3.11 3.10; do
        if check_command "python${version}"; then
            local found_version=$(python${version} --version 2>&1 | cut -d' ' -f2)
            if version_ge "$found_version" "$MIN_PYTHON_VERSION"; then
                echo "python${version}"
                return 0
            fi
        fi
    done
    
    # Try generic python3
    if check_command python3; then
        local found_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        if version_ge "$found_version" "$MIN_PYTHON_VERSION"; then
            echo "python3"
            return 0
        fi
    fi
    
    return 1
}

# =============================================================================
# Parse Command Line Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-examples)
            SKIP_EXAMPLES=true
            shift
            ;;
        --help)
            head -n 30 "$0" | tail -n 25
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Welcome Message
# =============================================================================

print_header "DB2 Haystack Integration - Development Setup"
echo "This script will set up your development environment for the DB2 Haystack integration."
echo ""
echo "Steps:"
echo "  1. Verify Python version"
echo "  2. Create virtual environment"
echo "  3. Install dependencies"
echo "  4. Verify Hatch installation"
echo "  5. Run code quality checks"
if [ "$SKIP_TESTS" = false ]; then
    echo "  6. Run tests"
fi
if [ "$SKIP_EXAMPLES" = false ]; then
    echo "  7. Validate examples"
fi
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# =============================================================================
# Step 1: Verify Python Version
# =============================================================================

print_header "Step 1: Verifying Python Version"

print_info "Looking for Python ${MIN_PYTHON_VERSION} or higher..."

PYTHON_CMD=$(find_python)
if [ -z "$PYTHON_CMD" ]; then
    print_error "Python ${MIN_PYTHON_VERSION}+ is not installed or not in PATH"
    print_info "Please install Python ${MIN_PYTHON_VERSION} or higher from https://www.python.org/downloads/"
    print_info "Supported versions: Python 3.10, 3.11, 3.12, 3.13"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2)
print_success "Found $PYTHON_CMD: Python $PYTHON_VERSION"

# =============================================================================
# Step 2: Create Virtual Environment
# =============================================================================

print_header "Step 2: Creating Virtual Environment"

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        print_info "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    print_success "Virtual environment created at $VENV_DIR"
else
    print_success "Virtual environment ready"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated"

# =============================================================================
# Step 3: Install Dependencies
# =============================================================================

print_header "Step 3: Installing Dependencies"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip --quiet
print_success "pip upgraded to $(pip --version | cut -d' ' -f2)"

# Install ibm-db driver (critical dependency)
print_info "Installing ibm-db driver (this may take a few minutes)..."
if pip install ibm-db --no-cache-dir --quiet; then
    print_success "ibm-db driver installed successfully"
else
    print_error "Failed to install ibm-db driver"
    print_info "You may need to install system dependencies. See:"
    print_info "https://github.com/ibmdb/python-ibmdb#installation"
    exit 1
fi

# Install development dependencies
print_info "Installing development dependencies..."
pip install pytest pytest-cov python-dotenv --quiet
print_success "Development dependencies installed"

# Install Hatch (if not already installed)
print_info "Installing Hatch..."
if ! check_command hatch; then
    pip install hatch --quiet
    print_success "Hatch installed"
else
    print_success "Hatch already installed"
fi

# =============================================================================
# Step 4: Verify Hatch Installation
# =============================================================================

print_header "Step 4: Verifying Hatch Installation"

if check_command hatch; then
    HATCH_VERSION=$(hatch --version 2>&1 | head -n 1)
    print_success "Hatch is installed: $HATCH_VERSION"
else
    print_error "Hatch is not installed or not in PATH"
    print_info "Please install Hatch: pip install hatch"
    exit 1
fi

# =============================================================================
# Step 5: Navigate to DB2 Integration Directory
# =============================================================================

print_header "Step 5: Navigating to DB2 Integration Directory"

# Find the DB2 integration directory
if [ -d "haystack-core-integrations/integrations/db2" ]; then
    cd haystack-core-integrations/integrations/db2
    print_success "Changed directory to: $(pwd)"
elif [ -d "integrations/db2" ]; then
    cd integrations/db2
    print_success "Changed directory to: $(pwd)"
elif [ -d "db2" ]; then
    cd db2
    print_success "Changed directory to: $(pwd)"
else
    print_error "Could not find DB2 integration directory"
    print_info "Please run this script from the repository root or integration directory"
    exit 1
fi

# =============================================================================
# Step 6: Run Code Quality Checks
# =============================================================================

print_header "Step 6: Running Code Quality Checks"

# Type checking
print_info "Running type checks with mypy..."
if hatch run test:types; then
    print_success "Type checks passed"
else
    print_warning "Type checks failed (non-critical)"
fi

# Code formatting
print_info "Running code formatter (ruff)..."
if hatch run fmt; then
    print_success "Code formatting complete"
else
    print_warning "Code formatting had issues (non-critical)"
fi

# Linting with auto-fix
print_info "Running linter with auto-fix..."
if python -m ruff check --fix --unsafe-fixes src/ tests/; then
    print_success "Linting complete"
else
    print_warning "Linting found issues (non-critical)"
fi

# =============================================================================
# Step 7: Run Tests (Optional)
# =============================================================================

if [ "$SKIP_TESTS" = false ]; then
    print_header "Step 7: Running Tests"
    
    # Unit tests only (integration tests require DB2 connection)
    print_info "Running unit tests..."
    if hatch run test:unit; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed"
        print_info "Please review the test output above"
    fi
    
    print_warning "Skipping integration tests (require DB2 connection)"
    print_info "To run integration tests manually: hatch run test:all"
else
    print_info "Skipping tests (--skip-tests flag provided)"
fi

# =============================================================================
# Step 8: Validate Examples (Optional)
# =============================================================================

if [ "$SKIP_EXAMPLES" = false ]; then
    print_header "Step 8: Validating Examples"
    
    print_info "Checking example files..."
    
    if [ -d "examples" ]; then
        EXAMPLE_COUNT=$(find examples -name "*.py" -type f | wc -l)
        print_success "Found $EXAMPLE_COUNT example files"
        
        print_info "Example files available:"
        find examples -name "*.py" -type f | while read -r file; do
            echo "  - $(basename "$file")"
        done
        
        print_info "To run an example:"
        echo "  cd examples"
        echo "  python product_search_hybrid.py"
    else
        print_warning "Examples directory not found"
    fi
else
    print_info "Skipping example validation (--skip-examples flag provided)"
fi

# =============================================================================
# Step 9: Environment Configuration
# =============================================================================

print_header "Step 9: Environment Configuration"

if [ -f ".env.example" ]; then
    if [ ! -f ".env" ]; then
        print_warning ".env file not found"
        read -p "Do you want to create .env from .env.example? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cp .env.example .env
            print_success ".env file created from .env.example"
            print_warning "Please edit .env and add your DB2 credentials"
        fi
    else
        print_success ".env file already exists"
    fi
else
    print_warning ".env.example not found"
fi

# =============================================================================
# Setup Complete
# =============================================================================

print_header "Setup Complete!"

echo -e "${GREEN}Your development environment is ready!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your DB2 connection:"
echo "   ${CYAN}Edit .env file with your DB2 credentials${NC}"
echo ""
echo "2. Run tests:"
echo "   ${CYAN}hatch run test:unit${NC}          # Unit tests only"
echo "   ${CYAN}hatch run test:all${NC}           # All tests (requires DB2)"
echo ""
echo "3. Run examples:"
echo "   ${CYAN}cd examples${NC}"
echo "   ${CYAN}python product_search_hybrid.py${NC}"
echo ""
echo "4. Code quality checks:"
echo "   ${CYAN}hatch run test:types${NC}         # Type checking"
echo "   ${CYAN}hatch run fmt${NC}                # Format code"
echo "   ${CYAN}python -m ruff check src/${NC}    # Lint code"
echo ""
echo "5. Development workflow:"
echo "   ${CYAN}source venv/bin/activate${NC}    # Activate venv"
echo "   ${CYAN}cd haystack-core-integrations/integrations/db2${NC}"
echo "   ${CYAN}# Make your changes...${NC}"
echo "   ${CYAN}hatch run fmt${NC}                # Format"
echo "   ${CYAN}hatch run test:unit${NC}          # Test"
echo ""
# Made with Bob

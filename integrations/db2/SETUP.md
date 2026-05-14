# DB2 Haystack Integration - Development Setup Guide

This guide helps new developers set up their development environment for the DB2 Haystack integration.

## Table of Contents

- [Quick Start (Automated)](#quick-start-automated)
- [Manual Setup](#manual-setup)
- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Running Tests](#running-tests)
- [Running Examples](#running-examples)
- [Development Workflow](#development-workflow)
- [Troubleshooting](#troubleshooting)

---

## Quick Start (Automated)

We provide an automated setup script that handles the entire development environment setup:

```bash
# From the repository root
cd haystack-core-integrations/integrations/db2

# Run the setup script
./setup_dev_environment.sh
```

### Script Options

```bash
./setup_dev_environment.sh --help              # Show help
./setup_dev_environment.sh --skip-tests        # Skip running tests
./setup_dev_environment.sh --skip-examples     # Skip example validation
```

### What the Script Does

1. ✅ Verifies Python 3.11 installation
2. ✅ Creates virtual environment
3. ✅ Upgrades pip
4. ✅ Installs `ibm-db` driver
5. ✅ Installs development dependencies (`pytest`, `pytest-cov`, `python-dotenv`)
6. ✅ Installs and verifies Hatch
7. ✅ Runs type checks (`hatch run test:types`)
8. ✅ Runs code formatter (`hatch run fmt`)
9. ✅ Runs linter with auto-fix (`ruff`)
10. ✅ Runs unit tests (optional)
11. ✅ Validates examples (optional)
12. ✅ Creates `.env` from `.env.example` (optional)

---

## Manual Setup

If you prefer to set up manually or the automated script doesn't work for your environment:

### Step 1: Prerequisites

Ensure you have the following installed:

- **Python 3.10+** (required - Python 3.10, 3.11, 3.12, or 3.13)
- **pip** (latest version)
- **Git**
- **Hatch** (will be installed if not present)

#### Install Python 3.10+

**Ubuntu/Debian:**
```bash
sudo apt update
# Install Python 3.11 (recommended)
sudo apt install python3.11 python3.11-venv python3.11-dev

# Or Python 3.10
sudo apt install python3.10 python3.10-venv python3.10-dev
```

**macOS (using Homebrew):**
```bash
# Install Python 3.11 (recommended)
brew install python@3.11

# Or Python 3.10
brew install python@3.10
```

**Windows:**
Download Python 3.10+ from [python.org](https://www.python.org/downloads/)

**Note:** The setup script will automatically detect and use the highest available Python version (3.10+).

### Step 2: Clone Repository

```bash
git clone https://github.com/deepset-ai/haystack-core-integrations.git
cd haystack-core-integrations/integrations/db2
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment with Python 3.10+
# The script will auto-detect the best version, or you can specify:
python3.11 -m venv venv  # If you have Python 3.11
# OR
python3.10 -m venv venv  # If you have Python 3.10

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows
```

**Note:** The automated setup script (`setup_dev_environment.sh`) will automatically find and use the highest available Python version (3.10+).

### Step 4: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 5: Install IBM DB2 Driver

The `ibm-db` driver is critical for DB2 connectivity:

```bash
pip install ibm-db --no-cache-dir
```

**Note:** This may take several minutes as it compiles native extensions.

#### Troubleshooting ibm-db Installation

If installation fails, you may need system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get install gcc python3-dev libxml2-dev libxslt1-dev
```

**macOS:**
```bash
xcode-select --install
```

**Windows:**
- Install Visual Studio Build Tools
- See: https://github.com/ibmdb/python-ibmdb#installation

### Step 6: Install Development Dependencies

```bash
pip install pytest pytest-cov python-dotenv
```

### Step 7: Install Hatch

Hatch is used for project management, testing, and code quality:

```bash
pip install hatch
```

Verify installation:
```bash
hatch --version
```

### Step 8: Verify Setup

```bash
# Run type checks
hatch run test:types

# Format code
hatch run fmt

# Run unit tests
hatch run test:unit
```

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.11+ | Runtime environment |
| pip | Latest | Package management |
| Hatch | Latest | Project management & testing |
| ibm-db | Latest | DB2 driver |

### Required Python Packages

Installed automatically by the setup script:

- `pytest` - Testing framework
- `pytest-cov` - Code coverage
- `python-dotenv` - Environment variable management
- `ibm-db` - IBM DB2 driver

### Optional Tools

- **DB2 CLI** - For manual database validation
- **Docker** - For running DB2 in containers (testing)

---

## Environment Configuration

### Step 1: Create .env File

```bash
# Copy example configuration
cp .env.example .env
```

### Step 2: Configure DB2 Credentials

Edit `.env` and add your DB2 connection details:

```bash
# DB2 Connection Settings
DB2_HOST=your-db2-host.com
DB2_PORT=50000
DB2_DATABASE=your_database
DB2_USERNAME=your_username
DB2_PASSWORD=your_password
DB2_SCHEMA=your_schema  # Optional, defaults to username

# Optional: SSL Configuration
DB2_USE_SSL=false
DB2_SSL_CERT_PATH=/path/to/cert.pem

# Optional: Connection Pool Settings
DB2_POOL_SIZE=5
DB2_MAX_OVERFLOW=10
```

### Step 3: Verify Connection

Test your DB2 connection:

```bash
cd examples
python basic_usage.py
```

---

## Running Tests

### Unit Tests Only

Unit tests don't require a DB2 connection:

```bash
hatch run test:unit
```

### Integration Tests

Integration tests require a DB2 connection configured in `.env`:

```bash
hatch run test:all
```

**Note:** Integration tests may fail if DB2 is not accessible. This is expected in local development.

### Type Checking

```bash
hatch run test:types
```

### Code Coverage

```bash
hatch run test:cov
```

### Specific Test File

```bash
pytest tests/test_document_store.py -v
```

### Specific Test Function

```bash
pytest tests/test_document_store.py::test_write_documents -v
```

---

## Running Examples

All examples are located in the `examples/` directory.

### Basic Usage

```bash
cd examples
python basic_usage.py
```

### Vector Search Examples

```bash
# Pure vector search
python embedding_retrieval.py

# Hybrid search (vector + keyword)
python hybrid_retrieval.py

# Product search with intelligent parsing
python product_search_hybrid.py
```

### Advanced Examples

```bash
# Reranking with cross-encoder
python reranking_example.py

# Model validation
python model_validation_example.py

# Complete product search scenarios
python product_search_from_db2_table.py
```

### Example Requirements

All examples require:
1. ✅ Valid `.env` configuration
2. ✅ DB2 connection available
3. ✅ Virtual environment activated

---

## Development Workflow

### Daily Development Cycle

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Navigate to DB2 integration
cd haystack-core-integrations/integrations/db2

# 3. Make your changes to src/ or tests/

# 4. Format code
hatch run fmt

# 5. Run linter
python -m ruff check --fix --unsafe-fixes src/ tests/

# 6. Run type checks
hatch run test:types

# 7. Run unit tests
hatch run test:unit

# 8. Commit changes
git add .
git commit -m "Your descriptive commit message"
```

### Code Quality Commands

```bash
# Format code (black + isort)
hatch run fmt

# Lint with auto-fix
python -m ruff check --fix src/ tests/

# Lint with unsafe fixes
python -m ruff check --fix --unsafe-fixes src/ tests/

# Type checking
hatch run test:types

# All quality checks
hatch run fmt && hatch run test:types && python -m ruff check src/
```

### Testing Commands

```bash
# Unit tests only (fast, no DB2 required)
hatch run test:unit

# Integration tests (requires DB2)
hatch run test:all

# Specific test file
pytest tests/test_filters.py -v

# With coverage report
hatch run test:cov

# Watch mode (re-run on file changes)
pytest-watch tests/
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add new feature"

# Push to remote
git push origin feature/your-feature-name

# Create pull request on GitHub
```

---

## Troubleshooting

### Issue 1: ibm-db Installation Fails

**Symptoms:**
```
error: command 'gcc' failed with exit status 1
```

**Solution:**
Install system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install gcc python3-dev libxml2-dev libxslt1-dev

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools
```

### Issue 2: Python 3.11 Not Found

**Symptoms:**
```
python3.11: command not found
```

**Solution:**
Install Python 3.11:

```bash
# Ubuntu/Debian
sudo apt install python3.11

# macOS
brew install python@3.11

# Verify
python3.11 --version
```

### Issue 3: Hatch Command Not Found

**Symptoms:**
```
hatch: command not found
```

**Solution:**
```bash
pip install hatch
# OR
python -m pip install hatch

# Verify
hatch --version
```

### Issue 4: DB2 Connection Fails

**Symptoms:**
```
ibm_db.conn_error: [IBM][CLI Driver] SQL30081N
```

**Solution:**
1. Verify `.env` configuration
2. Check DB2 host/port accessibility
3. Verify credentials
4. Check firewall rules
5. Test with DB2 CLI:
   ```bash
   db2 connect to DATABASE user USERNAME using PASSWORD
   ```

### Issue 5: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'haystack_integrations'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall in development mode
cd haystack-core-integrations/integrations/db2
pip install -e .
```

### Issue 6: Tests Fail with "Connection Refused"

**Symptoms:**
```
Connection refused: DB2 server not accessible
```

**Solution:**
- Integration tests require DB2 connection
- Run unit tests only: `hatch run test:unit`
- Or configure DB2 connection in `.env`

### Issue 7: Permission Denied on setup_dev_environment.sh

**Symptoms:**
```
bash: ./setup_dev_environment.sh: Permission denied
```

**Solution:**
```bash
chmod +x setup_dev_environment.sh
./setup_dev_environment.sh
```

---

## Additional Resources

### Documentation

- **Architecture Guide**: [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- **Product Search Architecture**: [`examples/PRODUCT_SEARCH_ARCHITECTURE.md`](./examples/PRODUCT_SEARCH_ARCHITECTURE.md)
- **Contributing Guidelines**: [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md)
- **Haystack Documentation**: https://docs.haystack.deepset.ai/

### Example Files

| File | Description |
|------|-------------|
| `basic_usage.py` | Basic document store operations |
| `embedding_retrieval.py` | Vector search examples |
| `hybrid_retrieval.py` | Hybrid search (vector + keyword) |
| `product_search_hybrid.py` | Product search with intelligent parsing |
| `reranking_example.py` | Cross-encoder reranking |
| `model_validation_example.py` | Embedding model validation |

### Useful Commands Reference

```bash
# Environment
source venv/bin/activate              # Activate venv
deactivate                            # Deactivate venv

# Testing
hatch run test:unit                   # Unit tests
hatch run test:all                    # All tests
hatch run test:cov                    # With coverage
pytest tests/ -v                      # Verbose output
pytest tests/ -k "test_name"          # Specific test

# Code Quality
hatch run fmt                         # Format code
hatch run test:types                  # Type check
python -m ruff check src/             # Lint
python -m ruff check --fix src/       # Lint + fix

# Examples
cd examples && python basic_usage.py  # Run example

# Git
git status                            # Check status
git add .                             # Stage changes
git commit -m "message"               # Commit
git push origin branch-name           # Push
```

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check existing issues**: https://github.com/deepset-ai/haystack-core-integrations/issues
2. **Create new issue**: Provide error messages, environment details, and steps to reproduce
3. **Ask in discussions**: https://github.com/deepset-ai/haystack-core-integrations/discussions
4. **Haystack Discord**: https://discord.gg/haystack

---

## Summary

You now have a complete development environment for the DB2 Haystack integration!

**Quick Reference:**
```bash
# Setup (one-time)
./setup_dev_environment.sh

# Daily workflow
source venv/bin/activate
cd haystack-core-integrations/integrations/db2
# Make changes...
hatch run fmt && hatch run test:unit
git commit -m "Your changes"

# Run examples
cd examples
python product_search_hybrid.py
```

Happy coding! 🚀
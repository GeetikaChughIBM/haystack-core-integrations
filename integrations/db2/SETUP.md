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

1. ✅ Verifies Python 3.10+ installation
2. ✅ Creates virtual environment
3. ✅ Upgrades pip
4. ✅ Installs [`ibm-db`](./pyproject.toml)
5. ✅ Installs development dependencies
6. ✅ Installs and verifies Hatch
7. ✅ Runs type checks
8. ✅ Runs formatting and linting
9. ✅ Runs unit tests (optional)
10. ✅ Lists available examples (optional)
11. ✅ Creates [`.env`](./.env) from [`.env.example`](./.env.example) if needed

---

## Manual Setup

If you prefer to set up manually or the automated script doesn't work for your environment:

### Step 1: Prerequisites

Ensure you have the following installed:

- **Python 3.10+**
- **pip**
- **Git**
- **Hatch**

#### Install Python 3.10+

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**macOS (using Homebrew):**
```bash
brew install python@3.11
```

**Windows:**
Download Python from [python.org](https://www.python.org/downloads/)

### Step 2: Clone Repository

```bash
git clone https://github.com/deepset-ai/haystack-core-integrations.git
cd haystack-core-integrations/integrations/db2
```

### Step 3: Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### Step 4: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 5: Install IBM DB2 Driver

```bash
pip install ibm-db --no-cache-dir
```

### Step 6: Install Development Dependencies

```bash
pip install pytest pytest-cov pytest-asyncio python-dotenv hatch
```

### Step 7: Verify Setup

```bash
hatch run test:types
hatch run fmt
hatch run test:unit
```

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Runtime environment |
| pip | Latest | Package management |
| Hatch | Latest | Project management and testing |
| ibm-db | Latest | DB2 driver |

### Optional Tools

- **DB2 CLI** - For manual database validation
- **Docker** - For running DB2 in containers (testing)

---

## Environment Configuration

### Step 1: Create `.env`

```bash
cp .env.example .env
```

### Step 2: Configure DB2 Credentials

Edit [`.env`](./.env) and provide your DB2 connection details:

```bash
DB2_DATABASE=TESTDB
DB2_HOSTNAME=your-db2-host.example.com
DB2_PORT=50000
DB2_SSL_PORT=50001
DB2_SSL_ENABLED=false
DB2_USER=your_username
DB2_PASSWORD=your_password
```

Optional settings:

```bash
# Optional full connection string
# DB2_CONNECTION_STRING=DATABASE=TESTDB;HOSTNAME=your-db2-host.example.com;PORT=50000;PROTOCOL=TCPIP;UID=your_username;PWD=your_password;

# Optional SSL certificate path
# DB2_SSL_CERTIFICATE=/absolute/path/to/db2server.crt
# DB2_SSL_CERT_PATH=/absolute/path/to/db2server.crt

# Optional test/example overrides
# DB2_TEST_TABLE=haystack_documents
# DB2_EMBEDDING_DIMENSION=384
```

### Step 3: Verify Connection

```bash
cd examples
python basic_usage.py
```

---

## Running Tests

### Unit Tests Only

```bash
hatch run test:unit
```

### Integration Tests

Integration tests require a reachable DB2 instance configured through [`.env`](./.env):

```bash
hatch run test:integration
```

### All Tests

```bash
hatch run test:all
```

### Type Checking

```bash
hatch run test:types
```

### Coverage

```bash
pytest tests/ --cov=haystack_integrations --cov-report=html
```

### Specific Test File

```bash
pytest tests/test_document_store.py -v
```

---

## Running Examples

All examples are located in the [`examples/`](./examples/) directory.

### Basic Usage

```bash
cd examples
python basic_usage.py
```

### Retrieval Examples

```bash
python embedding_retrieval.py
python hybrid_retrieval.py
python hybrid_retrieval_simple.py
```

### Product Example

```bash
python product_search_hybrid.py
```

### Example Requirements

All examples require:

1. ✅ Valid [`.env`](./.env) configuration
2. ✅ Reachable DB2 connection
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
pytest tests/test_keyword_retriever.py -v
```

---

## Troubleshooting

### Issue 1: `ibm-db` Installation Fails

**Symptoms:**
```bash
error: command 'gcc' failed with exit status 1
```

**Solution:**
Install system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get install gcc python3-dev libxml2-dev libxslt1-dev
```

**macOS:**
```bash
xcode-select --install
```

**Windows:**
Install Visual Studio Build Tools.

### Issue 2: Python Not Found

**Symptoms:**
```bash
python3.11: command not found
```

**Solution:**
Install Python 3.10+ and verify:

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

### Issue 7: Permission Denied on [`setup_dev_environment.sh`](./setup_dev_environment.sh)

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

- **[README](./README.md)** - Package overview
- **[Examples](./examples/)** - Available example scripts
- **[Contributing Guidelines](../../CONTRIBUTING.md)** - Repository-wide contribution guide
- **Haystack Documentation** - https://docs.haystack.deepset.ai/

---

## Summary

You now have a development environment for the DB2 Haystack integration.

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
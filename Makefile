# GNU nano 7.2    Makefile
# Variables
PYTHON = python3
PIP = pip
VENV = venv

# Install dependencies
install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt

# Train the model
train:
	@echo "Training the model..."
	$(PYTHON) main.py --train --save

# Evaluate the model
evaluate:
	@echo "Evaluating the model..."
	$(PYTHON) main.py --evaluate --load

# Code Quality - Linting
lint:
	@echo "Running code quality checks (flake8)..."
	flake8 .

# Code Formatting - using black
format:
	@echo "Formatting code (black)..."
	black .

# Security check (using bandit)
security:
	@echo "Running security checks (bandit)..."
	bandit -r .

# Clean up Python bytecode files
clean:
	@echo "Cleaning up Python bytecode files..."
	find . -name "*.pyc" -exec rm -f {} \;

# Default target to install, train, and evaluate
all: install train evaluate

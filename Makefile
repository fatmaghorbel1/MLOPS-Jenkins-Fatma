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
	$(PYTHON) notify.py "Training Completed" "The model has been trained successfully!"
# Evaluate the model
evaluate:
	@echo "Evaluating the model..."
	$(PYTHON) main.py --evaluate --load
	$(PYTHON) notify.py "Evaluation Completed" "The model evaluation is done!"
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

# Start MLflow UI
mlflow-ui:
	@echo "Starting MLflow UI..."
	mlflow ui --host 0.0.0.0 --port 5000					
	$(PYTHON) notify.py "MLflow UI Started" "The MLflow UI is running on port 5000!"
# Default target to install, train, and evaluate
all: install train evaluate							
	$(PYTHON) notify.py "Pipeline Completed" "All tasks (install, train, evaluate) are finished!"

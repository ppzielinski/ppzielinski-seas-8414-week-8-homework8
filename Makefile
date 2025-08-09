# Makefile
# Auto-detects whether to use 'docker-compose' or 'docker compose'
COMPOSE_CMD := $(shell command -v docker-compose 2>/dev/null)
ifeq ($(COMPOSE_CMD),)
  COMPOSE_CMD := docker compose
endif

.PHONY: all up down clean logs

# Default command
all: up

# Build and start the application
up:
	@echo "Starting application..."
	@$(COMPOSE_CMD) up --build -d

# Stop the application
down:
	@echo "Stopping application..."
	@$(COMPOSE_CMD) down

# Clean up the environment
clean: down
	@echo "Cleaning up generated files..."
	@rm -rf ./models ./data ./__pycache__

# View application logs
logs:
	@echo "Tailing logs..."
	@$(COMPOSE_CMD) logs -f
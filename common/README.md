# Common Directory

This directory contains shared code and utilities used across multiple tools in the collection.

## Contents

- **Utils** - Common utility functions
- **API** - Shared API clients and interfaces
- **Models** - Shared data models and schemas
- **Config** - Common configuration utilities
- **Logging** - Standardized logging framework
- **Testing** - Shared test utilities and fixtures

## Best Practices

When adding code to the common directory:

1. Ensure it's truly reusable across multiple tools
2. Write comprehensive documentation and examples
3. Include unit tests for all functionality
4. Follow consistent coding standards
5. Minimize dependencies
6. Version modules appropriately

## Key Modules

- **google_apis.py** - Utilities for interacting with Google APIs
- **firebase_utils.py** - Firebase integration utilities
- **gemini_client.py** - Client for Gemini API interactions
- **media_utils.py** - Media file processing utilities
- **validation.py** - Input/output validation functions
- **error_handling.py** - Standardized error handling
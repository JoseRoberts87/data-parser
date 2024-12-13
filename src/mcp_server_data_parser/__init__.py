from . import server
import asyncio

def main():
    """Main entry point for the package."""
    print("Starting server...")
    asyncio.run(server.main())

# Optionally expose other important items at package level
__all__ = ['main', 'server']

if __name__ == "__main__":
    main()

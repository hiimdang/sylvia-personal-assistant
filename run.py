import asyncio
import sys
import uvicorn
import os
from dotenv import load_dotenv
from pathlib import Path

ENV_FILE_PATH = Path(__file__).parent / ".env"
if ENV_FILE_PATH.exists():
    load_dotenv(ENV_FILE_PATH)

if __name__ == '__main__':
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=6969,
        reload=True
    )

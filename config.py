import os
# Load .env file manually if python-dotenv is not available
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

CREDENTIALS = {
    "username": os.getenv("BRAIN_USERNAME", ""),
    "password": os.getenv("BRAIN_PASSWORD", ""),
}

# WorldQuant Brain API base URL
BASE_URL = "https://api.worldquantbrain.com"

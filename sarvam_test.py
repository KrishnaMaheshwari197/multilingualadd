"""One Sarvam chat call to check the setup. Run: python3 sarvam_test.py"""

import os
from pathlib import Path

from sarvamai import SarvamAI

# Load .env without an extra dependency; a real environment variable wins.
env = Path(__file__).with_name(".env")
if env.exists():
    for line in env.read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            os.environ.setdefault(name.strip(), value.strip())

key = os.environ.get("SARVAM_API_KEY")
if not key:
    raise SystemExit("SARVAM_API_KEY is not set. Put it in .env.")

client = SarvamAI(api_subscription_key=key)
response = client.chat.completions(
    model="sarvam-105b-conversations",
    messages=[{"role": "user", "content": "Say namaste and introduce yourself in one line, in Hindi."}],
)
print(response.choices[0].message.content)

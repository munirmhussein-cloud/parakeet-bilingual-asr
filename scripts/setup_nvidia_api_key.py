"""
Securely prompt for the NVIDIA API key.

The key is stored only in the current Python process via
os.environ["NVIDIA_API_KEY"].
"""

import getpass
import os

if "NVIDIA_API_KEY" not in os.environ:
    os.environ["NVIDIA_API_KEY"] = getpass.getpass("NVIDIA API key: ")

print("✓ NVIDIA_API_KEY loaded into environment.")

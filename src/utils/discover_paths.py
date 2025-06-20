import sys
import os

print("--- sys.path inspection (from LuminDetectNet4web.py) ---")
print(f"Current Working Directory: {os.getcwd()}")
print(f"Script Directory: {os.path.dirname(os.path.abspath(__file__))}")
print("sys.path entries:")
for p in sys.path:
    print(f" - {p}")
print("-----------------------------------------------------")


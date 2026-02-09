
# Monkeypatch imghdr for Python 3.13+ compatibility
import sys
import types
if sys.version_info >= (3, 11):
    try:
        import imghdr
    except ImportError:
        # Create a dummy module
        imghdr = types.ModuleType('imghdr')
        # Mock 'what' function which is commonly used
        def what(file, h=None):
            return None 
        imghdr.what = what
        sys.modules['imghdr'] = imghdr

from azureml.core import Workspace, Experiment, Run
import os

# Connect to Workspace
try:
    ws = Workspace.from_config()
    print(f"[*] Connected to Workspace: {ws.name}")
except Exception as e:
    print(f"[!] Could not connect to Workspace: {e}")
    sys.exit(1)

# Get the Run
run_id = "battery-anomaly-detection_1770613923_8b109e57"
experiment_name = "battery-anomaly-detection"
exp = Experiment(ws, experiment_name)
run = Run(exp, run_id)

print(f"[*] Run Status: {run.get_status()}")

details = run.get_details()
if 'error' in details:
    print(f"[!] Run Error Details: {details['error']}")

# List log files
print("[*] Log Files:")
file_names = run.get_file_names()
for f in file_names:
    print(f" - {f}")

# Find logs
for target_log in ['std_log.txt', '70_driver_log.txt', '20_image_build_log.txt']:
    found_log = None
    for f in file_names:
        if target_log in f:
            found_log = f
            break
            
    if found_log:
        print(f"\n[*] Content of {found_log} (Last 50 lines):")
        try:
           # Download temporarily and read
            run.download_file(found_log, output_file_path=f"temp_{target_log.replace('/', '_')}")
            with open(f"temp_{target_log.replace('/', '_')}", 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print("".join(lines[-50:])) # Print last 50 lines
            
            # Remove temp file
            os.remove(f"temp_{target_log.replace('/', '_')}")
            
        except Exception as e:
            print(f"[!] Error reading {found_log}: {e}")

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import sys
import os

def download_logs(job_name):
    try:
        ml_client = MLClient(
            DefaultAzureCredential(), 
            "b850d62a-25fe-4d3a-9697-ea40449528a9", 
            "8ai-final-team6", 
            "vision"
        )
        
        output_path = f"logs_{job_name}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        print(f"Downloading logs for job: {job_name}...")
        ml_client.jobs.download(name=job_name, download_path=output_path, all=True)
        print(f"Log download completed to: {os.path.abspath(output_path)}")
        
        # Look for the main error log
        # Usually in named-outputs/system_logs or user_logs
        log_found = False
        for root, dirs, files in os.walk(output_path):
            for file in files:
                if file == "std_log.txt" or file == "user_logs.txt":
                    log_path = os.path.join(root, file)
                    print(f"\n--- Reading log: {file} ---")
                    with open(log_path, 'r') as log_f:
                        lines = log_f.readlines()
                        # Show last 50 lines for the error
                        for line in lines[-100:]:
                            print(line.strip())
                    log_found = True
        
        if not log_found:
            print("No std_log.txt found. Checking other log files...")
            # Try to print any found .txt files in user_logs
            for root, dirs, files in os.walk(output_path):
                if "user_logs" in root:
                    for file in files:
                        if file.endswith(".txt") or file.endswith(".log"):
                            print(f"\n--- Reading log: {file} ---")
                            with open(os.path.join(root, file), 'r') as f:
                                print(f.read()[-2000:]) # Last 2KB

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_logs("khaki_prune_8rvvgmxcvs")

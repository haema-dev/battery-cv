from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os

def analyze_recent_jobs():
    try:
        ml_client = MLClient(
            DefaultAzureCredential(), 
            "b850d62a-25fe-4d3a-9697-ea40449528a9", 
            "8ai-final-team6", 
            "vision"
        )
        
        jobs = list(ml_client.jobs.list(max_results=5))
        print("-" * 60)
        print(f"{'Job Name':<25} | {'Display Name':<20} | {'Status':<10}")
        print("-" * 60)
        
        for job in jobs:
            print(f"{job.name:<25} | {str(job.display_name)[:20]:<20} | {job.status:<10}")
            
            # If failed, try to pinpoint why
            if job.status == "Failed":
                print(f"  > Analyzing failure for {job.name}...")
                log_dir = f"logs_{job.name}"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                try:
                    # Download only the necessary logs for analysis
                    ml_client.jobs.download(name=job.name, download_path=log_dir, all=False)
                    # Search for std_log.txt or user_logs in downloaded files
                    for root, dirs, files in os.walk(log_dir):
                        for file in files:
                            if "std_log.txt" in file or "user_logs" in file:
                                with open(os.path.join(root, file), 'r') as f:
                                    print(f"  > Log Snippet ({file}):")
                                    lines = f.readlines()
                                    for line in lines[-15:]: # Last 15 lines usually contain the error
                                        print(f"    {line.strip()}")
                except Exception as ex:
                    print(f"  > Couldn't download logs: {ex}")
            print("-" * 60)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_recent_jobs()

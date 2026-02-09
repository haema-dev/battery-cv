from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import sys

try:
    ml_client = MLClient(
        DefaultAzureCredential(), 
        "b850d62a-25fe-4d3a-9697-ea40449528a9", 
        "8ai-final-team6", 
        "vision"
    )
    jobs = list(ml_client.jobs.list(max_results=5))
    
    print("-" * 50)
    print(f"{'Job Name':<25} | {'Status':<12}")
    print("-" * 50)
    
    for job in jobs:
        name = job.display_name or job.name
        status = job.status
        url = job.services.get("Studio").endpoint if job.services else "N/A"
        print(f"{name:<25} | {status:<12}")
        print(f"URL: {url}")
        print("-" * 50)
        
except Exception as e:
    print(f"Error occurred: {e}")
    sys.exit(1)

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import json

try:
    ml_client = MLClient(
        DefaultAzureCredential(), 
        "b850d62a-25fe-4d3a-9697-ea40449528a9", 
        "8ai-final-team6", 
        "vision"
    )
    # Get the latest job
    jobs = list(ml_client.jobs.list(max_results=1))
    if not jobs:
        result = {"error": "No jobs found"}
    else:
        job = jobs[0]
        result = {
            "name": job.display_name or job.name,
            "status": job.status,
            "url": job.services.get("Studio").endpoint if job.services else "N/A"
        }
except Exception as e:
    result = {"error": str(e)}

with open("job_status.json", "w") as f:
    json.dump(result, f, indent=4)

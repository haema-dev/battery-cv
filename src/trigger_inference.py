
import os
import sys
from azure.ai.ml import MLClient, command, Input, Output
from azure.identity import DefaultAzureCredential

def submit_job():
    print("🚀 [Smart Trigger] Starting Phase 2 Inference Job Submission via Python SDK...")
    
    # 로그인 정보 가져오기 (secrets 환경변수 활용)
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
    workspace_name = os.environ.get("AZURE_WORKSPACE")

    if not all([subscription_id, resource_group, workspace_name]):
        print("❌ Error: Missing Azure environment variables.")
        sys.exit(1)

    # MLClient 초기화
    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace_name
    )

    # 1. 태스크 정의 (inference-job.yml의 파이썬 버전 서비스)
    job = command(
        code="./src",
        command="python inference.py --data_path ${{inputs.data}} --model_path ${{inputs.model}} --output_dir ${{outputs.inference_results}}",
        inputs={
            "data": Input(
                type="uri_folder", 
                path="azureml:battery_data_256_256@latest",
                mode="ro_mount"
            ),
            "model": Input(
                type="uri_file",
                # Phase 1의 성공적인 결과물을 SDK로 직접 연결 (URI 검증 우회)
                path="azureml://jobs/modest_foot_4b2xj4ntn6/outputs/outputs/model.pt",
                mode="download"
            )
        },
        outputs={
            "inference_results": Output(type="uri_folder", mode="rw_mount")
        },
        compute="gpu-advanced",
        environment="azureml:env-yolo@latest",
        experiment_name="Battery_S1_AnomalyDetection",
        display_name="S1_Phase2_Heatmap_Generation_SDK"
    )

    # 2. 작업 제출
    print("⏳ Submitting job to Azure ML...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"✅ Job submitted successfully! Job URL: {returned_job.services['Studio'].endpoint}")

if __name__ == "__main__":
    submit_job()

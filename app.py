import boto3
import os
import uuid
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Read AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("❌ AWS credentials are missing.")

# Initialize AWS Rekognition Client
rekognition_client = boto3.client(
    "rekognition",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)


@app.get("/")
def home():
    return {"message": "Liveliness Detection API is Running!"}


@app.post("/start_liveness_session/")
async def start_liveness_session():
    try:
        # Step 1: Create a Liveness Session
        response = rekognition_client.create_face_liveness_session(
            Settings={
                "OutputConfig": {
                    "S3Bucket": "smartkyc",
                }
            }
        )

        session_id = response["SessionId"]
        print(f"✅ Liveliness Session Started: {session_id}")

        return {"session_id": session_id}

    except Exception as e:
        print(f"🔥 AWS Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")


@app.get("/get_liveness_result/")
async def get_liveness_result(session_id: str):
    try:
        # Step 2: Get Liveness Detection Results
        result = rekognition_client.get_face_liveness_session_results(
            SessionId=session_id
        )

        confidence = result["Confidence"]
        is_live = confidence > 85  # Threshold for liveliness detection

        return {
            "session_id": session_id,
            "liveliness_detected": is_live,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")

import boto3
import os
from fastapi import FastAPI, File, UploadFile
import time

app = FastAPI()

# Read AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError(
        "❌ AWS credentials are missing. Set them in Render Environment Variables!")

# Initialize Rekognition Client
rekognition_client = boto3.client(
    "rekognition",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)


@app.get("/")
def home():
    return {"message": "Liveliness Detection API is Running!"}


@app.post("/detect_liveliness/")
async def detect_liveliness(video: UploadFile = File(...)):
    try:
        video_bytes = await video.read()

        # Call AWS Rekognition Liveliness Detection API
        response = rekognition_client.start_face_liveness_session(
            Video={'Bytes': video_bytes}
        )

        session_id = response["SessionId"]
        print(f"✅ Liveliness Session Started: {session_id}")

        # Wait for results
        time.sleep(10)  # AWS takes time to process

        # Get results
        result = rekognition_client.get_face_liveness_session_results(
            SessionId=session_id
        )

        is_live = result["Confidence"] > 85  # Confidence threshold

        return {
            "liveliness_detected": is_live,
            "confidence": result["Confidence"]
        }

    except Exception as e:
        return {"error": str(e)}

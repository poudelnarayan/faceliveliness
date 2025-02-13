import boto3
import os
import time
import uuid
from fastapi import FastAPI, File, UploadFile

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
        # Generate a unique session ID
        session_id = str(uuid.uuid4())

        # Create a liveliness session
        response = rekognition_client.create_face_liveness_session(
            ClientRequestToken=session_id
        )

        session_id = response["SessionId"]
        print(f"✅ Liveliness Session Started: {session_id}")

        # Wait for AWS to process the session
        time.sleep(10)

        # Get the session results
        result = rekognition_client.get_face_liveness_session_results(
            SessionId=session_id
        )

        is_live = result.get("Confidence", 0) > 85  # Confidence threshold

        return {
            "session_id": session_id,
            "liveliness_detected": is_live,
            "confidence": result.get("Confidence", 0)
        }

    except Exception as e:
        return {"error": str(e)}

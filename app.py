import boto3
import os
import time
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
import asyncio

app = FastAPI()

# Read AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")  # Set this in Render

if not AWS_ACCESS_KEY or not AWS_SECRET_KEY or not S3_BUCKET_NAME:
    raise ValueError("❌ AWS credentials or S3 bucket name are missing.")

# Initialize AWS Clients
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

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
        print("📹 Video Received")
        # Generate a unique filename for the uploaded video
        video_filename = f"liveness_videos/{uuid.uuid4()}.mp4"
        print(f"📹 Uploading video to S3: {video_filename}")

        # Read file content asynchronously and upload to S3
        file_content = await video.read()

        # Upload the video to S3
        try:
            s3_client.put_object(
                Body=file_content,
                Bucket=S3_BUCKET_NAME,
                Key=video_filename,
            )
            print(f"✅ Video uploaded to S3: {video_filename}")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"❌ S3 Upload Error: {str(e)}")

        # Start the face liveliness session using the S3 URL
        response = rekognition_client.create_face_liveness_session(
            Video={'S3Object': {'Bucket': S3_BUCKET_NAME, 'Name': video_filename}}
        )

        session_id = response["SessionId"]
        print(f"✅ Liveliness Session Started: {session_id}")

        # Wait for Rekognition to process the video asynchronously (increase the time if needed)

        # Get liveliness results
        result = await rekognition_client.get_face_liveness_session_results(
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

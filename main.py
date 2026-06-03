# clipforge-backend
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import groq
import subprocess
import os
import json
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/")
def root():
    return {"status": "ClipForge backend running"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    tmp_dir = f"/tmp/{job_id}"
    os.makedirs(tmp_dir, exist_ok=True)

    # Save uploaded video
    video_path = f"{tmp_dir}/input.mp4"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # Extract audio with FFmpeg
    audio_path = f"{tmp_dir}/audio.mp3"
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-q:a", "0", "-map", "a",
        audio_path, "-y"
    ], check=True)

    # Transcribe with Groq Whisper
    with open(audio_path, "rb") as audio_file:
        transcription = groq_client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    transcript_text = transcription.text
    segments = transcription.segments

    # Use Groq LLaMA to find best clips
    prompt = f"""You are a viral content expert. Analyze this video transcript and find the 4 best moments to clip for TikTok/Reels/Shorts.

Transcript:
{transcript_text}

Segments with timestamps:
{json.dumps(segments, indent=2)}

Return ONLY a JSON array with exactly 4 clips. Each clip must have:
- title: catchy name for the clip
- start: start time in seconds (number)
- end: end time in seconds (number)
- score: viral potential score 1-100 (number)
- reason: why this clip will perform well
- transcript: the exact words spoken in this clip

Return only the JSON array, no other text."""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    clips_raw = response.choices[0].message.content.strip()
    clips_raw = clips_raw.replace("```json", "").replace("```", "").strip()
    clips = json.loads(clips_raw)

    # Cut clips with FFmpeg
    output_clips = []
    for i, clip in enumerate(clips):
        out_path = f"{tmp_dir}/clip_{i}.mp4"
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-ss", str(clip["start"]),
            "-to", str(clip["end"]),
            "-c:v", "libx264", "-c:a", "aac",
            out_path, "-y"
        ], check=True)
        clip["file"] = f"/download/{job_id}/clip_{i}.mp4"
        output_clips.append(clip)

    return {"job_id": job_id, "clips": output_clips}

@app.get("/download/{job_id}/{filename}")
def download_clip(job_id: str, filename: str):
    path = f"/tmp/{job_id}/{filename}"
    return FileResponse(path, media_type="video/mp4", filename=filename)

# 🎬 FullyAutoYoutubeVideoCreator

**aividcreator** is an AI-powered video generation tool that takes creative prompts and generates narrated videos scene-by-scene using Google's Gemini API and Veo video models.

---

## 💡 What It Does

- ✏️ Input a creative idea (e.g., *"What if the sun froze?"*)
- 🤖 Generates a short script with multiple scenes (narration + visual prompt)
- 🎥 Sends each scene to Gemini Veo to generate a video clip
- 🪄 Automatically stitches all clips into a final video, organized in timestamped folders

---

## 🧠 Features

- Scene-by-scene narration and visual prompt generation using GPT-4
- Sequential clip creation with progress feedback
- Final automatic stitching of all clips into one video
- Output folders are timestamped for easy organization

---

## 📁 Project Structure

- `main.py` – Main application code
- `output_videos/` – All generated video folders
- `generated_script.json` – Last generated script

---

## 🚀 How to Use

1. **Run** the Python script (`main.py`).
2. **Enter** your creative idea in the app.
3. **Click** "Send to Gemini" to generate a script.
4. **Click** "Send Clips" to generate clips one-by-one.
5. The final stitched video will appear in  
   `output_videos/<project_name_timestamp>/`.

---

## ⚠️ Project Status: On Hold

> **Note:**  
> This project is currently paused because the public Gemini Veo API does **not yet support seamless image-to-video first frame continuation** (as seen in Google’s Flow demo).  
> 
> The goal: **fully automated video creation**—writing scripts with GPT and generating smooth, frame-chained video scenes with Gemini—but this requires an API update for true frame-to-frame chaining.
>
> **Development will continue when Google enables this feature in the public API.**

---

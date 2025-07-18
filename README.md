# aividcreator

**aividcreator** is an AI-powered video generation tool that takes creative prompts and generates narrated videos scene-by-scene using Google's Gemini API and Veo video models.

## 💡 What It Does
- You input a creative idea (e.g., "What if the sun froze?").
- It generates a short script with multiple scenes (narration + visual prompt).
- Each scene is sent to Gemini (Veo) to generate a video clip.
- Once all clips are ready, it stitches them together into a final video.

## 🧠 Features
- Scene-by-scene narration and visual prompts
- Sequential clip generation with feedback
- Final automatic stitching of clips into one video
- Output folders timestamped for organization

## 📁 Project Structure
## 🚀 How to Use
1. Run the Python script (`main.py`).
2. Enter a creative idea into the app.
3. Click "Send to Gemini" to generate a script.
4. Click "Send Clips" to generate clips one-by-one.
5. Final stitched video will appear in `output_videos/<project_name_timestamp>/`.

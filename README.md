aividcreator

aividcreator is an AI-powered video generation tool that takes creative prompts and generates narrated videos scene-by-scene using Google’s Gemini API and Veo video models.

💡 What It Does
	•	You input a creative idea (e.g., “What if the sun froze?”).
	•	The app generates a short script with multiple scenes (each with narration + a visual prompt).
	•	Each scene is sent to Gemini Veo to generate a video clip.
	•	All clips are automatically stitched into a final video, organized by timestamped output folders.

🧠 Features
	•	Scene-by-scene narration and visual prompt generation using GPT-4
	•	Sequential clip creation and feedback
	•	Automatic stitching of all clips into one final video
	•	Output folders are timestamped for easy organization

🚀 How to Use
	1.	Run the Python script (main.py).
	2.	Enter your creative idea in the app.
	3.	Click “Send to Gemini” to generate a script.
	4.	Click “Send Clips” to generate clips one-by-one.
	5.	The final stitched video will appear in output_videos/<project_name_timestamp>/.

⸻

⚠️ Project Status: On Hold

Note:
The project is currently paused because the public Gemini Veo API does not yet support seamless image-to-video first frame continuation (as seen in Google’s Flow demo). The goal is to achieve fully automated video creation—writing scripts with GPT and generating smooth, frame-chained video scenes with Gemini—but this requires an API update for true frame-to-frame chaining.

Once Google enables this in the API, development will continue to support full, seamless transitions between generated video scenes.

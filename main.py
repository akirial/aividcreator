from openai import OpenAI
import os
import json
import datetime
import json as _json
from typing import List
from google import genai
from google.genai import types
import tkinter as tk
from tkinter import ttk, messagebox
import requests
from PIL import Image, ImageTk
import io
from tkinter import PhotoImage
import subprocess
import threading
import re

# Rate limit configuration
MINUTE_CLIP_LIMIT = 2
DAILY_CLIP_LIMIT = 50
RETRY_INTERVAL = 60  # seconds

# Usage tracking file
USAGE_FILE = "clip_usage.json"

# Initialize in-memory list for recent call timestamps
recent_calls = []


def load_usage():
    today = datetime.date.today().isoformat()
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, "r") as f:
            data = _json.load(f)
        if data.get("date") == today:
            return data
    return {"date": today, "daily_count": 0}


def save_usage(data):
    with open(USAGE_FILE, "w") as f:
        _json.dump(data, f)


# Configuration
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
openai = OpenAI(api_key=api_key)

# Google Gemini API key configuration
genai_api_key = os.getenv("GOOGLE_API_KEY")
if not genai_api_key:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

# Constants
COST_PER_CLIP = 20  # credits
CREDITS_PER_DOLLAR = 100  # 2500 credits = $25
SECONDS_PER_CLIP = 8
DEFAULT_VIDEO_DURATION = 16
NARRATOR_DESCRIPTION = "35-year-old man, calm and curious tone"


def calculate_cost(duration_seconds: int) -> tuple:
    num_clips = duration_seconds // SECONDS_PER_CLIP
    total_credits = num_clips * COST_PER_CLIP
    total_cost = total_credits / CREDITS_PER_DOLLAR
    return num_clips, total_credits, total_cost


def generate_script_scenes(idea: str, duration_seconds: int, send_to_api: bool, special_request: str) -> List[dict]:
    if not send_to_api:
        print("[DEBUG] Skipping OpenAI API call (send_to_api=False)")
        return []

    scene_count = duration_seconds // SECONDS_PER_CLIP
    prompt = f"""
Create a YouTube video script based on the idea: "{idea}".
The total duration is about {duration_seconds} seconds, divided into {scene_count} scenes, each roughly {SECONDS_PER_CLIP} seconds long.
Each scene should include:
- Narration (1-2 short sentences)
- A vivid visual description (for a video generation model like Veo 3)
- Narrator description: {NARRATOR_DESCRIPTION}

Format the output as JSON with fields: scene, narration, narrator, visual_prompt.
Start with an engaging hook, build curiosity, and deliver on it.
"""
    if special_request:
        prompt += f"\n\nSpecial instructions: {special_request}"

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9
    )

    try:
        content = response.choices[0].message.content
        print("[DEBUG] Full response content:")
        print(content)
        # Attempt to fix invalid JSON if multiple objects/fragments are returned
        if content.strip().startswith("{") and "},\n{" in content:
            content = "[" + content.replace("},\n{", "},\n{") + "]"
        elif content.strip().startswith("{") and content.strip().endswith("}") and content.count("}\n{") > 0:
            content = "[" + content.replace("}\n{", "},\n{") + "]"
        elif content.strip().startswith("{") and content.strip().endswith("}") and content.count("},") > 0:
            content = "[" + content + "]"
        data = json.loads(content)

        # Normalize structure
        if isinstance(data, dict):
            if "scenarios" in data:
                data = data["scenarios"]
            elif "scenes" in data and isinstance(data["scenes"], list):
                data = data["scenes"]
            else:
                # Wrap dict values if it looks like {"scene_1": {...}, "scene_2": {...}}
                if all(isinstance(v, dict) for v in data.values()):
                    data = list(data.values())
                else:
                    data = [data]
        elif isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            # Sometimes a single list with a nested object that contains scenes
            nested = data[0]
            if all(isinstance(v, dict) for v in nested.values()):
                data = list(nested.values())

        return data
    except Exception as e:
        error_msg = f"Error parsing script response: {e}\nRaw response content:\n{content}"
        print(error_msg)
        return error_msg


def download_video(url, path, progress_callback=None):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            downloaded = 0
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(8192, downloaded, total)
            return True
    except Exception as e:
        print(f"Download error: {e}")
        return False


def generate_veo_clip(scene: dict, output_dir: str, send_to_api: bool, progress=None, veo_version: str = "veo3") -> str:
    # Normalize scene fields to handle different key formats
    narration = scene.get('narration') or scene.get('Narration') or ""
    narrator = scene.get('narrator') or scene.get('Narrator') or ""
    visual_prompt = (
            scene.get('visual_prompt')
            or scene.get('Visual_prompt')
            or scene.get('visualPrompt')
            or scene.get('visualPrompt')
            or ""
    )
    # Ensure scene number exists
    if 'scene' not in scene:
        scene['scene'] = ""

    if not send_to_api:
        print(f"[DEBUG] Skipping Veo API call for scene {scene['scene']} (send_to_api=False)")
        return ""

    # Load usage and enforce daily limit
    usage = load_usage()
    if usage["daily_count"] >= DAILY_CLIP_LIMIT:
        print(f"[ERROR] Daily clip limit of {DAILY_CLIP_LIMIT} reached. Try again tomorrow.")
        return ""

    # Choose model and prompt based on version
    if veo_version == "veo2":
        model_name = "models/veo-2.0-generate-001"
        prompt = (
            f"\nGenerate an {SECONDS_PER_CLIP}-second cinematic video clip.\n\n"
            f"Visuals: {visual_prompt}\n"
            "No text overlay. No text on screen.\n"
        )
    else:
        model_name = "models/veo-3.0-generate-preview"
        prompt = (
            f"\nGenerate an {SECONDS_PER_CLIP}-second cinematic video clip.\n\n"
            f"Narration: \"{narration}\"\n"
            f"Voice: {narrator}\n\n"
            f"Visuals: {visual_prompt}\n"
            "Subtitles: none\nNo text overlay. No text on screen.\n"
        )
    import time
    while True:
        # Enforce minute rate limit
        now = time.time()
        # Remove timestamps older than 60 seconds
        while recent_calls and now - recent_calls[0] > 60:
            recent_calls.pop(0)
        if len(recent_calls) >= MINUTE_CLIP_LIMIT:
            wait = 60 - (now - recent_calls[0])
            print(f"[WARNING] Minute rate limit reached. Waiting {int(wait) + 1} seconds...")
            time.sleep(wait + 1)
            continue
        try:
            client = genai.Client(api_key=genai_api_key)

            # Only add image argument if a previous frame exists and can be loaded as bytes
            prev_frame_path = scene.get('prev_frame_image')
            image_arg = None
            if prev_frame_path and os.path.exists(prev_frame_path):
                try:
                    with open(prev_frame_path, "rb") as f:
                        image_arg = types.Image.from_bytes(f.read(), mime_type="image/png")
                    print(f"[DEBUG] Loaded prev_frame_image and passing as context: {prev_frame_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to load prev_frame_image {prev_frame_path}: {e}")

            # Build kwargs only if image_arg is not None
            if image_arg is not None:
                operation = client.models.generate_videos(
                    model=model_name,
                    prompt=prompt,
                    image=image_arg,
                    config=types.GenerateVideosConfig(
                        person_generation="allow_adult",
                        aspect_ratio="16:9",
                    ),
                )
            else:
                operation = client.models.generate_videos(
                    model=model_name,
                    prompt=prompt,
                    config=types.GenerateVideosConfig(
                        person_generation="allow_adult",
                        aspect_ratio="16:9",
                    ),
                )

            while not operation.done:
                time.sleep(5)
                operation = client.operations.get(operation)

            if operation.response and getattr(operation.response, "generated_videos", None):
                generated = operation.response.generated_videos[0]
                client.files.download(file=generated.video)
                video_path = os.path.join(output_dir, f"scene_{scene['scene']}.mp4")
                generated.video.save(video_path)
                print(f"[DEBUG] Downloaded to {video_path}")
                # Save last frame of this clip for inspection (cv2 fallback to ffmpeg)
                last_frame_png = os.path.splitext(video_path)[0] + ".png"
                try:
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                    ret, frame = cap.read()
                    cap.release()
                    if ret and cv2.imwrite(last_frame_png, frame):
                        print(f"[DEBUG] Saved last frame image via cv2 at: {last_frame_png}")
                    else:
                        print(f"[DEBUG] cv2 failed to extract last frame for {video_path}")
                except ImportError:
                    # Fallback to ffmpeg if cv2 is unavailable
                    try:
                        import subprocess
                        cmd = ["ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path, "-vframes", "1", last_frame_png]
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        print(f"[DEBUG] Extracted last frame via ffmpeg at: {last_frame_png}")
                    except Exception as ff_err:
                        print(f"[DEBUG] Fallback ffmpeg frame extraction failed: {ff_err}")
                except Exception as e:
                    print(f"[DEBUG] Exception extracting last frame via cv2: {e}")
                # Track usage
                recent_calls.append(time.time())
                usage["daily_count"] += 1
                save_usage(usage)
                return video_path
            else:
                print(f"[ERROR] Veo API returned no video for scene {scene['scene']}. Full operation: {operation}")
                if hasattr(operation, "error"):
                    print(f"[ERROR] Veo operation error: {operation.error}")
                return ""

        except Exception as e:
            error_str = str(e)
            if "FAILED_PRECONDITION" in error_str:
                billing_msg = ("Veo model requires Google Cloud billing to be enabled. "
                               "Please enable billing in your GCP project to use this model.")
                print(f"[ERROR] {billing_msg}")
                return ""
            elif "RESOURCE_EXHASED" in error_str or "quota" in error_str.lower():
                print(f"[ERROR] Quota/rate limit error: {e}. Retrying in {RETRY_INTERVAL} seconds...")
                for remaining in range(RETRY_INTERVAL, 0, -1):
                    print(f"[INFO] Retrying in {remaining} seconds...", end="\r")
                    time.sleep(1)
                print()
                continue
            else:
                print(f"[ERROR] Error generating scene {scene['scene']}: {e}")
                return ""


def save_script_to_file(scenes: List[dict], filename: str):
    with open(filename, "w") as f:
        json.dump(scenes, f, indent=2)
    print(f"Saved script to {filename}")


# GUI Functions
class VideoApp:
    def get_output_directory(self):
        import datetime
        import re
        project_raw = self.idea_entry.get().strip()
        project_name = re.sub(r'[^A-Za-z0-9_-]', '_', project_raw) or "project"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = "output_videos"
        output_dir = os.path.join(base_dir, f"{project_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def send_clips_one_by_one(self, scenes):
        output_dir = self.get_output_directory()
        self.generated_clips = []  # Track scene files
        for idx, scene in enumerate(scenes):
            scene_label = f"{scene.get('scene', idx + 1)}"
            self.log_debug(f"[SEND] Sending scene {scene_label} to Gemini...")
            self.status_output.config(text=f"Sending scene {scene_label} to Gemini...")
            self.root.update()
            self.generate_video([scene], output_dir=output_dir)
            self.log_debug(f"[SEND] Finished sending scene {scene_label}.")
        # Stitch after all
        self.stitch_videos(output_dir, scenes)

    def __init__(self, root):
        self.root = root
        self.root.title("AI YouTube Video Generator")

        self.last_send_time = 0
        self.progress_bars = {}

        # Gemini Quota label at top
        self.gemini_quota_label = ttk.Label(root, text="Gemini Quota: ...")
        self.gemini_quota_label.pack(anchor="nw", padx=10, pady=(5, 0))
        self.refresh_gemini_quota()

        self.notebook = ttk.Notebook(root)
        self.script_frame = ttk.Frame(self.notebook)
        self.video_frame = ttk.Frame(self.notebook)
        self.debug_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.script_frame, text="Generate Script")
        self.notebook.add(self.video_frame, text="Generate Video")
        self.notebook.add(self.debug_frame, text="Debug Log")
        self.notebook.pack(fill="both", expand=True)

        self.debug_button = ttk.Button(root, text="View Debug Log",
                                       command=lambda: self.notebook.select(self.debug_frame))
        self.debug_button.pack(anchor="ne", padx=10, pady=5)

        # Chain clips option (Veo2 only) -- for video generation UI
        self.chain_clips_var = tk.BooleanVar(value=True)
        self.chain_check = ttk.Checkbutton(
            self.video_frame,
            text="Chain clips (Veo2 only)",
            variable=self.chain_clips_var
        )
        # Place under the Veo version radios
        self.chain_check.pack(anchor="w", padx=10, pady=5)

        self.idea_label = ttk.Label(self.script_frame, text="Video Idea:")
        self.idea_label.pack()
        self.idea_entry = ttk.Entry(self.script_frame, width=50)
        self.idea_entry.insert(0, "How small actually are we?")
        self.idea_entry.pack()

        self.length_label = ttk.Label(self.script_frame, text="Video Length (seconds):")
        self.length_label.pack()
        self.length_spinbox = ttk.Spinbox(self.script_frame, from_=8, to=600, increment=8, width=10)
        self.length_spinbox.set(8)
        self.length_spinbox.pack()

        self.cost_label = ttk.Label(self.script_frame, text="Cost Estimate:")
        self.cost_label.pack()
        self.cost_output = ttk.Label(self.script_frame, text="")
        self.cost_output.pack()

        self.minutes_output = ttk.Label(self.script_frame, text="")
        self.minutes_output.pack()

        self.send_to_api_var = tk.BooleanVar()
        self.send_to_api_check = ttk.Checkbutton(self.script_frame, text="Enable API Calls",
                                                 variable=self.send_to_api_var)
        self.send_to_api_check.pack()

        self.special_label = ttk.Label(self.script_frame, text="Special Script Requests:")
        self.special_label.pack()
        self.special_entry = tk.Text(self.script_frame, height=4, wrap="word")
        self.special_entry.pack(fill="x", padx=10)

        self.generate_script_btn = ttk.Button(self.script_frame, text="Generate Script", command=self.generate_script)
        self.generate_script_btn.pack(pady=10)

        self.status_output = ttk.Label(self.script_frame, text="")
        self.status_output.pack()

        self.script_display_label = ttk.Label(self.script_frame, text="Generated Script:")
        self.script_display_label.pack()

        self.script_output = tk.Text(self.script_frame, height=10, wrap="word")
        self.script_output.pack(fill="x", padx=10)

        self.debug_output = tk.Text(self.debug_frame, height=30, wrap="word")
        self.debug_output.pack(fill="both", expand=True, padx=10, pady=10)

        self.error_output = ttk.Label(self.script_frame, text="", foreground="red")
        self.error_output.pack()

        self.length_spinbox.bind("<KeyRelease>", self.update_cost)
        self.length_spinbox.bind("<FocusOut>", self.update_cost)
        self.update_cost()

        self.send_video_label = ttk.Label(self.video_frame, text="Send to Gemini for Video Generation")
        self.send_video_label.pack(pady=5)

        # Veo version selection
        self.veo_version_var = tk.StringVar(value="veo2")
        veo2_radio = ttk.Radiobutton(self.video_frame, text="Veo 2", variable=self.veo_version_var, value="veo2")
        veo3_radio = ttk.Radiobutton(self.video_frame, text="Veo 3", variable=self.veo_version_var, value="veo3")
        veo2_radio.pack()
        veo3_radio.pack()

        self.clip_count_var = tk.IntVar(value=1)
        self.clip_option_1 = ttk.Radiobutton(self.video_frame, text="1 Clip (8 sec)", variable=self.clip_count_var,
                                             value=1)
        self.clip_option_2 = ttk.Radiobutton(self.video_frame, text="2 Clips (16 sec)", variable=self.clip_count_var,
                                             value=2)
        self.clip_option_3 = ttk.Radiobutton(self.video_frame, text="3 Clips (24 sec)", variable=self.clip_count_var,
                                             value=3)
        self.clip_option_1.pack()
        self.clip_option_2.pack()
        self.clip_option_3.pack()

        self.send_video_button = ttk.Button(self.video_frame, text="Send to Gemini",
                                            command=self.confirm_send_to_gemini)
        self.send_video_button.pack(pady=10)
        self.send_video_button["state"] = "disabled"

        # New: Canvas with horizontal scrollbar for clips
        self.clips_canvas = tk.Canvas(self.video_frame, height=200)
        self.clips_scrollbar = ttk.Scrollbar(self.video_frame, orient="horizontal", command=self.clips_canvas.xview)
        self.clips_canvas.configure(xscrollcommand=self.clips_scrollbar.set)
        self.clips_scrollbar.pack(side="bottom", fill="x")
        self.clips_canvas.pack(fill="x", expand=False)

        self.clips_frame = ttk.Frame(self.clips_canvas)
        self.clips_canvas.create_window((0, 0), window=self.clips_frame, anchor="nw")

        def on_frame_configure(event):
            self.clips_canvas.configure(scrollregion=self.clips_canvas.bbox("all"))

        self.clips_frame.bind("<Configure>", on_frame_configure)

        self.video_display_frame = ttk.Frame(self.video_frame)
        self.video_display_frame.pack()

        # Gemini model listing for debug purposes
        try:
            client = genai.Client(api_key=genai_api_key)
            models = client.models.list()
            print("[DEBUG] Available Gemini Models:")
            for model in models:
                print(f"- {model.name}: {getattr(model, 'supported_generation_methods', 'N/A')}")
        except Exception as e:
            print(f"[ERROR] Failed to list Gemini models: {e}")

    # ... rest of methods remain unchanged ...



    def log_debug(self, message):
        # Prepend timestamp in [YYYY-MM-DD HH:MM:SS] format
        timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        self.debug_output.insert("end", f"{timestamp} {message}\n")
        self.debug_output.see("end")

    def update_cost(self, *_):
        try:
            duration = int(self.length_spinbox.get())
            clips, credits, cost = calculate_cost(duration)
            self.cost_output.config(text=f"{clips} clips · {credits} credits · ${cost:.2f}")
            minutes = duration / 60
            self.minutes_output.config(text=f"{minutes:.1f} minutes")
        except ValueError:
            self.cost_output.config(text="Invalid input")

    def generate_script(self):
        import time
        self.send_video_button["state"] = "disabled"
        if time.time() - self.last_send_time < 5:
            self.error_output.config(text="Please wait a few seconds before sending again.")
            return
        self.last_send_time = time.time()

        idea = self.idea_entry.get().strip()
        try:
            duration = int(self.length_spinbox.get())
        except ValueError:
            self.error_output.config(text="Please enter a valid number for duration.")
            return

        if not idea:
            self.error_output.config(text="Please enter a video idea.")
            return

        send_to_api = self.send_to_api_var.get()
        if not send_to_api:
            self.error_output.config(text="Please enable API Calls before generating a script.")
            return
        special_request = self.special_entry.get("1.0", "end").strip()

        self.status_output.config(text="Generating script...")
        self.debug_output.delete("1.0", "end")
        self.debug_output.insert("end", "Generating script...\n")
        self.debug_output.see("end")
        self.root.update()

        result = generate_script_scenes(idea, duration, send_to_api, special_request)
        if isinstance(result, str):
            self.status_output.config(text="Script generation failed.")
            self.log_debug(result)
            self.error_output.config(text="Script generation failed. See debug log.")
            return
        # Updated normalization logic for nested 'scenes' data
        scenes = []
        if isinstance(result, list):
            # Handle a list with a single dict containing 'scenes'
            if len(result) == 1 and isinstance(result[0], dict) and "scenes" in result[0]:
                scenes = result[0]["scenes"]
            else:
                scenes = result
        elif isinstance(result, dict):
            if "scenes" in result:
                scenes = result["scenes"]
            else:
                scenes = [result]
        else:
            self.status_output.config(text="Failed to parse scenes.")
            self.log_debug("Unexpected result format for scenes.")
            return

        for i, scene in enumerate(scenes):
            if "scene" not in scene:
                scene["scene"] = f"Scene {i + 1}"

        if scenes:
            save_script_to_file(scenes, "generated_script.json")
            self.status_output.config(text="Script generated and saved to file.")
            script_text = ""
            scenes_list = scenes if isinstance(scenes, list) else [scenes]
            # Build script text
            for scene in scenes_list:
                try:
                    script_text += f"{scene['scene']}:\nNarration: {scene['narration']}\nVisual: {scene['visual_prompt']}\n\n"
                except Exception as e:
                    self.log_debug(f"Skipping scene due to error: {e}")

            self.script_output.delete("1.0", "end")
            self.script_output.insert("end", script_text)
            self.debug_output.insert("end", script_text)
            self.log_debug(f"Generated {len(scenes_list)} scenes.")
            self.error_output.config(text="")
            self.notebook.select(self.video_frame)
            self.send_video_button["state"] = "normal"
            self.log_debug(json.dumps(scenes, indent=2))
            # self.generate_video(scenes)
        else:
            self.status_output.config(text="Failed to generate script.")
            self.log_debug("No scenes generated.")
            self.error_output.config(text="Script generation failed. See debug log.")

    def generate_video(self, scenes, output_dir=None):
        import re
        import os
        # Ensure each scene has a 'scene' key for indexing
        for idx, scene in enumerate(scenes):
            if 'scene' not in scene or not isinstance(scene['scene'], (str, int)):
                scene['scene'] = idx + 1
        # Determine output directory
        if output_dir is None:
            output_dir = self.get_output_directory()
        self.status_output.config(text="Generating video clips...")
        self.root.update()

        for widget in self.video_display_frame.winfo_children():
            widget.destroy()
        for bar in self.progress_bars.values():
            bar.destroy()
        self.progress_bars = {}

        # Add loading spinner (indeterminate progressbar) to status_output
        spinner = ttk.Progressbar(self.status_output.master, mode='indeterminate', length=120)
        spinner.pack(pady=5)
        spinner.start()
        self.root.update()

        # Track threads so we can know when all clips are done
        threads = []
        completed = []

        def on_clip_finished():
            completed.append(1)
            self.status_output.config(text=f"{len(completed)}/{len(scenes)} clips processed and downloaded.")

        # Find the scene_frame for each scene, if possible, in self.clips_frame
        # Build a mapping from scene number to frame
        scene_frame_map = {}
        for child in self.clips_frame.winfo_children():
            # Try to find the label inside the frame to match scene number
            for label in child.winfo_children():
                if isinstance(label, ttk.Label):
                    text = label.cget("text")
                    match = re.match(r"Scene (\d+)", text)
                    if match:
                        scene_num = int(match.group(1))
                        scene_frame_map[scene_num] = child
                        break

        # Provide update_play_button function for use in run_download
        def update_play_button(video_path, container):
            play_btn = ttk.Button(container, text="Play Scene", command=lambda p=video_path: subprocess.Popen(["open", p]))
            play_btn.pack(pady=5)

        # Chaining logic for Veo2 if chain_clips_var is set
        veo_version = self.veo_version_var.get()
        chain_clips = self.chain_clips_var.get()
        if veo_version == "veo2" and chain_clips and len(scenes) > 1:
            # Sequentially generate clips, passing last frame of previous as context for next
            import cv2
            prev_clip_path = None
            prev_frame_image = None
            for idx, scene in enumerate(scenes):
                progress = ttk.Progressbar(self.video_display_frame, length=150, mode='determinate')
                progress.pack(padx=5, pady=5)
                # Extract scene number robustly
                scene_str = str(scene['scene'])
                scene_num_match = re.search(r'\d+', scene_str)
                scene_num = int(scene_num_match.group()) if scene_num_match else idx + 1
                self.progress_bars[scene_num] = progress
                self.root.update()
                # Find the scene_frame for this scene
                scene_frame = scene_frame_map.get(scene_num, None)
                # If not the first scene, capture last frame from previous clip
                if idx > 0 and prev_clip_path and os.path.exists(prev_clip_path):
                    try:
                        cap = cv2.VideoCapture(prev_clip_path)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            # Derive PNG path from the previous clip by swapping .mp4 to .png
                            video_base, _ = os.path.splitext(prev_clip_path)
                            last_frame_path = f"{video_base}.png"
                            # Save frame image next to video file
                            if cv2.imwrite(last_frame_path, frame):
                                scene['prev_frame_image'] = last_frame_path
                                self.log_debug(f"Saved last frame image for previous clip at: {last_frame_path}")
                            else:
                                self.log_debug(f"Failed to write last frame image to: {last_frame_path}")
                        else:
                            self.log_debug(f"Failed to capture last frame from {prev_clip_path}")
                    except Exception as e:
                        self.log_debug(f"Error capturing last frame from {prev_clip_path}: {e}")
                # Generate the clip
                def run_download(scene=scene, progress=progress, scene_frame=scene_frame, idx=idx):
                    video_path = generate_veo_clip(
                        scene,
                        output_dir,
                        self.send_to_api_var.get(),
                        progress,
                        veo_version
                    )
                    scene_str_local = str(scene['scene'])
                    scene_num_match_local = re.search(r'\d+', scene_str_local)
                    scene_num_local = int(scene_num_match_local.group()) if scene_num_match_local else idx + 1
                    if video_path and os.path.exists(video_path):
                        self.log_debug(f"Scene {scene['scene']} video saved at: {video_path}")
                        if scene_frame is not None:
                            update_play_button(video_path, container=scene_frame)
                        self.status_output.config(text=f"Scene {scene['scene']} downloaded.")
                        self.error_output.config(text="")  # Clear previous error text if any
                    else:
                        self.log_debug(f"Scene {scene['scene']} video generation failed. Check model availability and API permissions.")
                        self.status_output.config(text=f"Failed to generate scene {scene['scene']}.")
                        self.error_output.config(text=f"Error with scene {scene['scene']}. See debug log.")
                    on_clip_finished()
                run_download()
                # Set prev_clip_path for next iteration
                prev_clip_path = os.path.join(output_dir, f"scene_{scene['scene']}.mp4")
            # Remove/stop spinner after all clips
            spinner.stop()
            spinner.destroy()
            # All clips are done, stitch videos
            # self.stitch_videos(output_dir, scenes)
            return

        # Default: generate all clips as before (possibly in parallel)
        for idx, scene in enumerate(scenes):
            progress = ttk.Progressbar(self.video_display_frame, length=150, mode='determinate')
            progress.pack(padx=5, pady=5)
            # Extract scene number robustly
            scene_str = str(scene['scene'])
            scene_num_match = re.search(r'\d+', scene_str)
            scene_num = int(scene_num_match.group()) if scene_num_match else idx + 1
            self.progress_bars[scene_num] = progress
            self.root.update()

            # Find the scene_frame for this scene
            scene_frame = scene_frame_map.get(scene_num, None)

            def run_download(scene=scene, progress=progress, scene_frame=scene_frame, idx=idx):
                video_path = generate_veo_clip(
                    scene,
                    output_dir,
                    self.send_to_api_var.get(),
                    progress,
                    self.veo_version_var.get()
                )
                scene_str_local = str(scene['scene'])
                scene_num_match_local = re.search(r'\d+', scene_str_local)
                scene_num_local = int(scene_num_match_local.group()) if scene_num_match_local else idx + 1
                if video_path and os.path.exists(video_path):
                    self.log_debug(f"Scene {scene['scene']} video saved at: {video_path}")
                    # Button creation moved to scene_frame via update_play_button if available
                    if scene_frame is not None:
                        update_play_button(video_path, container=scene_frame)
                    self.status_output.config(text=f"Scene {scene['scene']} downloaded.")
                    self.error_output.config(text="")  # Clear previous error text if any
                else:
                    self.log_debug(f"Scene {scene['scene']} video generation failed. Check model availability and API permissions.")
                    self.status_output.config(text=f"Failed to generate scene {scene['scene']}.")
                    self.error_output.config(text=f"Error with scene {scene['scene']}. See debug log.")
                on_clip_finished()
            t = threading.Thread(target=run_download)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Remove/stop spinner after threads finish
        spinner.stop()
        spinner.destroy()

        # All clips are done, stitch videos
        # self.stitch_videos(output_dir, scenes)

    def stitch_videos(self, output_dir, scenes):
        import subprocess
        import os

        list_file_path = os.path.join(output_dir, "video_list.txt")
        with open(list_file_path, "w") as list_file:
            for scene in scenes:
                scene_filename = f"scene_{scene['scene']}.mp4"
                scene_path = os.path.join(output_dir, scene_filename)
                if os.path.exists(scene_path):
                    list_file.write(f"file '{scene_filename}'\n")

        # The ffmpeg command should be run with cwd=output_dir and file paths relative to that directory
        try:
            subprocess.run(
                ["ffmpeg", "-f", "concat", "-safe", "0", "-i", "video_list.txt", "-c", "copy", "final_video.mp4"],
                check=True,
                cwd=output_dir
            )
            final_output_path = os.path.join(output_dir, "final_video.mp4")
            self.log_debug(f"Final stitched video created at: {final_output_path}")
            # Button to open the final stitched video
            self.open_final_btn = ttk.Button(
                self.video_display_frame,
                text="Open Final Video",
                command=lambda: subprocess.Popen(["open", final_output_path])
            )
            self.open_final_btn.pack(pady=10)
            # Add Auto-Preview button
            self.auto_preview_btn = ttk.Button(
                self.video_display_frame,
                text="Auto-Preview",
                command=lambda: subprocess.Popen(["open", final_output_path])
            )
            self.auto_preview_btn.pack(pady=(0, 10))
            # Disable the "Send All Clips" button once final video is available
            if hasattr(self, "send_all_btn"):
                self.send_all_btn["state"] = "disabled"
            # Export metadata summary after stitching
            self.export_metadata(scenes, output_dir)
        except FileNotFoundError:
            err_msg = "ffmpeg not found. Please install ffmpeg to enable video stitching."
            print(f"[ERROR] {err_msg}")
            self.log_debug(err_msg)
            messagebox.showerror("Missing Dependency", err_msg)
            # Update status label and disable relevant buttons
            self.status_output.config(text=err_msg)
            if hasattr(self, "send_all_btn"):
                self.send_all_btn["state"] = "disabled"
            if hasattr(self, "open_final_btn"):
                self.open_final_btn["state"] = "disabled"
            if hasattr(self, "auto_preview_btn"):
                self.auto_preview_btn["state"] = "disabled"
        except subprocess.CalledProcessError as e:
            self.log_debug(f"Failed to stitch videos: {e}")
            messagebox.showerror("Stitching Error", f"Video stitching failed: {e}")

    def export_metadata(self, scenes, output_dir):
        """
        Write a summary of all scenes into metadata_summary.json in the output directory.
        """
        summary = []
        for scene in scenes:
            summary.append({
                "scene": scene.get("scene"),
                "narration": scene.get("narration"),
                "visual_prompt": scene.get("visual_prompt"),
                "narrator": scene.get("narrator"),
            })
        metadata_path = os.path.join(output_dir, "metadata_summary.json")
        try:
            with open(metadata_path, "w") as f:
                json.dump(summary, f, indent=2)
            self.log_debug(f"Exported scene metadata to {metadata_path}")
        except Exception as e:
            self.log_debug(f"Failed to export metadata: {e}")

    def refresh_gemini_quota(self):
        """
        Fetch and update Gemini quota status every minute.
        """
        import threading
        import time
        def fetch_quota():
            # Dummy implementation: You should replace this with actual Gemini quota API if available.
            # For demo, we show daily and minute limits and usage from load_usage().
            usage = load_usage()
            daily = usage.get("daily_count", 0)
            minute = len(recent_calls)
            quota_text = f"Gemini Quota: {minute}/{MINUTE_CLIP_LIMIT} clips/min, {daily}/{DAILY_CLIP_LIMIT} clips/day"
            self.gemini_quota_label.config(text=quota_text)
        # Fetch in main thread (it's fast)
        fetch_quota()
        # Schedule next update in 60 seconds
        self.root.after(60000, self.refresh_gemini_quota)

    def confirm_send_to_gemini(self):
        clip_count = self.clip_count_var.get()
        if messagebox.askyesno("Confirm", f"Are you sure you want to send {clip_count} clip(s) to Gemini?"):
            self.send_to_gemini(clip_count)

    def send_to_gemini(self, clip_count):
        import time
        import os
        if not os.path.exists("generated_script.json"):
            self.log_debug("Script file not found. Please generate a script first.")
            self.status_output.config(text="Script file not found. Please generate a script first.")
            self.error_output.config(text="Script not ready.")
            return
        if time.time() - self.last_send_time < 5:
            self.log_debug("Please wait a few seconds before sending again.")
            return
        self.last_send_time = time.time()

        self.status_output.config(text="Sending to Gemini...")
        self.log_debug("Started sending clips to Gemini...")
        self.send_video_button["state"] = "disabled"
        self.root.update()
        self.script_output.insert("end", "Sending to Gemini...\n")
        self.script_output.see("end")

        try:
            with open("generated_script.json") as f:
                scenes_data = json.load(f)

            # Normalize to a list of scenes
            if isinstance(scenes_data, dict):
                if "scenes" in scenes_data and isinstance(scenes_data["scenes"], list):
                    scenes = scenes_data["scenes"]
                elif "scene" in scenes_data:
                    scenes = [scenes_data]
                else:
                    scenes = [value for value in scenes_data.values() if isinstance(value, dict)]
            else:
                scenes = scenes_data if isinstance(scenes_data, list) else [scenes_data]

            if not scenes:
                self.log_debug("No script scenes found.")
                self.status_output.config(text="No script scenes found.")
                self.error_output.config(text="No valid script scenes.")
                return

            self.status_output.config(text=f"{len(scenes)} scenes ready for generation.")
            self.error_output.config(text="")
            messagebox.showinfo("Scenes Ready", f"{len(scenes)} scenes are ready to be sent to Gemini.")

            # Populate the clips_frame with scene previews and buttons
            for widget in self.clips_frame.winfo_children():
                widget.destroy()

            def send_single_clip(scene):
                self.generate_video([scene])

            for scene in scenes:
                frame = ttk.Frame(self.clips_frame, width=150, height=150, relief="ridge", borderwidth=2)
                frame.pack_propagate(False)
                frame.pack(side="left", padx=5, pady=5)
                label = ttk.Label(frame, text=f"Scene {scene.get('scene', '?')}")
                label.pack(pady=5)
                btn = ttk.Button(frame, text="Send Clip Individually", command=lambda s=scene: send_single_clip(s))
                btn.pack(pady=5)
                # Insert update_play_button function in this scope for later use by generate_video
                def update_play_button(video_path, container=frame):
                    play_btn = ttk.Button(container, text="Play Scene", command=lambda p=video_path: subprocess.Popen(["open", p]))
                    play_btn.pack(pady=5)

            # Add the "Send All Clips" button (below the canvas)
            if hasattr(self, 'send_all_btn') and self.send_all_btn.winfo_exists():
                self.send_all_btn.destroy()
            import threading
            self.send_all_btn = ttk.Button(
                self.video_frame,
                text="Send All Clips",
                command=lambda: threading.Thread(target=self.send_clips_one_by_one, args=(scenes[:clip_count],)).start()
            )
            self.send_all_btn.pack(pady=10)
        except Exception as e:
            self.log_debug(f"Error loading script: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
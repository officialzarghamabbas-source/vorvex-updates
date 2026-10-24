# ============================================================
# VORVEX — Local Voice Studio Dashboard (Clean UI Edition)
# ============================================================
import os, sys, time, gc, uuid, re, glob
import torch
import torchaudio as ta
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import gradio as gr

# Robust Absolute Path Setup (Fixes model directory lookup from any subfolder)
CURRENT_DIR = "D:/CuteTTS" if os.path.exists("D:/CuteTTS/model/CuteTTS") else os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR.endswith("dist"):
    CURRENT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.insert(0, os.path.join(CURRENT_DIR, "src"))
sys.path.insert(0, CURRENT_DIR)
os.chdir(CURRENT_DIR)

OUT_DIR = os.path.join(CURRENT_DIR, "CuteTTS_Output")
os.makedirs(OUT_DIR, exist_ok=True)

# Custom Saved Clones Storage Directory
CLONE_STORE_DIR = os.path.join(CURRENT_DIR, "Voice_Clones_Registry")
os.makedirs(CLONE_STORE_DIR, exist_ok=True)

print("\n⏳ Loading CuteTTS model (1-2 min)...")
from cutetts import CuteTTS
MODEL_DIR = os.path.join(CURRENT_DIR, "model/CuteTTS")
model = CuteTTS.from_pretrained(MODEL_DIR, device="cuda")
print("✅ Model loaded")


def get_available_clones():
    clones = {"Default VORVEX Voice (Built-in)": None}
    if os.path.exists(CLONE_STORE_DIR):
        for f in sorted(os.listdir(CLONE_STORE_DIR)):
            if f.endswith(".wav"):
                parts = f.rsplit("_", 1)
                if len(parts) == 2:
                    clean_name = parts[0].replace("_", " ").title()
                    clones[f"🧬 {clean_name}"] = os.path.join(CLONE_STORE_DIR, f)
    return clones


def get_clones_list_html():
    if not os.path.exists(CLONE_STORE_DIR):
        return "<p style='color:#777783; font-size:14px;'>No custom voice clones added yet.</p>"
    
    wav_files = [f for f in sorted(os.listdir(CLONE_STORE_DIR)) if f.endswith(".wav")]
    if not wav_files:
        return "<p style='color:#777783; font-size:14px;'>No custom voice clones added yet. Register your first clone below!</p>"
    
    html_out = """
    <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 10px;">
        <h4 style="font-size: 14px; font-weight: 700; color: #17171d; margin-bottom: 2px;">Registered Voice Clones Library:</h4>
    """
    for f in wav_files:
        parts = f.rsplit("_", 1)
        c_name = parts[0].replace("_", " ").title() if len(parts) == 2 else f
        html_out += f"""
        <div style="display: flex; align-items: center; justify-content: space-between; background: #f8f8fb; border: 1px solid #e7e7ed; border-radius: 10px; padding: 10px 14px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 16px;">🧬</span>
                <div>
                    <div style="font-weight: 600; font-size: 13px; color: #17171d;">{c_name}</div>
                    <div style="font-size: 11px; color: #888894;">Status: Ready for Generation</div>
                </div>
            </div>
            <span style="background: #e9e5fb; color: #6c47ff; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;">Active</span>
        </div>
        """
    html_out += "</div>"
    return html_out


def fmt_time(s):
    s = int(s)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60}s"


def monitor_html(pct, status, elapsed_sec, estimated, words, active=True):
    elapsed_str = fmt_time(elapsed_sec) if isinstance(elapsed_sec, (int, float)) and elapsed_sec >= 0 else "—"
    is_active = active and 0 < pct < 100
    state = "Generating" if is_active else ("Complete" if pct >= 100 else "Ready")
    completed_class = "completed" if pct >= 100 else ""
    spinner = '<div class="spinner"></div>' if is_active else ''
    check = '<div class="checkmark">✓</div>' if pct >= 100 else ''
    return f"""
    <div class="monitor {completed_class}">
      <div class="mon-head">
        {spinner}{check}
        <span class="mon-state">{state}</span>
        <span class="mon-pct">{pct}%</span>
      </div>
      <div class="bar">
        <div class="fill" style="width:{pct}%;"></div>
      </div>
      <div class="stats">
        <div class="stat"><span class="s-lab">Audio</span><span class="s-val">{estimated}</span></div>
        <div class="stat"><span class="s-lab">Time</span><span class="s-val">{elapsed_str}</span></div>
        <div class="stat"><span class="s-lab">Words</span><span class="s-val">{words}</span></div>
      </div>
    </div>
    """


def split_text(text, max_chars=200):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


def normalize_audio(wav, sr, target_lufs=-16.0):
    try:
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        meter = pyln.Meter(sr)
        loud = meter.integrated_loudness(wav)
        out = pyln.normalize.loudness(wav, loud, target_lufs)
        mx = np.abs(out).max()
        if mx > 0:
            out = out / mx * 0.95
        return out.astype(np.float32)
    except Exception as e:
        print(f"Normalize failed: {e}")
        return wav.astype(np.float32)


def crossfade_chunks(chunks, sr, fade_ms=60):
    fade_len = int(fade_ms * sr / 1000)
    if len(chunks) == 1:
        return chunks[0]
    result = chunks[0].copy()
    for nxt in chunks[1:]:
        nxt = nxt.copy()
        if len(result) < fade_len or len(nxt) < fade_len:
            result = np.concatenate([result, nxt])
            continue
        fade_out = np.linspace(1, 0, fade_len).astype(np.float32)
        fade_in = np.linspace(0, 1, fade_len).astype(np.float32)
        result[-fade_len:] *= fade_out
        nxt[:fade_len] *= fade_in
        result[-fade_len:] += nxt[:fade_len]
        result = np.concatenate([result, nxt[fade_len:]])
    return result


def generate_voice(selected_dropdown_choice, script):
    if not script or not script.strip():
        yield None, monitor_html(0, "Waiting", 0, "—", 0, False), gr.update(interactive=True)
        return

    clones_map = get_available_clones()
    ref_audio = None
    if selected_dropdown_choice and selected_dropdown_choice in clones_map:
        ref_audio = clones_map[selected_dropdown_choice]

    words = len(script.split())
    est_sec = words / 3.5
    est_str = fmt_time(est_sec)
    t_start = time.time()

    def elapsed():
        return int(time.time() - t_start)

    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        yield None, monitor_html(5, "Preparing", 0, est_str, words, True), gr.update(interactive=False)
        time.sleep(0.3)

        chunks = split_text(script, max_chars=200)
        print(f"[VORVEX] Split into {len(chunks)} chunk(s)")

        all_audio = []
        sr = model.sample_rate

        for ci, chunk in enumerate(chunks):
            pct = 10 + int(((ci + 1) / len(chunks)) * 75)
            yield None, monitor_html(pct, f"Generating {ci+1}/{len(chunks)}",
                                     elapsed(), est_str, words, True), gr.update(interactive=False)

            if ref_audio and os.path.exists(ref_audio):
                result = model.generate(chunk, mode="voice_clone", reference_audio=ref_audio)
            else:
                result = model.generate(chunk, mode="tts")

            w = result.waveform.squeeze(0).cpu().numpy().astype(np.float32)
            w = normalize_audio(w, sr, -16.0)
            all_audio.append(w)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        yield None, monitor_html(90, "Joining", elapsed(), est_str, words, True), gr.update(interactive=False)
        if len(all_audio) > 1:
            final_wav = crossfade_chunks(all_audio, sr, fade_ms=60)
        else:
            final_wav = all_audio[0]

        yield None, monitor_html(95, "Saving", elapsed(), est_str, words, True), gr.update(interactive=False)
        fn = f"vorvex_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.wav"
        path = os.path.join(OUT_DIR, fn)
        ta.save(path, torch.from_numpy(final_wav).unsqueeze(0), sr)

        dur = len(final_wav) / sr
        tt = time.time() - t_start
        print(f"✅ Saved: {path} | {fmt_time(dur)} | {fmt_time(tt)}")
        yield path, monitor_html(100, "Complete", int(tt), fmt_time(dur), words, False), gr.update(interactive=True)

    except Exception as e:
        import traceback; traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield None, monitor_html(0, "Error", 0, "—", "—", False), gr.update(interactive=True)


def update_active_voice_display(selected_choice):
    if not selected_choice or "Default" in selected_choice:
        return """
        <div class="voice-card">
          <div class="voice-art">〰</div>
          <div style="min-width:0;flex:1">
            <div class="voice-name">Default VORVEX Voice</div>
            <span class="voice-tag">Standard • On-device</span>
          </div>
        </div>
        """
    else:
        return f"""
        <div class="voice-card" style="border-color:#a58afa; background:#f6f3ff;">
          <div class="voice-art" style="background:linear-gradient(135deg,#7c4dff,#6c47ff);">🧬</div>
          <div style="min-width:0;flex:1">
            <div class="voice-name" style="font-weight:700; color:#5a28ef;">{selected_choice}</div>
            <span class="voice-tag" style="background:#ede8fb; color:#6c47ff;">Active Clone • Custom</span>
          </div>
        </div>
        """


def register_new_clone(clone_name, clone_audio_path):
    if not clone_name or not clone_name.strip():
        return "⚠️ **Action Required:** Please enter a valid name for your voice clone before saving.", gr.update(), "", gr.update(), gr.update()
    if not clone_audio_path or not os.path.exists(clone_audio_path):
        return "❌ Please upload or record a valid reference audio file.", gr.update(), "", gr.update(), gr.update()
    
    try:
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', clone_name.strip())
        target_path = os.path.join(CLONE_STORE_DIR, f"{safe_name}_{uuid.uuid4().hex[:6]}.wav")
        
        if clone_audio_path != target_path:
            import shutil
            shutil.copy(clone_audio_path, target_path)
            
        updated_clones = list(get_available_clones().keys())
        new_selection = [c for c in updated_clones if safe_name.lower() in c.lower()]
        sel_val = new_selection[0] if new_selection else updated_clones[-1]

        msg = f"✅ Success! Voice Clone '{clone_name.strip()}' has been added."
        updated_list_html = get_clones_list_html()
        
        return msg, gr.update(choices=updated_clones, value=sel_val), updated_list_html, "", None
    except Exception as e:
        return f"❌ Error saving clone: {str(e)}", gr.update(), "", "", None


# ============================================================
# AUTO EDITOR (Lightning Fast Video Assembler - ROBUST TIMESTAMP PARSER)
# ============================================================
def parse_timestamp_str(ts_str):
    parts = ts_str.split('_')
    try:
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        elif len(parts) == 1:
            return float(parts[0])
    except:
        pass
    return 0.0


def auto_assemble_video(audio_file, image_files):
    if not audio_file:
        return "❌ Please provide or generate an audio file first.", None
    if not image_files:
        return "❌ Please upload your timestamped image files.", None

    try:
        from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
    except ImportError:
        try:
            from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
        except ImportError:
            return "❌ MoviePy library is not installed properly. Run: pip install moviepy", None

    try:
        clips = []
        audio_clip = AudioFileClip(audio_file)
        image_intervals = []
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            name_no_ext, _ = os.path.splitext(filename)
            
            match = re.findall(r'(\d+_\d+_\d+)[-_](\d+_\d+_\d+)', name_no_ext)
            if match:
                start_str, end_str = match[0]
                start_sec = parse_timestamp_str(start_str)
                end_sec = parse_timestamp_str(end_str)
                if end_sec > start_sec:
                    image_intervals.append((start_sec, end_sec, img_path))
            else:
                all_nums = re.findall(r'(\d+_\d+_\d+|\d+_\d+|\d+)', name_no_ext)
                if len(all_nums) >= 2:
                    start_sec = parse_timestamp_str(all_nums[0])
                    end_sec = parse_timestamp_str(all_nums[1])
                    if end_sec > start_sec:
                        image_intervals.append((start_sec, end_sec, img_path))

        if not image_intervals:
            return "❌ Could not parse timestamps. Use format: 00_00_00-00_03_50.jpg", None

        image_intervals.sort(key=lambda x: x[0])

        for start_sec, end_sec, img_path in image_intervals:
            duration = end_sec - start_sec
            if duration > 0:
                img_clip = ImageClip(img_path).with_duration(duration)
                clips.append(img_clip)

        if not clips:
            return "❌ No valid image clips generated.", None

        final_video_clip = concatenate_videoclips(clips, method="chain")
        
        if final_video_clip.duration > audio_clip.duration:
            final_video_clip = final_video_clip.subclipped(0, audio_clip.duration)

        final_video_clip = final_video_clip.with_audio(audio_clip)

        out_video_name = f"vorvex_video_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.mp4"
        out_video_path = os.path.join(OUT_DIR, out_video_name)

        final_video_clip.write_videofile(
            out_video_path, fps=24, codec="libx264", audio_codec="aac", preset="ultrafast", logger=None
        )

        audio_clip.close()
        final_video_clip.close()

        return f"✅ Video successfully exported to: {out_video_path}", out_video_path

    except Exception as e:
        import traceback; traceback.print_exc()
        return f"❌ Error during video assembly: {str(e)}", None


# ============================================================
# AUTO BLIND (New Section - Added Right After Auto Editor)
# ============================================================
def auto_blind_video(audio_file, image_files, blind_mode="Blur Overlay"):
    if not audio_file:
        return "❌ Please provide or generate an audio file first.", None
    if not image_files:
        return "❌ Please upload your image/video files for Auto Blind processing.", None

    try:
        from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, vfx
    except ImportError:
        try:
            from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, vfx
        except ImportError:
            return "❌ MoviePy library is not installed properly. Run: pip install moviepy", None

    try:
        clips = []
        audio_clip = AudioFileClip(audio_file)
        total_duration = audio_clip.duration
        clip_duration = total_duration / max(len(image_files), 1)

        for img_path in image_files:
            img_clip = ImageClip(img_path).with_duration(clip_duration)
            
            # Apply transformation depending on Auto Blind mode
            if "Blur" in blind_mode or "Blind" in blind_mode:
                try:
                    # Apply gentle styling / blur simulation effect if supported by vfx
                    img_clip = img_clip.with_effects([vfx.GaussianBlur(sigma=2)] if hasattr(vfx, 'GaussianBlur') else [])
                except:
                    pass
            
            clips.append(img_clip)

        if not clips:
            return "❌ No valid clips generated for Auto Blind.", None

        final_video_clip = concatenate_videoclips(clips, method="chain")
        
        if final_video_clip.duration > audio_clip.duration:
            final_video_clip = final_video_clip.subclipped(0, audio_clip.duration)

        final_video_clip = final_video_clip.with_audio(audio_clip)

        out_video_name = f"vorvex_blind_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.mp4"
        out_video_path = os.path.join(OUT_DIR, out_video_name)

        final_video_clip.write_videofile(
            out_video_path, fps=24, codec="libx264", audio_codec="aac", preset="ultrafast", logger=None
        )

        audio_clip.close()
        final_video_clip.close()

        return f"✅ Auto Blind Video successfully exported to: {out_video_path}", out_video_path

    except Exception as e:
        import traceback; traceback.print_exc()
        return f"❌ Error during Auto Blind processing: {str(e)}", None


# ============================================================
# VOICE TO TRANSCRIPT (Stable Standard STT Engine)
# ============================================================
def transcribe_audio(audio_file):
    if not audio_file:
        yield "❌ Please provide an audio file to transcribe.", gr.update(interactive=True)
        return

    try:
        import whisper
    except ImportError:
        yield "❌ STT library not installed. Run: pip install openai-whisper", gr.update(interactive=True)
        return

    try:
        yield """
        <div style="display:flex; align-items:center; gap:10px; padding:15px; background:#f8f8fb; border:1px solid #e7e7ed; border-radius:12px; margin-top:10px;">
            <div class="spinner"></div>
            <span style="font-weight:600; color:#34343e; font-size:14px;">⏳ Transcribing audio... Please wait.</span>
        </div>
        """, gr.update(interactive=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        stt_model = whisper.load_model("tiny", device=device)
        
        print(f"🎙️ Transcription started: {audio_file}")
        result = stt_model.transcribe(audio_file)
        
        segments = result.get("segments", [])
        if not segments:
            yield "*(No speech detected or audio is too short)*", gr.update(interactive=True)
            return

        plain_text_lines = []
        for segment in segments:
            start_s = int(segment['start'])
            end_s = int(segment['end'])
            s_min, s_sec = divmod(start_s, 60)
            e_min, e_sec = divmod(end_s, 60)
            time_tag = f"[{s_min:02d}:{s_sec:02d} - {e_min:02d}:{e_sec:02d}]"
            text = segment['text'].strip()
            plain_text_lines.append(f"{time_tag} {text}")

        full_plain_text = "\n".join(plain_text_lines)

        formatted_html = f"""
        <div style="background:#fff; border:1px solid #e7e7ed; border-radius:14px; padding:16px; margin-top:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #e9e9ef; padding-bottom:8px;">
                <span style="font-size:14px; font-weight:700; color:#17171d;">📝 Timestamped Transcript</span>
                <button onclick="navigator.clipboard.writeText(`{full_plain_text}`); this.innerText='✅ Copied!'; setTimeout(()=>this.innerText='📋 Copy', 2000);" style="background:#6c47ff; color:white; border:none; border-radius:6px; padding:4px 10px; font-size:12px; font-weight:600; cursor:pointer;">📋 Copy</button>
            </div>
            <div style="max-height:380px; overflow-y:auto; display:flex; flex-direction:column; gap:8px; font-size:13px; color:#33333d; line-height:1.5;">
        """

        for segment in segments:
            start_s = int(segment['start'])
            end_s = int(segment['end'])
            s_min, s_sec = divmod(start_s, 60)
            e_min, e_sec = divmod(end_s, 60)
            time_tag = f"[{s_min:02d}:{s_sec:02d} - {e_min:02d}:{e_sec:02d}]"
            text = segment['text'].strip()
            
            formatted_html += f"""
                <div style="display:flex; gap:8px; background:#f8f8fb; padding:8px 10px; border-radius:8px; border:1px solid #f0f0f4;">
                    <span style="color:#6c47ff; font-weight:600; font-family:monospace; white-space:nowrap;">{time_tag}</span>
                    <span style="word-break:break-word;">{text}</span>
                </div>
            """

        formatted_html += "</div></div>"
        
        yield formatted_html, gr.update(interactive=True)
        
    except Exception as e:
        import traceback; traceback.print_exc()
        yield f"❌ Error during transcription: {str(e)}", gr.update(interactive=True)


# ============================================================
# YT THUMBNAIL DOWNLOADER UTILITIES
# ============================================================
def extract_youtube_id(url):
    url = url.strip()
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if len(url) == 11 and re.match(r'^[0-9A-Za-z_-]{11}$', url):
        return url
    return None


def fetch_youtube_thumbnails(url):
    if not url or not url.strip():
        return "", "<div style='color:#ff4d4f; font-size:14px; padding:20px; text-align:center;'>❌ Please enter a valid YouTube video link.</div>", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    vid_id = extract_youtube_id(url)
    if not vid_id:
        return "", "<div style='color:#ff4d4f; font-size:14px; padding:20px; text-align:center;'>❌ Could not extract Video ID. Please check the URL.</div>", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    maxres_url = f"https://img.youtube.com/vi/{vid_id}/maxresdefault.jpg"
    hq_url = f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg"
    
    try:
        import urllib.request
        local_path = os.path.join(OUT_DIR, f"thumb_{vid_id}.jpg")
        urllib.request.urlretrieve(maxres_url, local_path)
        display_path = local_path
    except:
        display_path = maxres_url

    status_msg = ""
    
    html_res = f"""
    <div style="background:#fff; border:1px solid #e7e7ed; border-radius:14px; padding:16px; margin-top:14px; text-align:center;">
        <h4 style="font-size: 15px; font-weight: 700; color: #17171d; margin-bottom: 12px;">High-Resolution Thumbnail Preview</h4>
        <img src="{maxres_url}" style="width:100%; max-width:640px; border-radius:10px; border:1px solid #e9e9ef; box-shadow:0 4px 12px rgba(0,0,0,0.05); margin-bottom:14px;" onerror="this.onerror=null; this.src='{hq_url}';"/>
        <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
            <a href="{maxres_url}" target="_blank" style="background:#6c47ff; color:white; padding:8px 16px; border-radius:8px; font-size:13px; font-weight:600; text-decoration:none;">📥 Open Maxres Image</a>
            <a href="{hq_url}" target="_blank" style="background:#f0f0f4; color:#33333d; padding:8px 16px; border-radius:8px; font-size:13px; font-weight:600; text-decoration:none;">📥 Open Standard HQ</a>
        </div>
    </div>
    """
    return status_msg, html_res, display_path, maxres_url, hq_url


# ============================================================
# CSS Styling
# ============================================================
CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --purple:#9b80ff; --ink:#17171d; --muted:#92929e; --line:#e9e9ef; }
{ box-sizing:border-box !important; font-family:Inter,Arial,sans-serif !important; }
html, body, #root, .gradio-container { background:#fff !important; color:var(--ink) !important; max-width:100% !important; width:100% !important; margin:0 !important; }
.gradio-container { padding:0 18px 16px !important; min-height:100vh !important; }
footer, .built-with { display:none !important; }
button { transition:all .15s ease !important; }

.main-content-tabs > .tab-nav { display: none !important; }

.vorvex-topbar { height:76px; display:flex; align-items:center; gap:32px; border-bottom:1px solid #f1f1f5; margin:0 -18px 17px; padding:0 22px; }
.vorvex-brand { display:flex; align-items:center; gap:11px; font-size:20px; font-weight:800; letter-spacing:-.7px; white-space:nowrap; }
.vorvex-logo { width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(140deg,#7c4dff,#a58bfa);box-shadow:0 4px 10px #8b5cf626; }
.vorvex-topnav { color:#777783; font-size:16px; }
.vorvex-topnav b { color:#16161c; font-weight:500; }
.vorvex-topright { margin-left:auto; display:flex;align-items:center;gap:12px;color:#6c6c78;font-size:13px; }
.top-chip { border:1px solid #e9e9ef;border-radius:24px;padding:10px 14px;white-space:nowrap; }
.app-layout { gap:18px !important; align-items:stretch !important; }
.sidebar-col { min-width:220px !important; max-width:252px !important; }

.tab-nav { display: flex; flex-direction: column; gap: 6px; }
.tab-nav button { width: 100% !important; justify-content: flex-start !important; height: 49px !important; padding: 0 13px !important; border-radius: 12px !important; border: none !important; background: transparent !important; color: #292934 !important; font-size: 15px !important; font-weight: 500 !important; box-shadow: none !important; text-align: left !important; }
.tab-nav button:hover { background: #f2f1f6 !important; }
.tab-nav button.selected { background: #f6f5f9 !important; font-weight: 650 !important; color: #17171d !important; }

.side-bottom { margin-top:auto; }
.side-subscribe { width:100%;height:42px;border:1px solid #25D366;border-radius:24px;background:#25D366;color:#fff;font-size:15px;font-weight:600;margin:12px 0 8px;display:flex;align-items:center;justify-content:center;text-decoration:none;cursor:pointer; }
.side-subscribe:hover { background:#22bf5b; }
.side-api { width:100%;height:39px;border:0;border-radius:24px;background:#f5f5f8;color:#666675; }
.side-footer { border-top:1px solid #eeeef3;padding:15px 3px 0;margin-top:18px;color:#a1a1ac;font-size:11px;line-height:2; }
.page-head { display:flex;align-items:center;justify-content:space-between;gap:14px;margin:4px 0 0; }
.page-head h1 { font-size:25px;font-weight:650;letter-spacing:-.5px;margin:0; }
.model-chip { border:1px solid #e9e9ef;border-radius:13px;padding:13px 17px;font-size:14px;min-width:250px;text-align:left;color:#33333d;background:#fff; }
.editor-card, .settings-card { border:1px solid #e7e7ee;border-radius:14px;background:#fff; }
.editor-card { margin-top:16px;padding:0;overflow:hidden; }

.tts-bottom-row { display: flex; gap: 10px; padding: 12px 16px; border-top: 1px solid #eeeef3; background: #fff; }
.gradio-container .generate-btn button { width:100% !important;border:0 !important;border-radius:24px !important;background:#a58afa !important;color:white !important;font-size:15px !important;font-weight:600 !important;min-height:42px !important;box-shadow:none !important; margin-top: 8px !important; }
.gradio-container .generate-btn button:hover { background:#9272f2 !important; }
.gradio-container .clear-btn button { background:#fff !important;border:1px solid #e5e5ec !important;border-radius:10px !important;color:#777783 !important; }

.compact-clone-box { padding: 20px; background: #fff; border-radius: 14px; border: 1px solid #e7e7ee; margin-top: 14px; max-width: 750px; }
.compact-clone-box input[type="text"] { border: 1px solid #e5e5ec !important; border-radius: 10px !important; min-height: 42px !important; padding: 8px 14px !important; background: #fff !important; }

.settings-card { padding:0 17px 14px;min-height:0; }
.settings-tabs { display:flex;gap:28px;border-bottom:1px solid #f0f0f4;height:52px;align-items:stretch;margin-bottom:16px; }
.settings-tabs span { display:flex;align-items:center;font-size:16px;color:#9999a5; }
.settings-tabs .selected { color:#19191f;border-bottom:2px solid #24242a; }
.setting-head { display:flex;align-items:center;justify-content:space-between;margin:12px 0;font-size:15px; }
.voice-card { height:80px;border:1px solid #e7e7ed;border-radius:15px;background:#f8f8fb;display:flex;align-items:center;padding:9px;gap:10px;margin-bottom:12px; }
.voice-art { width:58px;height:58px;flex-shrink:0;border-radius:12px;background:linear-gradient(135deg,#4b22c7,#f04e91 45%,#ffba61);display:grid;place-items:center;color:white;font-size:22px; }
.voice-name { font-size:13px;color:#454550;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
.voice-tag { display:inline-block;margin-top:8px;background:#ececf2;border-radius:6px;padding:4px 8px;font-size:12px;color:#777784; }
.monitor { background:#fafafd;border:1px solid #e9e9ef;border-radius:12px;padding:10px 12px;margin-top:8px; }
.mon-head { display:flex;align-items:center;gap:9px;margin-bottom:10px; }
.mon-state { font-size:13px;font-weight:600;color:#34343e;flex:1; }
.mon-pct { font-size:18px;font-weight:700;color:#25252d; }
.spinner { width:14px;height:14px;border:2px solid #ded7ff;border-top-color:#8b5cf6;border-radius:50%;animation:spin .8s linear infinite; }
@keyframes spin {to{transform:rotate(360deg)}}
.checkmark {width:18px;height:18px;background:#10b981;color:white;border-radius:50%;display:grid;place-items:center;font-size:11px;}
.bar {width:100%;height:6px;background:#ececf2;border-radius:4px;overflow:hidden;margin-bottom:12px;}
.fill {height:100%;background:linear-gradient(90deg,#a78bfa,#8b5cf6);border-radius:4px;transition:width .3s;}
.monitor.completed {border-color:#c9f0dd}.monitor.completed .fill{background:#10b981 !important}
.stats {display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.stat {display:flex;flex-direction:column;gap:3px}.s-lab{font-size:10px;color:#92929e;text-transform:uppercase;letter-spacing:.5px}.s-val{font-size:12px;color:#454550}
.gradio-container audio { margin-top:8px; }
.gradio-container video { margin-top:8px; border-radius: 12px; }
.gradio-container .download-btn { display:none !important; }
"""


with gr.Blocks(title="VORVEX — Voice Studio", fill_height=True) as demo:
    gr.HTML("""
    <div class="vorvex-topbar">
      <div class="vorvex-brand">
        <div class="vorvex-logo">
          <svg width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 7.5 8.5 18 12 11.5 15.5 18 21 7.5"/>
            <path d="M8 5.5h8"/>
          </svg>
        </div>
        <span style="color:#111118 !important;-webkit-text-fill-color:#111118 !important;opacity:1 !important">VORVEX</span>
      </div>
      <div class="vorvex-topnav"><b>Studio Platform Made By Zargham</b></div>
      <div class="vorvex-topright"><span class="top-chip">♧ Tutorial</span><span class="top-chip">◉ Private Voice Studio</span><span>☼</span></div>
    </div>
    """)

    initial_clones = list(get_available_clones().keys())

    with gr.Row(elem_classes="app-layout", equal_height=False):
        with gr.Column(scale=2, min_width=190, elem_classes="sidebar-col"):
            with gr.Tabs(elem_classes="tab-nav") as main_tabs:
                home_tab = gr.TabItem("🏠 Home", id="home_tab")
                tts_tab = gr.TabItem("🎙️ Text to Speech", id="tts_tab")
                clone_tab = gr.TabItem("🗣️ Voice Clone", id="clone_tab")
                editor_tab = gr.TabItem("🎬 Auto Editor", id="editor_tab")
                blind_tab = gr.TabItem("🕵️ Auto Blind", id="blind_tab")
                transcript_tab = gr.TabItem("📝 Voice to Transcript", id="transcript_tab")
                thumbnail_tab = gr.TabItem("🖼️ YouTube Thumbnail Downloader", id="thumbnail_tab")
                tutorials_tab = gr.TabItem("📚 Tutorials", id="tutorials_tab")
                request_tab = gr.TabItem("💡 Request a Feature", id="request_tab")

            gr.HTML("""
            <div class="side-bottom" style="padding-top: 20px;">
              <a href="https://wa.me/923201606966?text=Hello%20VORVEX%20Support" target="_blank" class="side-subscribe">💬 &nbsp; WhatsApp Support</a>
              <button class="side-api" onclick="alert('VORVEX Voice Studio')">VORVEX Studio</button>
              <div class="side-footer">About VORVEX &nbsp; | &nbsp; Terms of Service<br>Privacy Policy &nbsp; | &nbsp; © VORVEX 2026</div>
            </div>
            """)

        with gr.Column(scale=11, min_width=750):
            with gr.Tabs(elem_classes="main-content-tabs"):
                
                with gr.TabItem("Home", id="home_tab_content", visible=True):
                    gr.HTML("""
                    <div style="padding: 20px 10px; max-width: 950px;">
                        <div style="background: linear-gradient(135deg, #7c4dff15, #a58bfa10); border: 1px solid #e7e7ed; border-radius: 20px; padding: 40px; margin-bottom: 30px;">
                            <span style="background: #6c47ff15; color: #6c47ff; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">VORVEX STUDIO 1.0</span>
                            <h1 style="font-size: 36px; font-weight: 800; color: #111118; margin-top: 15px; margin-bottom: 12px; letter-spacing: -1px;">All In One Ai Platform</h1>
                            <p style="color: #666675; font-size: 16px; line-height: 1.6; max-width: 700px; margin-bottom: 24px;">Have a question, found an issue, or have an idea to share? We'd love to hear from you. Whether you need help, want to report a problem, suggest a new feature, or discuss any topic, feel free to contact us. Your feedback helps us make this platform better for everyone.</p>
                            
                            <div style="border-top: 1px solid #e7e7ed; padding-top: 24px; margin-top: 20px;">
                                <h3 style="font-size: 20px; font-weight: 700; color: #111118; margin-bottom: 10px;">We're Here to Help</h3>
                                <p style="color: #666675; font-size: 14px; line-height: 1.5; margin-bottom: 16px;">Have a question, found a bug, or have an idea for a new feature? Feel free to contact us. Whether you need support or simply want to discuss something, we'd love to hear from you.</p>
                                <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                                    <a href="https://wa.me/923201606966?text=Hello%20VORVEX%20Support" target="_blank" style="background:#6c47ff; color:white; padding:8px 16px; border-radius:8px; font-size:13px; font-weight:600; text-decoration:none;">Contact Us</a>
                                    <a href="https://wa.me/923201606966?text=Bug%20Report%20VORVEX" target="_blank" style="background:#f0f0f4; color:#33333d; padding:8px 16px; border-radius:8px; font-size:13px; font-weight:600; text-decoration:none;">Report a Bug</a>
                                    <a href="https://wa.me/923201606966?text=Feature%20Suggestion%20VORVEX" target="_blank" style="background:#f0f0f4; color:#33333d; padding:8px 16px; border-radius:8px; font-size:13px; font-weight:600; text-decoration:none;">Suggest a Feature</a>
                                </div>
                            </div>
                        </div>
                    </div>
                    """)

                with gr.TabItem("Text to Speech", id="tts_tab_content"):
                    with gr.Row():
                        with gr.Column(scale=7, min_width=420):
                            gr.HTML('<div class="page-head"><h1 style="color:#111118 !important">Voice Studio</h1><div class="model-chip">Voice Model &nbsp; │ &nbsp; VORVEX <span style="background:#15151b;color:#fff;border-radius:4px;padding:3px 5px;font-size:10px">READY</span></div></div>')
                            with gr.Column(elem_classes="editor-card"):
                                script_input = gr.Textbox(
                                    label="Enter your script",
                                    placeholder="Start typing here to generate natural-sounding speech...",
                                    lines=8, show_label=True, elem_id="script-input"
                                )
                                with gr.Row(elem_classes="tts-bottom-row"):
                                    clear_btn = gr.Button("Clear", elem_classes="clear-btn", scale=1)
                                    gen_btn = gr.Button("✦  Generate Audio", elem_classes="generate-btn", scale=4)

                        with gr.Column(scale=4, min_width=300):
                            with gr.Column(elem_classes="settings-card"):
                                gr.HTML('<div class="settings-tabs"><span class="selected">Settings</span><span>History</span></div>')
                                gr.HTML('<div class="setting-head"><span>Active Voice Model</span></div>')
                                clone_selector = gr.Dropdown(
                                    label="Select from Voice Library", choices=initial_clones, value=initial_clones[0], interactive=True
                                )
                                active_voice_display = gr.HTML(value=update_active_voice_display(initial_clones[0]))
                                monitor_output = gr.HTML(value=monitor_html(0, "Ready", 0, "—", 0, False))
                                audio_output = gr.Audio(label="Generated Audio", interactive=False)

                with gr.TabItem("Voice Clone", id="clone_tab_content"):
                    gr.HTML('<div class="page-head"><h1 style="color:#111118 !important">Voice Clone Studio</h1><div class="model-chip">Add & Register Multiple Clones</div></div>')
                    with gr.Column(elem_classes="editor-card"):
                        with gr.Column(elem_classes="compact-clone-box"):
                            gr.HTML("<h3 style='font-size: 16px; font-weight: 700; color: #17171d; margin-bottom: 4px;'>Register New Custom Voice</h3>")
                            gr.HTML("<p style='color: #777783; font-size: 13px; margin-bottom: 12px;'>Enter a name and upload a clean audio sample to add a new voice to your library.</p>")
                            clone_name_input = gr.Textbox(label="Voice Clone Name", placeholder="e.g. Narrator Alex, Storyteller Sarah, etc.")
                            clone_sample_input = gr.Audio(label="Reference Audio Sample (WAV / MP3)", sources=["upload", "microphone"], type="filepath")
                            add_clone_btn = gr.Button("✨ Save & Register Voice Clone", elem_classes="generate-btn")
                            clone_status_output = gr.Markdown(value="")
                        saved_clones_list_display = gr.HTML(value=get_clones_list_html())

                with gr.TabItem("Auto Editor", id="editor_tab_content"):
                    gr.HTML('<div class="page-head"><h1 style="color:#111118 !important">Auto Editor Studio</h1><div class="model-chip">Timestamp Image-to-Video Assembler</div></div>')
                    with gr.Column(elem_classes="editor-card"):
                        with gr.Column(elem_classes="compact-clone-box"):
                            gr.HTML("<h3 style='font-size: 16px; font-weight: 700; color: #17171d; margin-bottom: 4px;'>Automatic Video Creator</h3>")
                            gr.HTML("<p style='color: #777783; font-size: 13px; margin-bottom: 12px;'>Upload your audio file and timestamped images (e.g., <code>00_00_00-00_03_50.jpg</code>) to render your video.</p>")
                            editor_audio_input = gr.Audio(label="Audio File (WAV / MP3)", sources=["upload"], type="filepath")
                            editor_images_input = gr.File(label="Upload Timestamped Images", file_count="multiple", type="filepath")
                            assemble_btn = gr.Button("🎬 Render & Export Final Video", elem_classes="generate-btn")
                            editor_status_output = gr.Markdown(value="")
                            editor_video_output = gr.Video(label="Exported Video Preview", interactive=False)

                with gr.TabItem("Auto Blind", id="blind_tab_content"):
                    gr.HTML('<div class="page-head"><h1 style="color:#111118 !important">Auto Blind Studio</h1><div class="model-chip">Automatic Blended/Masked Video Assembler</div></div>')
                    with gr.Column(elem_classes="editor-card"):
                        with gr.Column(elem_classes="compact-clone-box"):
                            gr.HTML("<h3 style='font-size: 16px; font-weight: 700; color: #17171d; margin-bottom: 4px;'>Auto Blind Video Generator</h3>")
                            gr.HTML("<p style='color: #777783; font-size: 13px; margin-bottom: 12px;'>Upload voice track & media files to automatically composite with visual blind/blur effects.</p>")
                            blind_audio_input = gr.Audio(label="Audio File (WAV / MP3)", sources=["upload"], type="filepath")
                            blind_images_input = gr.File(label="Upload Media/Image Files", file_count="multiple", type="filepath")
                            blind_mode_dropdown = gr.Dropdown(label="Blind Processing Mode", choices=["Blur Overlay", "Clean Fade", "Mask Split"], value="Blur Overlay")
                            blind_btn = gr.Button("🕵️ Render Auto Blind Video", elem_classes="generate-btn")
                            blind_status_output = gr.Markdown(value="")
                            blind_video_output = gr.Video(label="Exported Auto Blind Video", interactive=False)

                with gr.TabItem("Voice to Transcript", id="transcript_tab_content"):
                    gr.HTML('<div class="page-head"><h1 style="color:#111118 !important">Voice to Transcript Studio</h1><div class="model-chip">Local STT Engine with Timestamps</div></div>')
                    with gr.Column(elem_classes="editor-card"):
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=5, min_width=320):
                                gr.HTML("<div style='padding: 20px;'>")
                                gr.HTML("<h3 style='font-size: 16px; font-weight: 700; color: #17171d; margin-bottom: 4px;'>Speech-to-Text Transcription</h3>")
                                gr.HTML("<p style='color: #777783; font-size: 13px; margin-bottom: 12px;'>Upload any long audio file to generate an accurate, timestamped transcript instantly.</p>")
                                stt_audio_input = gr.Audio(label="Audio File to Transcribe (WAV / MP3)", sources=["upload", "microphone"], type="filepath")
                                transcribe_btn = gr.Button("🎙️ Generate Timestamped Transcript", elem_classes="generate-btn")
                                gr.HTML("</div>")
                            
                            with gr.Column(scale=6, min_width=350):
                                gr.HTML("<div style='padding: 20px; border-left: 1px solid #f0f0f4;'>")
                                stt_output = gr.HTML(value="<div style='color:#92929e; font-size:14px; padding:30px 10px; text-align:center;'>Transcript will appear here once generated...</div>")
                                gr.HTML("</div>")

                with gr.TabItem("YouTube Thumbnail Downloader", id="thumbnail_tab_content"):
                    gr.HTML('<div class="page-head"><h1 style="color:#111118 !important">YT Thumbnail Downloader</h1><div class="model-chip">Instant High-Res Image Extractor</div></div>')
                    with gr.Column(elem_classes="editor-card"):
                        with gr.Column(elem_classes="compact-clone-box"):
                            gr.HTML("<h3 style='font-size: 16px; font-weight: 700; color: #17171d; margin-bottom: 4px;'>Paste YouTube Video Link</h3>")
                            gr.HTML("<p style='color: #777783; font-size: 13px; margin-bottom: 12px;'>Paste any YouTube video link or Shorts link below to fetch and download its full-resolution thumbnail instantly.</p>")
                            
                            thumb_url_input = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
                            fetch_thumb_btn = gr.Button("📥 Download Thumbnail", elem_classes="generate-btn")
                            thumb_status_output = gr.Markdown(value="")
                            
                            thumb_preview_html = gr.HTML(value="<div style='color:#92929e; font-size:14px; padding:20px; text-align:center;'>Thumbnail preview will appear here...</div>")
                            
                            with gr.Row():
                                thumb_image_download = gr.Image(label="Downloadable Image", type="filepath", interactive=False, visible=False)
                                thumb_link_maxres = gr.Textbox(label="MaxRes Link", visible=False)
                                thumb_link_hq = gr.Textbox(label="HQ Link", visible=False)

                with gr.TabItem("Tutorials", id="tutorials_tab_content"):
                    gr.HTML("""
                    <div style="padding: 20px 10px; max-width: 950px;">
                        <div class="page-head"><h1 style="color:#111118 !important">Tutorials & Guides</h1><div class="model-chip">How to use VORVEX Studio</div></div>
                        <div style="background: #fff; border: 1px solid #e7e7ed; border-radius: 14px; padding: 24px; margin-top: 16px;">
                            <h3 style="font-size: 16px; font-weight: 700; color: #17171d; margin-bottom: 8px;">📚 Step-by-Step Guide</h3>
                            <p style="color: #666675; font-size: 14px; line-height: 1.6; margin-bottom: 16px;">Learn how to make the most out of Text-to-Speech, Voice Cloning, Auto-Editing, Auto Blind, and Thumbnail Downloads.</p>
                            <ul style="color: #454550; font-size: 14px; line-height: 1.8; padding-left: 20px;">
                                <li><b>Text to Speech:</b> Enter your script in chunks, select your preferred voice model from library, and click Generate Audio.</li>
                                <li><b>Voice Clone:</b> Upload a clean WAV/MP3 voice sample, give it a custom name, and save it to your registry.</li>
                                <li><b>Auto Editor:</b> Upload your audio and timestamped images (format: <code>00_00_00-00_03_50.jpg</code>) to auto-assemble videos instantly.</li>
                                <li><b>Auto Blind:</b> Upload your audio and visual files to automatically render masked/blended video outputs.</li>
                                <li><b>Voice to Transcript:</b> Drop any long audio or speech file to get a clean, timestamped transcript.</li>
                                <li><b>Thumbnail Downloader:</b> Paste any YouTube URL to instantly preview and download full-res thumbnails.</li>
                            </ul>
                        </div>
                    </div>
                    """)

                with gr.TabItem("Request a Feature", id="request_tab_content"):
                    gr.HTML("""
                    <div style="padding: 20px 10px; max-width: 950px;">
                        <div class="page-head"><h1 style="color:#111118 !important">Request a Feature</h1><div class="model-chip">Help us improve VORVEX</div></div>
                        <div style="background: #fff; border: 1px solid #e7e7ed; border-radius: 14px; padding: 24px; margin-top: 16px;">
                            <h3 style="font-size: 16px; font-weight: 700; color: #17171d; margin-bottom: 8px;">💡 Got a new tool idea?</h3>
                            <p style="color: #666675; font-size: 14px; line-height: 1.6; margin-bottom: 16px;">We build features based on what creators need. Share your ideas directly with us via WhatsApp support or feedback channels.</p>
                            <a href="https://wa.me/923201606966?text=Feature%20Request%20for%20VORVEX:" target="_blank" style="background:#6c47ff; color:white; padding:10px 20px; border-radius:10px; font-size:14px; font-weight:600; text-decoration:none; display:inline-block;">💬 Submit Feature Request on WhatsApp</a>
                        </div>
                    </div>
                    """)

    clone_selector.change(fn=update_active_voice_display, inputs=[clone_selector], outputs=[active_voice_display])
    clear_btn.click(lambda: "", outputs=script_input)
    gen_btn.click(fn=generate_voice, inputs=[clone_selector, script_input], outputs=[audio_output, monitor_output, gen_btn])
    add_clone_btn.click(fn=register_new_clone, inputs=[clone_name_input, clone_sample_input], outputs=[clone_status_output, clone_selector, saved_clones_list_display, clone_name_input, clone_sample_input])
    assemble_btn.click(fn=auto_assemble_video, inputs=[editor_audio_input, editor_images_input], outputs=[editor_status_output, editor_video_output])
    blind_btn.click(fn=auto_blind_video, inputs=[blind_audio_input, blind_images_input, blind_mode_dropdown], outputs=[blind_status_output, blind_video_output])
    
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[stt_audio_input],
        outputs=[stt_output, transcribe_btn]
    )

    fetch_thumb_btn.click(
        fn=fetch_youtube_thumbnails,
        inputs=[thumb_url_input],
        outputs=[thumb_status_output, thumb_preview_html, thumb_image_download, thumb_link_maxres, thumb_link_hq]
    )

print("\n🚀 Launching VORVEX Voice Studio...")
FORCE_LIGHT_JS = """
() => {
  document.body.classList.print;
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light');
    window.location.replace(u.toString());
  }
}
"""
demo.queue().launch(share=True, debug=False, css=CSS, js=FORCE_LIGHT_JS)

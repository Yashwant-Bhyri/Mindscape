import mesop as me
import mindscape_engine
import os
import re
import time
from dataclasses import field

@me.stateclass
class State:
    transcript: str = ""
    is_recording: bool = False
    is_processing: bool = False
    status_message: str = "Ready"
    
    # BSV Data
    bsv_valence: float = 0.0
    bsv_arousal: float = 0.0
    bsv_dominance: float = 0.0
    
    # Hypothesis Data
    hypothesis_name: str = "Pending Analysis..."
    hypothesis_confidence: str = ""
    hypothesis_reasoning: str = "" 
    hypothesis_evidence: str = ""
    follow_up_questions: list[str] = field(default_factory=list)
    safety_gate: str = "PENDING"

def on_load(e: me.LoadEvent):
    me.set_theme_mode("dark")

@me.page(path="/", on_load=on_load)
def home():
    state = me.state(State)
    
    # ----------------------------------------------------
    # STYLES - Silicon Valley Dark Mode
    # ----------------------------------------------------
    color_bg = "#0a0a0a"
    color_card = "rgba(25, 25, 25, 0.6)" # Glass-like
    color_border = "rgba(255, 255, 255, 0.1)"
    color_accent_blue = "#2979ff"
    color_accent_red = "#ff1744"
    color_accent_green = "#00e676"
    color_text_primary = "#ffffff"
    color_text_secondary = "#a0a0a0"
    font_main = "Inter, sans-serif"

    style_page = me.Style(
        background=color_bg,
        height="100vh",
        font_family=font_main,
        color=color_text_primary,
        display="flex",
        flex_direction="column",
        padding=me.Padding.all(0)
    )

    style_header = me.Style(
        padding=me.Padding.symmetric(vertical=24, horizontal=40),
        border=me.Border(bottom=me.BorderSide(width=1, color=color_border)),
        display="flex",
        justify_content="space-between",
        align_items="center",
        background="rgba(10,10,10,0.8)",
        backdrop_filter="blur(10px)"
    )

    style_content = me.Style(
        padding=me.Padding.all(40),
        display="grid",
        grid_template_columns="1fr 400px",
        gap=40,
        height="100%",
        overflow_y="auto"
    )

    style_card = me.Style(
        background=color_card,
        border=me.Border(
            top=me.BorderSide(width=1, color=color_border),
            bottom=me.BorderSide(width=1, color=color_border),
            left=me.BorderSide(width=1, color=color_border),
            right=me.BorderSide(width=1, color=color_border)
        ),
        border_radius=16,
        padding=me.Padding.all(24),
        backdrop_filter="blur(20px)",
        box_shadow="0 8px 32px 0 rgba(0, 0, 0, 0.3)"
    )

    # ----------------------------------------------------
    # UI STRUCTURE
    # ----------------------------------------------------
    with me.box(style=style_page):
        # HEADER
        with me.box(style=style_header):
            with me.box(style=me.Style(display="flex", align_items="center", gap=12)):
                with me.box(style=me.Style(background="linear-gradient(135deg, #2979ff, #00e676)", width=32, height=32, border_radius=8)):
                    pass # Logo placeholder
                me.text("MindScape // 2.0", style=me.Style(font_size=24, font_weight="700", letter_spacing="-0.5px"))
            
            with me.box(style=me.Style(display="flex", align_items="center", gap=12)):
                status_color = color_accent_green if state.status_message == "Ready" else color_accent_blue
                if state.is_recording: status_color = color_accent_red
                
                with me.box(style=me.Style(width=8, height=8, border_radius="50%", background=status_color, margin=me.Margin(right=8))):
                    pass
                me.text(state.status_message, style=me.Style(color=color_text_secondary, font_family="monospace"))

        # MAIN CONTENT GRID
        with me.box(style=style_content):
            
            # LEFT COLUMN: LIVE DETECTION & TRANSCRIPT
            with me.box(style=me.Style(display="flex", flex_direction="column", gap=24)):
                
                # Live Mic Control
                with me.box(style=me.Style(
                    background=f"linear-gradient(to right, {color_card}, rgba(41, 121, 255, 0.05))",
                    border=me.Border(
                        top=me.BorderSide(width=1, color=color_accent_blue if state.is_recording else color_border),
                        bottom=me.BorderSide(width=1, color=color_accent_blue if state.is_recording else color_border),
                        left=me.BorderSide(width=1, color=color_accent_blue if state.is_recording else color_border),
                        right=me.BorderSide(width=1, color=color_accent_blue if state.is_recording else color_border)
                    ),
                    border_radius=16,
                    padding=me.Padding.all(32),
                    display="flex",
                    flex_direction="column",
                    align_items="center",
                    justify_content="center",
                    transition="border-color 0.3s ease",
                    min_height=200
                )):
                    if state.is_recording:
                         # Pulsing Circle Animation (Simulated via text for now as animation APIS are complex)
                        with me.box(style=me.Style(
                            width=80, height=80, border_radius="50%", 
                            background=color_accent_red, 
                            display="flex", align_items="center", justify_content="center",
                            box_shadow=f"0 0 30px {color_accent_red}"
                        )):
                            me.icon("mic", style=me.Style(color="white", font_size=40))
                        me.text("Listening...", style=me.Style(color=color_accent_red, margin=me.Margin(top=16), font_weight="bold"))
                    elif state.is_processing:
                        me.progress_spinner()
                        me.text("Analyzing Audio Vector...", style=me.Style(color=color_accent_blue, margin=me.Margin(top=16)))
                    else:
                        with me.box(style=me.Style(
                            width=80, height=80, border_radius="50%", 
                            background=color_accent_blue, 
                            display="flex", align_items="center", justify_content="center",
                            cursor="pointer"
                        ), on_click=toggle_recording):
                            me.icon("mic", style=me.Style(color="white", font_size=40))
                        me.text("Tap to Listen", style=me.Style(color=color_text_secondary, margin=me.Margin(top=16)))

                    with me.box(style=me.Style(width="100%", height=1, background=color_border, margin=me.Margin(top=24, bottom=24))):
                        pass

                    # Uploader
                    me.uploader(
                        label="Or Upload Audio File",
                        on_upload=handle_upload,
                        accepted_file_types=["audio/*"],
                        style=me.Style(
                            background=color_card, 
                            color=color_text_primary,
                            font_weight="bold",
                            border_radius=8
                        )
                    )

                # Transcript View
                with me.box(style=style_card):
                    me.text("Live Transcript", style=me.Style(font_size=14, color=color_text_secondary, text_transform="uppercase", letter_spacing="1px", margin=me.Margin(bottom=16)))
                    if state.transcript:
                        render_tags(state.transcript)
                    else:
                        me.text("No speech detected yet.", style=me.Style(color="#444", font_style="italic"))

            # RIGHT COLUMN: INTELLIGENCE DASHBOARD
            with me.box(style=me.Style(display="flex", flex_direction="column", gap=24)):
                
                # BSV Visualization
                with me.box(style=style_card):
                    me.text("Behavioral State Vector", style=me.Style(font_size=14, color=color_text_secondary, text_transform="uppercase", letter_spacing="1px", margin=me.Margin(bottom=24)))
                    
                    bsv_meter("Valence", state.bsv_valence, -1, 1, ["#ff1744", "#00e676"]) # Red to Green
                    bsv_meter("Arousal", state.bsv_arousal, 0, 1, ["#2979ff", "#ff9100"])  # Blue to Orange
                    bsv_meter("Dominance", state.bsv_dominance, 0, 1, ["#651fff", "#00b0ff"]) # Purple to Cyan

                # Diagnostic Hypothesis
                render_hypothesis_card(state)

def render_hypothesis_card(state):
    color_bg = "#0a0a0a"
    color_card = "rgba(25, 25, 25, 0.6)" # Glass-like
    color_border = "rgba(255, 255, 255, 0.1)"
    color_accent_blue = "#2979ff"
    color_accent_red = "#ff1744"
    color_accent_green = "#00e676"
    color_text_primary = "#ffffff"
    color_text_secondary = "#a0a0a0"

    color_border_status = color_accent_green if state.safety_gate == "PASS" else color_accent_red
    
    with me.box(style=me.Style(
        background=color_card,
        border=me.Border(
            top=me.BorderSide(width=1, color=color_border),
            right=me.BorderSide(width=1, color=color_border),
            bottom=me.BorderSide(width=1, color=color_border),
            left=me.BorderSide(width=4, color=color_border_status), # Status indicator
        ),
        border_radius=12,
        padding=me.Padding.all(24),
        display="flex",
        flex_direction="column",
        gap=24
    )):
        with me.box(style=me.Style(display="flex", justify_content="space-between", align_items="center")):
            me.text("Preliminary Diagnosis", style=me.Style(color=color_text_secondary, font_weight="bold", font_size=14, text_transform="uppercase", letter_spacing="1px"))
            # Status Badge
            with me.box(style=me.Style(
                background="rgba(0, 230, 118, 0.1)" if state.safety_gate == "PASS" else "rgba(255, 23, 68, 0.1)",
                padding=me.Padding(top=6, bottom=6, left=12, right=12),
                border_radius=6
            )):
                me.text(f"GATE: {state.safety_gate}", style=me.Style(
                    color=color_accent_green if state.safety_gate == "PASS" else color_accent_red, 
                    font_size=12, font_weight="bold"
                ))

        me.box(style=me.Style(height=8)) # Spacer
        
        # Diagnosis Name + Confidence
        with me.box(style=me.Style(display="flex", align_items="baseline", gap=12, flex_wrap="wrap")):
             me.text(state.hypothesis_name, style=me.Style(color="white", font_size=32, font_weight="bold", letter_spacing="-0.5px"))
             if state.hypothesis_confidence:
                 me.text(f"({state.hypothesis_confidence})", style=me.Style(color=color_text_secondary, font_size=16, font_family="monospace"))
        
        if state.hypothesis_reasoning:
             me.markdown(state.hypothesis_reasoning, style=me.Style(color=color_text_secondary, font_size=16, font_style="italic", line_height="1.6"))

        # Evidence Section
        with me.box(style=me.Style(margin=me.Margin(top=0))):
            me.text("Evidence", style=me.Style(color=color_text_primary, font_weight="bold", font_size=14, margin=me.Margin(bottom=12)))
            me.markdown(state.hypothesis_evidence, style=me.Style(color=color_text_secondary, font_size=15, line_height="1.6"))
        
        # Follow-up Questions (Suggested Inquiry)
        if state.follow_up_questions:
             with me.box(style=me.Style(margin=me.Margin(top=8))):
                me.text("Suggested Inquiry", style=me.Style(color=color_accent_blue, font_weight="bold", font_size=14, margin=me.Margin(bottom=12)))
                with me.box(style=me.Style(display="flex", flex_direction="column", gap=8)):
                    for q in state.follow_up_questions:
                        with me.box(style=me.Style(
                            background="rgba(41, 121, 255, 0.1)",
                            padding=me.Padding.all(12),
                            border_radius=8,
                            border=me.Border(left=me.BorderSide(width=2, color=color_accent_blue))
                        )):
                             me.text(q, style=me.Style(color="#e0e0e0", font_size=14, font_style="italic"))

def bsv_meter(label, value, min_val, max_val, colors):
    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0.0, min(1.0, normalized))
    percentage = f"{normalized * 100}%"
    
    with me.box(style=me.Style(margin=me.Margin(bottom=20))):
        with me.box(style=me.Style(display="flex", justify_content="space-between", margin=me.Margin(bottom=8))):
            me.text(label, style=me.Style(font_weight="500", font_size=14))
            me.text(f"{value:.2f}", style=me.Style(font_family="monospace", color="#a0a0a0"))
        
        # Track
        with me.box(style=me.Style(background="rgba(255,255,255,0.1)", height=6, border_radius=3, overflow="hidden")):
            # Fill
            with me.box(style=me.Style(
                background=f"linear-gradient(90deg, {colors[0]}, {colors[1]})", 
                width=percentage, 
                height="100%", 
                border_radius=3,
                transition="width 0.5s ease"
            )):
                pass

def render_tags(text):
    parts = re.split(r'(<[^>]+>)', text)
    with me.box(style=me.Style(display="flex", flex_wrap="wrap", gap=8, align_items="center")):
        for part in parts:
            part = part.strip()
            if not part: continue
            if part.startswith("<") and part.endswith(">"):
                with me.box(style=me.Style(
                    background="rgba(41, 121, 255, 0.2)", 
                    color="#82b1ff", 
                    padding=me.Padding.symmetric(horizontal=10, vertical=4), 
                    border_radius=20, 
                    font_size=12, 
                    font_weight="bold",
                    border=me.Border(
                        top=me.BorderSide(width=1, color="rgba(41, 121, 255, 0.4)"),
                        bottom=me.BorderSide(width=1, color="rgba(41, 121, 255, 0.4)"),
                        left=me.BorderSide(width=1, color="rgba(41, 121, 255, 0.4)"),
                        right=me.BorderSide(width=1, color="rgba(41, 121, 255, 0.4)")
                    )
                )):
                    me.text(part)
            else:
                me.text(part, style=me.Style(font_size=16, line_height="1.6", color="#eee"))

def toggle_recording(e: me.ClickEvent):
    state = me.state(State)
    if not state.is_recording:
        state.is_recording = True
        state.status_message = "Listening..."
        state.hypothesis_name = "Analyzing..."
        state.transcript = ""
        yield
        
        # Trigger recording
        # This blocks in MVP (5 seconds). In real app, async.
        try:
             # Record 5 seconds by default for MVP demo
             # Note: This runs on the server (user's Mac)
             filepath = mindscape_engine.record_audio(duration=5)
             
             state.is_recording = False
             state.is_processing = True
             state.status_message = "Processing Spectra..."
             yield
             
             # Transcribe
             transcript = mindscape_engine.transcribe_audio(filepath)
             state.transcript = transcript
             yield 
             
             # Diagnose
             result = mindscape_engine.get_diagnosis(transcript)
             
             # Robust Extraction
             bsv = result.get('bsv', {})
             state.bsv_valence = float(bsv.get('valence', 0.0))
             state.bsv_arousal = float(bsv.get('arousal', 0.0))
             state.bsv_dominance = float(bsv.get('dominance', 0.0))
             
             state.hypothesis_name = result['hypothesis'].get('name', 'Unknown')
             state.hypothesis_confidence = result['hypothesis'].get('confidence', '')
             state.hypothesis_reasoning = result.get('reasoning', '')
             state.follow_up_questions = result.get('follow_up', [])
             state.safety_gate = result.get('safety_gate', 'FAIL')
             
             evidence = result['hypothesis'].get('evidence', [])
             if isinstance(evidence, list):
                state.hypothesis_evidence = "\n- ".join(evidence)
             else:
                state.hypothesis_evidence = str(evidence)
                
             state.status_message = "Analysis Complete"
             
        except Exception as ex:
             state.status_message = f"Error: {str(ex)}"
             state.is_recording = False
        
        state.is_processing = False
        yield

def handle_upload(event: me.UploadEvent):
    state = me.state(State)
    state.is_processing = True
    state.status_message = "Uploading & Analyzing..."
    state.hypothesis_name = "Analyzing File..."
    state.transcript = ""
    yield
    
    # Save file
    temp_dir = "/tmp/mindscape_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    filepath = os.path.join(temp_dir, event.file.name)
    
    with open(filepath, "wb") as f:
        f.write(event.file.read())
    
    try:
         # Transcribe
         transcript = mindscape_engine.transcribe_audio(filepath)
         state.transcript = transcript
         state.status_message = "Diagnosing..."
         yield 
         
         # Diagnose
         result = mindscape_engine.get_diagnosis(transcript)
         
         # Robust Extraction
         bsv = result.get('bsv', {})
         state.bsv_valence = float(bsv.get('valence', 0.0))
         state.bsv_arousal = float(bsv.get('arousal', 0.0))
         state.bsv_dominance = float(bsv.get('dominance', 0.0))
         state.hypothesis_name = result['hypothesis'].get('name', 'Unknown')
         state.hypothesis_confidence = result['hypothesis'].get('confidence', '')
         state.hypothesis_reasoning = result.get('reasoning', '')
         state.follow_up_questions = result.get('follow_up', [])
         state.safety_gate = result.get('safety_gate', 'FAIL')
         
         evidence = result['hypothesis'].get('evidence', [])
         if isinstance(evidence, list):
            state.hypothesis_evidence = "\n- ".join(evidence)
         else:
            state.hypothesis_evidence = str(evidence)
            
         state.status_message = "Analysis Complete"
         
    except Exception as ex:
         state.status_message = f"Error: {str(ex)}"
         print(f"Upload Error: {ex}")
    
    state.is_processing = False
    yield

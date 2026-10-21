import pytesseract
# Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
import shutil
import pytesseract

# Find Tesseract automatically
tesseract_path = shutil.which("tesseract")

# Use the Windows installation path if needed
if tesseract_path is None:
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = tesseract_path
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tempfile
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas


def annotate_xsafenet_frame(frame, result):
    """Draw red hazard boxes, confidence and frame-level hazard risk."""
    import cv2

    severity_points = {
        0: 1.0,  # D00 - Longitudinal crack
        1: 1.5,  # D10 - Transverse crack
        2: 2.5,  # D20 - Alligator crack
        3: 4.0,  # D40 - Pothole
    }

    hazard_names = {
        0: "Longitudinal Crack",
        1: "Transverse Crack",
        2: "Alligator Crack",
        3: "Pothole",
    }

    annotated = frame.copy()
    frame_points = 0.0

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            # Use your project's RDD2022 class mapping.
            name = hazard_names.get(
                class_id,
                str(result.names.get(class_id, "Road Hazard"))
            )

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            # Red bounding box.
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            label = f"{name} | Confidence: {confidence * 100:.1f}%"

            # Solid red label background.
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2

            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            label_y = max(y1, text_h + 10)

            cv2.rectangle(
                annotated,
                (x1, label_y - text_h - 8),
                (x1 + text_w + 8, label_y + baseline),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

            # Confidence-adjusted hazard severity.
            base_points = severity_points.get(class_id, 0.0)
            frame_points += base_points * (
                0.50 + 0.50 * confidence
            )

    # Frame-level estimate, consistent with your existing
    # severity-point normalization. This is not accident probability.
    frame_risk = min(100.0, frame_points / 4.0 * 100.0)

    risk_text = f"Frame Road-Hazard Risk: {frame_risk:.1f}/100"

    cv2.rectangle(annotated, (8, 8), (440, 45), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        risk_text,
        (18, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return annotated, frame_risk


def render_reports_page():
    import json
    import uuid
    import re
    from datetime import datetime
    from pathlib import Path

    # ---------------- STORAGE ----------------

    report_dir = Path(__file__).resolve().parent / "reports_data"
    evidence_dir = report_dir / "evidence"
    reports_file = report_dir / "reports.json"

    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        st.error(f"Cannot create report folders: {error}")
        return

    # ---------------- WHITE THEME ----------------

    st.markdown("""
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0B1730 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #0B1730 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [aria-selected="true"],
    [data-testid="stSidebar"] button[aria-pressed="true"] {
        background-color: #19365D !important;
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

    st.write("Register a road hazard and save its details and evidence in the XSafeNet dashboard.")

    # ---------------- REPORT FORM ----------------

    with st.form("road_hazard_report_form"):

        st.markdown("### 1. Road Hazard Details")

        col1, col2 = st.columns(2)

        with col1:
            hazard_type = st.selectbox(
                "Hazard Type",
                [
                    "Pothole",
                    "Longitudinal Crack",
                    "Transverse Crack",
                    "Alligator Crack",
                    "Broken Road Surface",
                    "Waterlogging",
                    "Other"
                ]
            )

            severity = st.selectbox(
                "Severity Level",
                ["Low", "Moderate", "High", "Critical"],
                index=2
            )

        with col2:
            road_category = st.selectbox(
                "Road Category",
                [
                    "Municipal / City Road",
                    "State Highway",
                    "National Highway",
                    "Village / Panchayat Road",
                    "Other / Unknown"
                ]
            )

            observed_date = st.date_input(
                "Date Observed",
                value=datetime.now().date()
            )

        description = st.text_area(
            "Problem Description *",
            placeholder="Describe the damage and traffic impact..."
        )

        st.divider()
        st.markdown("### 2. Road Location")

        col1, col2 = st.columns(2)

        with col1:
            street_name = st.text_input("Street / Road Name *")
            area = st.text_input("Area / Locality *")
            city = st.text_input("City / Town *")

        with col2:
            district = st.text_input("District *")
            state = st.text_input("State *", value="Tamil Nadu")
            pincode = st.text_input("PIN Code *", max_chars=6)

        st.divider()
        st.markdown("### 3. Authority Details")

        authority = st.selectbox(
            "Responsible Road Authority",
            [
                "Municipal Corporation / Municipality",
                "State Highways / PWD",
                "NHAI / National Highway Authority",
                "Village Panchayat / Rural Development Department",
                "Authority Not Identified — Verification Required"
            ]
        )

        col1, col2 = st.columns(2)

        with col1:
            recipient_name = st.text_input("Recipient Officer Name *")
            department = st.text_input("Department / Office Name *")

        with col2:
            office_location = st.text_input("Office Location / Division")
            official_contact = st.text_input(
                "Official Email / Contact (Optional)"
            )

        st.divider()
        st.markdown("### 4. Road Damage Evidence")

        proof_image = st.file_uploader(
            "Upload Road Damage Image *",
            type=["jpg", "jpeg", "png"],
            key="report_proof_image"
        )

        st.divider()
        st.markdown("### 5. Reporter Details")

        col1, col2 = st.columns(2)

        with col1:
            reporter_name = st.text_input("Your Name *")
            reporter_phone = st.text_input("Contact Number *")

        with col2:
            reporter_email = st.text_input("Email Address (Optional)")
            confirm_details = st.checkbox(
                "I confirm that the information is accurate."
            )

        submitted = st.form_submit_button(
            "Submit Road Hazard Report",
            type="primary",
            use_container_width=True
        )

    # ---------------- IMAGE PREVIEW ----------------

    if proof_image is not None:
        st.image(
            proof_image,
            caption="Road damage evidence",
            use_container_width=True
        )

    # ---------------- VALIDATION AND SAVE ----------------

    if submitted:
        errors = []

        required = {
            "Problem description": description,
            "Street / road name": street_name,
            "Area / locality": area,
            "City / town": city,
            "District": district,
            "State": state,
            "Recipient name": recipient_name,
            "Department": department,
            "Reporter name": reporter_name,
            "Contact number": reporter_phone
        }

        for field, value in required.items():
            if not value.strip():
                errors.append(f"{field} is required.")

        if not re.fullmatch(r"\d{6}", pincode.strip()):
            errors.append("PIN code must contain exactly 6 digits.")

        phone_digits = re.sub(r"\D", "", reporter_phone)

        if not 10 <= len(phone_digits) <= 15:
            errors.append(
                "Enter a valid contact number (10–15 digits)."
            )

        if reporter_email.strip() and not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            reporter_email.strip()
        ):
            errors.append("Enter a valid reporter email.")

        if proof_image is None:
            errors.append("Upload a road damage image.")

        if not confirm_details:
            errors.append("Confirm the report information.")

        if errors:
            for error in errors:
                st.error(error)

        else:
            report_id = (
                f"XSN-{datetime.now().strftime('%Y%m%d')}-"
                f"{uuid.uuid4().hex[:6].upper()}"
            )

            image_extension = Path(proof_image.name).suffix.lower()
            image_name = f"{report_id}{image_extension}"
            image_path = evidence_dir / image_name

            report = {
                "report_id": report_id,
                "submitted_at": datetime.now().astimezone().isoformat(),
                "status": "Registered in XSafeNet - Pending Authority Dispatch",
                "hazard_type": hazard_type,
                "severity": severity,
                "road_category": road_category,
                "observed_date": observed_date.isoformat(),
                "description": description.strip(),
                "street_name": street_name.strip(),
                "area": area.strip(),
                "city": city.strip(),
                "district": district.strip(),
                "state": state.strip(),
                "pincode": pincode.strip(),
                "authority": authority,
                "recipient_name": recipient_name.strip(),
                "department": department.strip(),
                "office_location": office_location.strip(),
                "official_contact": official_contact.strip(),
                "reporter_name": reporter_name.strip(),
                "reporter_phone": reporter_phone.strip(),
                "reporter_email": reporter_email.strip(),
                "evidence_image": f"evidence/{image_name}"
            }

            try:
                if reports_file.exists():
                    with reports_file.open(
                        "r", encoding="utf-8"
                    ) as file:
                        reports = json.load(file)

                    if not isinstance(reports, list):
                        raise ValueError(
                            "reports.json must contain a list."
                        )
                else:
                    reports = []

                image_path.write_bytes(proof_image.getvalue())

                try:
                    reports.append(report)

                    temp_file = reports_file.with_name(
                        f"reports_{uuid.uuid4().hex}.tmp"
                    )

                    try:
                        with temp_file.open(
                            "w", encoding="utf-8"
                        ) as file:
                            json.dump(
                                reports,
                                file,
                                indent=2,
                                ensure_ascii=False
                            )

                        temp_file.replace(reports_file)

                    finally:
                        if temp_file.exists():
                            temp_file.unlink(missing_ok=True)

                except Exception:
                    image_path.unlink(missing_ok=True)
                    raise

                st.success(
                    "Road hazard report registered successfully!"
                )

                st.markdown("### Report Receipt")
                st.code(report_id)

                st.write(f"**Hazard:** {hazard_type}")
                st.write(f"**Severity:** {severity}")
                st.write(
                    f"**Location:** {street_name}, {area}, {city}, "
                    f"{district}, {state} - {pincode}"
                )
                st.write(f"**Authority:** {authority}")
                st.write(f"**Recipient:** {recipient_name}")
                st.write(f"**Department:** {department}")
                st.write(f"**Status:** {report['status']}")

                st.warning(
                    "This report is saved locally in XSafeNet. "
                    "It has not been transmitted to a government authority."
                )

            except (OSError, ValueError, json.JSONDecodeError) as error:
                st.error(f"Unable to save the report: {error}")

    # ---------------- PREVIOUS REPORTS ----------------

    st.divider()
    st.markdown("## Previously Registered Reports")

    try:
        if reports_file.exists():
            with reports_file.open("r", encoding="utf-8") as file:
                saved_reports = json.load(file)

            if not isinstance(saved_reports, list):
                raise ValueError("Invalid reports.json format.")

            if not saved_reports:
                st.caption("No reports registered yet.")

            for report in reversed(saved_reports):
                if not isinstance(report, dict):
                    continue

                with st.expander(
                    f"{report.get('report_id', 'Report')} | "
                    f"{report.get('hazard_type', 'Hazard')} | "
                    f"{report.get('city', 'Location')}"
                ):
                    st.write(
                        "**Status:**",
                        report.get("status", "N/A")
                    )
                    st.write(
                        "**Severity:**",
                        report.get("severity", "N/A")
                    )
                    st.write(
                        "**Description:**",
                        report.get("description", "N/A")
                    )
                    st.write(
                        "**Location:**",
                        f"{report.get('street_name', '')}, "
                        f"{report.get('area', '')}, "
                        f"{report.get('city', '')}, "
                        f"{report.get('district', '')}, "
                        f"{report.get('state', '')} - "
                        f"{report.get('pincode', '')}"
                    )
                    st.write(
                        "**Authority:**",
                        report.get("authority", "N/A")
                    )
                    st.write(
                        "**Recipient:**",
                        report.get("recipient_name", "N/A")
                    )

                    evidence = report.get("evidence_image", "")

                    if evidence:
                        saved_image = (report_dir / evidence).resolve()

                        if (
                            saved_image.is_relative_to(report_dir.resolve())
                            and saved_image.is_file()
                        ):
                            st.image(
                                str(saved_image),
                                caption="Saved road damage evidence",
                                use_container_width=True
                            )
                        else:
                            st.caption("Evidence image is unavailable.")

        else:
            st.caption("No reports registered yet.")

    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not load previous reports: {error}")



# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="XSafeNet | Smart Road Safety",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# IMAGE AND MODEL PATHS
# ============================================================
IMAGE_PATH = Path(
    r"C:\Users\Madhumitha\OneDrive\Pictures\logoxsafenetnew.jpeg"
)

MODEL_PATH = Path(
    r"C:\PROJECT\XSAFENET\runs\detect\train-3-extended\weights\best.pt"
)

HAZARD_LABELS = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole"
}

SEVERITY = {
    "Longitudinal Crack": 0.50,
    "Transverse Crack": 0.60,
    "Alligator Crack": 0.75,
    "Pothole": 1.00
}

SEVERITY_POINTS = {
    "Longitudinal Crack": 1.0,
    "Transverse Crack": 1.5,
    "Alligator Crack": 2.5,
    "Pothole": 4.0
}

# ============================================================
# LOAD YOLO MODEL
# ============================================================
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(str(model_path))

# ============================================================
# GLOBAL DESIGN — SIDEBAR VISIBILITY FIX
# ============================================================
st.markdown("""
<style>
    /* Hide only unnecessary Streamlit elements.
       Do NOT hide the header or toolbar because they contain
       the sidebar expand/collapse control. */
    #MainMenu, footer {
        visibility: hidden;
    }

    [data-testid="stHeader"] {
        visibility: visible !important;
        display: block !important;
        background: transparent !important;
    }

    [data-testid="stToolbar"] {
        visibility: visible !important;
    }

    /* Keep the sidebar visible when expanded */
    [data-testid="stSidebar"] {
        background: #0b1c35 !important;
        min-width: 260px;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    /* Keep the sidebar open/close controls accessible */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
    }

    .stApp {
        background: #f2f6fb;
    }

    .block-container {
        padding-top: 1.5rem;
        max-width: 1600px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        color: #0b1c35;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #62748c;
        margin-bottom: 20px;
    }

    .objective {
        background: white;
        padding: 18px;
        border-radius: 12px;
        border-left: 4px solid #1673e6;
        margin-bottom: 14px;
        color: #152b4d;
    }

    .stRadio label {
        color: white;
    }

    .upload-title {
        font-family: "Times New Roman", Times, serif;
        font-size: 36px;
        font-weight: bold;
        color: #102B50;
        margin-bottom: 5px;
    }

    .upload-subtitle {
        font-family: "Times New Roman", Times, serif;
        font-size: 17px;
        color: #52657D;
        margin-bottom: 25px;
    }

    .metric-card {
        background: white;
        border: 1px solid #DCE5EF;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 3px 12px rgba(16, 43, 80, 0.06);
    }

    .metric-label {
        color: #52657D;
        font-size: 16px;
    }

    .metric-value {
        color: #102B50;
        font-size: 28px;
        font-weight: bold;
    }

    .section-heading {
        color: #102B50;
        font-size: 23px;
        font-weight: bold;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)



def render_road_simulation():
    import streamlit as st

    st.markdown("""
    <style>
    /* Road Simulation typography */
    .stApp,
    .stApp p,
    .stApp label,
    .stApp input,
    .stApp textarea,
    .stApp button,
    .stApp select,
    .stApp li,
    .stApp [data-testid="stMarkdownContainer"] {
        font-family: "Times New Roman", Times, serif !important;
    }

    /* Dark navy page background */
    .stApp {
        background-color: #071426 !important;
        color: #F4F7FB !important;
    }

    /* Main text */
    .stApp p,
    .stApp li,
    .stApp label {
        color: #F4F7FB;
    }

    /* Headings */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: #FFFFFF !important;
        font-family: "Times New Roman", Times, serif !important;
    }

    /* Secondary text */
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p {
        color: #B9CCE0 !important;
    }

    /* Select box */
    .stApp [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #10243B !important;
        color: #FFFFFF !important;
        border-color: #42617F !important;
    }

    /* Simulation panels */
    .sim-header,
    .sim-panel {
        background: #10243B !important;
        color: #FFFFFF !important;
        border: 1px solid #42617F;
        border-radius: 12px;
        padding: 18px;
    }

    .sim-header h1,
    .sim-header h2,
    .sim-header h3,
    .sim-header p,
    .sim-panel h1,
    .sim-panel h2,
    .sim-panel h3,
    .sim-panel p {
        color: #FFFFFF !important;
    }

    .stAlert p {
        color: #FFFFFF !important;
    }

    .sim-road {
        background: #303947;
        border: 2px solid #637185;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        color: white;
        font-size: 18px;
        line-height: 2;
    }

    .sim-line {
        color: #F5D76E;
        letter-spacing: 9px;
        font-size: 23px;
    }

    .sim-small {
        color: #B9CCE0;
        font-size: 13px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.success("Road Simulation page is running!")

    

    speed = st.slider("Vehicle Speed (km/h)", 0, 120, 40)

    road = st.selectbox(
        "Road Condition",
        ["Good Road", "Cracks", "Potholes", "Wet Road"]
    )

    st.markdown(f"""
    <div class="sim-panel">
        <h3>Vehicle Status</h3>
        <p>Speed: {speed} km/h</p>
        <p>Road condition: {road}</p>
    </div>
    """, unsafe_allow_html=True)

    # Demo controls
    st.markdown("### ⚙️ Simulation Controls")

    col1, col2 = st.columns(2)

    with col1:
        speed = st.slider(
            "Vehicle speed (km/h)",
            min_value=0,
            max_value=100,
            value=40,
            step=5,
            key="road_demo_speed",
        )

    with col2:
        hazard = st.selectbox(
            "Road condition",
            [
                "Clear Road",
                "Longitudinal Crack",
                "Transverse Crack",
                "Alligator Cracks",
                "Pothole",
                "Wet Road + Pothole",
            ],
            key="road_demo_hazard",
        )

    severity = {
        "Clear Road": 0,
        "Longitudinal Crack": 20,
        "Transverse Crack": 30,
        "Alligator Cracks": 55,
        "Pothole": 70,
        "Wet Road + Pothole": 90,
    }

    hazard_score = severity[hazard]

    # Transparent illustrative demo calculation
    speed_score = min(100, speed)
    demo_risk = round(
        0.65 * hazard_score + 0.35 * speed_score
    )

    if hazard == "Clear Road" and speed <= 50:
        status = "LOW"
        instruction = "Continue safely and remain attentive."
    elif demo_risk >= 60 or (
        hazard != "Clear Road" and speed >= 60
    ):
        status = "HIGH"
        instruction = "Slow down and approach the hazard cautiously."
    elif demo_risk >= 30:
        status = "MODERATE"
        instruction = "Reduce speed and watch the road surface."
    else:
        status = "LOW"
        instruction = "Continue cautiously."

    # Demo road scene
    st.markdown("### 🛣️ Virtual Road Scene")

    road_icons = {
        "Clear Road": "✅",
        "Longitudinal Crack": "〰️",
        "Transverse Crack": "➖",
        "Alligator Cracks": "🕸️",
        "Pothole": "🕳️",
        "Wet Road + Pothole": "🌧️ 🕳️",
    }

    st.markdown(f"""
    <div class="sim-road">
        <div>ROAD CAMERA — SIMULATED VIEW</div>
        <div class="sim-line">│ │ │ │ │ │ │</div>
        <div style="font-size:42px">🚘</div>
        <div class="sim-line">│ │ │ │ │ │ │</div>
        <div style="font-size:34px">{road_icons[hazard]}</div>
        <div>{hazard}</div>
        <div class="sim-small">
            Illustrative scene only — not a real road image
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    st.markdown("### 📊 Simulation Results")

    c1, c2, c3 = st.columns(3)

    c1.metric("Vehicle Speed", f"{speed} km/h")
    c2.metric("Hazard Severity", f"{hazard_score}/100")
    c3.metric("Demo Risk Index", f"{demo_risk}/100")

    if status == "HIGH":
        st.error(f"🚨 {status} RISK — {instruction}")
    elif status == "MODERATE":
        st.warning(f"⚠️ {status} RISK — {instruction}")
    else:
        st.success(f"✅ {status} RISK — {instruction}")

    st.progress(demo_risk / 100)

    st.markdown("### 🔍 Why was this alert generated?")

    st.write(f"- Selected road condition: **{hazard}**")
    st.write(f"- Simulated vehicle speed: **{speed} km/h**")
    st.write(f"- Hazard contribution: **65%**")
    st.write(f"- Speed contribution: **35%**")
    st.code(
        f"Demo Risk = 0.65 × {hazard_score} "
        f"+ 0.35 × {speed_score} = {demo_risk}"
    )

    st.caption(
        "This is a rule-based demonstration with illustrative weights, "
        "not a trained prediction or validated safety score. "
        "It does not use live GPS, telemetry, weather or YOLO detections."
    )


# ============================================================
# PERFORMANCE MONITORING MODULE
# ============================================================
def render_performance_monitoring():
    import streamlit as st
    import psutil
    import pandas as pd
    import time
    from datetime import datetime

    # Initialize monitoring session
    if "pm_start_time" not in st.session_state:
        st.session_state["pm_start_time"] = time.time()

    if "pm_history" not in st.session_state:
        st.session_state["pm_history"] = []

    if "pm_logs" not in st.session_state:
        st.session_state["pm_logs"] = [
            {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Level": "INFO",
                "Message": "Performance monitoring initialized"
            }
        ]

    # Read actual system metrics
    cpu_usage = psutil.cpu_percent(interval=0.2)
    memory = psutil.virtual_memory()
    ram_usage = memory.percent

    process = psutil.Process()
    process_ram = process.memory_info().rss / (1024 * 1024)

    uptime = int(time.time() - st.session_state["pm_start_time"])
    uptime_text = (
        f"{uptime // 3600:02d}:"
        f"{(uptime % 3600) // 60:02d}:"
        f"{uptime % 60:02d}"
    )

    current_time = datetime.now().strftime("%H:%M:%S")

    # Keep recent monitoring samples
    history = st.session_state["pm_history"]
    history.append({
        "Time": current_time,
        "CPU (%)": cpu_usage,
        "RAM (%)": ram_usage
    })
    st.session_state["pm_history"] = history[-20:]

    # Page-specific dark theme
    st.markdown("""
    <style>
    .stApp {
        background-color: #071426;
        color: #F4F7FB;
    }

    .stApp h1, .stApp h2, .stApp h3 {
        color: #FFFFFF !important;
        font-family: "Times New Roman", Times, serif !important;
    }

    .stApp p, .stApp label {
        color: #E1EAF5;
        font-family: "Times New Roman", Times, serif !important;
    }

    .pm-card {
        background: #10243B;
        border: 1px solid #345574;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 10px;
    }

    .pm-label {
        color: #B9CCE0;
        font-size: 14px;
    }

    .pm-value {
        color: #00D4FF;
        font-size: 28px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # Page heading
    st.markdown("""
    <div class="pm-card">
        <h1>⚙️ XSafeNet Performance Monitoring</h1>
        <p>
        Runtime monitoring · Resource usage · Input validation · Error logs
        </p>
        <p class="pm-label">
        Live system readings with demonstration input checks
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------
    st.markdown("## 🖥️ System Status")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("CPU Usage", f"{cpu_usage:.1f}%")
    col2.metric("System RAM", f"{ram_usage:.1f}%")
    col3.metric("App RAM", f"{process_ram:.1f} MB")
    col4.metric("Session Runtime", uptime_text)

    if cpu_usage >= 85:
        st.warning("High CPU usage detected.")
    else:
        st.success("CPU usage is below the 85% warning threshold.")

    if ram_usage >= 85:
        st.warning("System memory usage is high.")
    else:
        st.success("System memory usage is below the 85% warning threshold.")

    st.caption(
        "CPU and system RAM are actual machine readings. "
        "App RAM is the memory used by the current Python process. "
        "Runtime measures this monitoring session, not the full computer uptime."
    )

    # --------------------------------------------------------
    # PERFORMANCE GRAPHS
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📈 Resource Usage History")

    history_df = pd.DataFrame(st.session_state["pm_history"])

    if not history_df.empty:
        chart_df = history_df.set_index("Time")[["CPU (%)", "RAM (%)"]]
        st.line_chart(chart_df, height=300)
    else:
        st.info("Collecting performance samples...")

    st.caption(
        "The graph stores up to 20 samples as this page is rerun. "
        "It is not a background monitoring service."
    )

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🧪 Input Quality Validation")

    st.write(
        "Test whether a filename has an accepted image or video extension."
    )

    filename = st.text_input(
        "Test filename",
        value="road_test.mp4",
        key="pm_test_filename"
    )

    allowed_extensions = (
        ".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"
    )

    clean_filename = filename.strip()
    valid_input = (
        bool(clean_filename)
        and clean_filename.lower().endswith(allowed_extensions)
    )

    if valid_input:
        st.success(
            f"VALID FORMAT — {clean_filename}"
        )
    else:
        st.error(
            "INVALID FORMAT — enter a filename ending in an accepted "
            "image or video extension."
        )

    st.caption(
        "This is a filename-format demonstration only. "
        "It does not verify that a file exists, opens correctly, "
        "or contains valid road footage."
    )

    
    # --------------------------------------------------------
    # RISK-STABILITY MONITOR
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📉 Risk-Stability Monitor")

    st.write(
        "Demonstration mode: enter sequential risk scores to "
        "inspect changes between consecutive observations."
    )

    threshold = st.slider(
        "Large-change review threshold (risk points)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        key="stability_threshold"
    )

    new_score = st.number_input(
        "Current risk score (0–100)",
        min_value=0,
        max_value=100,
        value=35,
        step=1,
        key="stability_current_score"
    )

    if st.button(
        "Record Risk Score",
        key="record_stability_score"
    ):
        if "risk_stability_history" not in st.session_state:
            st.session_state["risk_stability_history"] = []

        scores = st.session_state["risk_stability_history"]

        previous_score = scores[-1]["Risk Score"] if scores else None
        change = (
            abs(new_score - previous_score)
            if previous_score is not None
            else None
        )

        scores.append({
            "Observation": len(scores) + 1,
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Risk Score": int(new_score),
            "Absolute Change": change
        })

        # Keep the latest 50 observations
        st.session_state["risk_stability_history"] = scores[-50:]

    scores = st.session_state.get("risk_stability_history", [])

    if scores:
        latest = scores[-1]
        latest_change = latest["Absolute Change"]

        m1, m2, m3 = st.columns(3)

        m1.metric("Latest Risk Score", f'{latest["Risk Score"]}/100')

        m2.metric(
            "Change From Previous",
            "N/A" if latest_change is None
            else f"{latest_change} points"
        )

        if latest_change is None:
            stability_status = "NEED MORE DATA"
            m3.metric("Stability Status", stability_status)
            st.info("Record another score to calculate the change.")
        elif latest_change > threshold:
            stability_status = "REVIEW"
            m3.metric("Stability Status", stability_status)
            st.warning(
                f"Risk changed by {latest_change} points. "
                "Inspect the frame and input modalities to determine "
                "whether the change is expected."
            )
        else:
            stability_status = "WITHIN THRESHOLD"
            m3.metric("Stability Status", stability_status)
            st.success(
                "The latest score change is within the configured threshold."
            )

        stability_df = pd.DataFrame(scores)

        st.markdown("### Risk Score Trend")
        st.line_chart(
            stability_df.set_index("Observation")[["Risk Score"]],
            height=250
        )

        st.markdown("### Score Change History")
        st.dataframe(
            stability_df.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        if st.button(
            "Reset Risk History",
            key="reset_stability_history"
        ):
            st.session_state["risk_stability_history"] = []
            st.rerun()

    else:
        st.info(
            "No risk observations recorded yet. Enter a score and "
            "select Record Risk Score to begin."
        )

    st.caption(
        "This initial monitor uses manually entered scores. It does "
        "not yet consume live AEMRF-Net predictions. A large change "
        "is a review flag, not proof of a model error or road danger."
    )


    # --------------------------------------------------------
    # ERROR AND EVENT LOGS
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📋 Runtime Event Logs")

    log_col1, log_col2 = st.columns(2)

    with log_col1:
        if st.button("Add Test Warning", use_container_width=True):
            st.session_state["pm_logs"].append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Level": "WARNING",
                "Message": "Manually generated test warning"
            })

    with log_col2:
        if st.button("Clear Logs", use_container_width=True):
            st.session_state["pm_logs"] = []

    logs_df = pd.DataFrame(st.session_state["pm_logs"])

    if not logs_df.empty:
        st.dataframe(
            logs_df.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No events recorded in this session.")

    st.caption(
        "These are session-level demonstration logs. "
        "Unhandled Python exceptions are not automatically captured "
        "by this sample module."
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.title("🚘 XSafeNet")
    st.caption("Smarter Vision. Safer Roads.")
    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Upload Dataset",
            "Multimodal Detection",
            "Reports",
            "Road Simulation",
            "Performance Monitoring"
        ],
        key="selected_page"
    )

    st.divider()
    st.caption("Predict · Explain · Prevent")

# ============================================================
# DASHBOARD PAGE
# ============================================================
if selected_page == "Dashboard":

    if IMAGE_PATH.is_file():
        st.image(
            str(IMAGE_PATH),
            use_container_width=True
        )
    else:
        st.error(f"Image not found: {IMAGE_PATH}")

# ============================================================
# UPLOAD DATASET PAGE
# ============================================================
elif selected_page == "Upload Dataset":

    st.markdown("""
    <style>
        .block-container {
            padding-left: 2rem !important;
            padding-top: 1.5rem !important;
        }

        h1, h2, h3, h4, p, label {
            font-family: "Times New Roman", Times, serif;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="upload-title">Upload Dataset</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="upload-subtitle">'
        'Upload a road image to detect cracks and potholes '
        'using the XSafeNet YOLO model.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # ROAD IMAGE INPUT
    # --------------------------------------------------------
    st.markdown(
        '<div class="section-heading">Road Image Input</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a road image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear road image for hazard detection.",
        key="dataset_road_image"
    )

    # Reset previous image results when a different image is uploaded
    current_upload_id = (
        f"{uploaded_file.name}_{uploaded_file.size}"
        if uploaded_file is not None
        else None
    )

    if st.session_state.get("dataset_upload_id") != current_upload_id:
        st.session_state["dataset_upload_id"] = current_upload_id
        st.session_state.pop("xsafenet_annotated_image", None)
        st.session_state.pop("xsafenet_detections", None)

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.markdown("### Original Image")
            st.image(
                image,
                use_container_width=True
            )

        with right_col:
            st.markdown("### XSafeNet Detection Output")

            if "xsafenet_annotated_image" in st.session_state:
                st.image(
                    st.session_state["xsafenet_annotated_image"],
                    caption="Detected hazards highlighted in red",
                    use_container_width=True
                )
            else:
                st.info("Run hazard detection to see the output.")

        # ----------------------------------------------------
        # DETECT ROAD HAZARDS
        # ----------------------------------------------------
        if st.button(
            "Detect Road Hazards",
            type="primary",
            use_container_width=True,
            key="detect_dataset_hazards"
        ):

            if not MODEL_PATH.is_file():
                st.error(f"Model weights not found: {MODEL_PATH}")
                st.info(
                    "Check that best.pt exists in the "
                    "train-3-extended/weights folder."
                )

            else:
                try:
                    with st.spinner("Analyzing road image..."):

                        model = load_yolo_model(MODEL_PATH)

                        results = model.predict(
                            source=image_array,
                            imgsz=960,
                            conf=0.15,
                            verbose=False
                        )

                    result = results[0]
                    annotated_image = image_array.copy()
                    detections = []

                    if result.boxes is not None:
                        for box in result.boxes:

                            class_id = int(box.cls[0].item())
                            confidence = float(box.conf[0].item())

                            label = HAZARD_LABELS.get(
                                class_id,
                                model.names.get(
                                    class_id,
                                    f"Class {class_id}"
                                )
                            )

                            x1, y1, x2, y2 = map(
                                int,
                                box.xyxy[0].tolist()
                            )

                            # Red bounding box
                            cv2.rectangle(
                                annotated_image,
                                (x1, y1),
                                (x2, y2),
                                (230, 35, 45),
                                3
                            )

                            label_text = (
                                f"{label} {confidence * 100:.1f}%"
                            )

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.6
                            thickness = 2

                            (text_w, text_h), baseline = (
                                cv2.getTextSize(
                                    label_text,
                                    font,
                                    font_scale,
                                    thickness
                                )
                            )

                            label_y = max(y1, text_h + 12)

                            cv2.rectangle(
                                annotated_image,
                                (
                                    x1,
                                    label_y - text_h - 10
                                ),
                                (
                                    x1 + text_w + 10,
                                    label_y + baseline - 5
                                ),
                                (230, 35, 45),
                                -1
                            )

                            cv2.putText(
                                annotated_image,
                                label_text,
                                (x1 + 5, label_y - 5),
                                font,
                                font_scale,
                                (255, 255, 255),
                                thickness,
                                cv2.LINE_AA
                            )

                            detections.append({
                                "Hazard": label,
                                "Confidence (%)": round(
                                    confidence * 100, 2
                                ),
                                "Severity Weight": SEVERITY.get(
                                    label, 0.5
                                ),
                                "Bounding Box": (
                                    f"({x1}, {y1}, {x2}, {y2})"
                                )
                            })

                    st.session_state[
                        "xsafenet_annotated_image"
                    ] = annotated_image

                    st.session_state[
                        "xsafenet_detections"
                    ] = detections

                    st.rerun()

                except Exception as error:
                    st.error(f"Hazard detection failed: {error}")

    else:
        st.info("Upload a road image to begin hazard detection.")

    # --------------------------------------------------------
    # DETECTION SUMMARY
    # --------------------------------------------------------
    detections = st.session_state.get(
        "xsafenet_detections", []
    )

    if uploaded_file is not None and (
        "xsafenet_annotated_image" in st.session_state
    ):

        st.markdown("---")
        st.markdown(
            '<div class="section-heading">Detection Summary</div>',
            unsafe_allow_html=True
        )

        total_detections = len(detections)

        hazard_counts = {}

        for item in detections:
            hazard = item["Hazard"]
            hazard_counts[hazard] = (
                hazard_counts.get(hazard, 0) + 1
            )

        average_confidence = (
            round(
                sum(
                    item["Confidence (%)"]
                    for item in detections
                ) / total_detections,
                1
            )
            if total_detections > 0
            else 0.0
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Total Detections",
            total_detections
        )

        metric2.metric(
            "Hazard Types",
            len(hazard_counts)
        )

        metric3.metric(
            "Average Confidence",
            f"{average_confidence:.1f}%"
        )

        if detections:
            st.dataframe(
                pd.DataFrame(detections),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning(
                "No hazards were detected above the selected "
                "confidence threshold. This does not prove the "
                "road is safe."
            )

        # ----------------------------------------------------
        # DRIVER-SAFETY RISK ASSESSMENT
        # ----------------------------------------------------
        st.markdown("---")
        st.markdown(
            '<div class="section-heading">'
            'Driver-Safety Risk Assessment'
            '</div>',
            unsafe_allow_html=True
        )

        hazard_points = sum(
            SEVERITY_POINTS.get(item["Hazard"], 1.0)
            * (
                0.5
                + 0.5 * item["Confidence (%)"] / 100.0
            )
            for item in detections
        )

        water_pothole = st.checkbox(
            "Water is visible in one or more detected potholes",
            key="water_pothole_context"
        )

        water_adjustment = (
            10
            if water_pothole
            and hazard_counts.get("Pothole", 0) > 0
            else 0
        )

        risk_score = round(
            min(100, hazard_points * 12 + water_adjustment),
            1
        )

        if total_detections == 0:
            risk_category = "UNASSESSED"
        elif risk_score < 25:
            risk_category = "LOW"
        elif risk_score < 50:
            risk_category = "MODERATE"
        elif risk_score < 75:
            risk_category = "HIGH"
        else:
            risk_category = "CRITICAL"

        risk_col1, risk_col2 = st.columns(2)

        risk_col1.metric(
            "Preliminary Risk Score",
            f"{risk_score}/100"
            if total_detections > 0
            else "Not assessed"
        )

        risk_col2.metric(
            "Risk Category",
            risk_category
        )

        # ----------------------------------------------------
        # EXPLAINABLE AI
        # ----------------------------------------------------
        st.markdown("---")
        st.markdown(
            "### Explainable AI — Road Condition & Driver Safety"
        )

        if total_detections == 0:

            st.warning(
                "No hazards were detected above the selected "
                "confidence threshold. This does not confirm that "
                "the road is safe. Drive cautiously."
            )

        else:

            for hazard, count in hazard_counts.items():

                st.markdown(
                    f"#### {hazard}: {count} detected"
                )

                if hazard == "Pothole":

                    st.write(
                        "Potholes can cause tyre damage, vehicle "
                        "instability, or sudden steering and braking."
                    )

                    if water_pothole:
                        st.warning(
                            "Water may conceal the pothole's edges "
                            "and depth. The image alone cannot "
                            "establish its actual depth."
                        )

                    st.write(
                        "**Driver guidance:** Reduce speed gradually "
                        "when safe, increase following distance, and "
                        "avoid abrupt braking or swerving. Avoid "
                        "standing water when its depth is uncertain."
                    )

                elif hazard == "Longitudinal Crack":

                    st.write(
                        "Cracks along the road direction may indicate "
                        "pavement deterioration."
                    )

                    st.write(
                        "**Driver guidance:** Maintain a stable path "
                        "and watch for loose material or nearby damage."
                    )

                elif hazard == "Transverse Crack":

                    st.write(
                        "Cracks across the road may indicate pavement "
                        "stress or an uneven surface."
                    )

                    st.write(
                        "**Driver guidance:** Adjust speed to the "
                        "visible road condition and avoid sudden steering."
                    )

                elif hazard == "Alligator Crack":

                    st.write(
                        "Interconnected cracks may indicate extensive "
                        "pavement distress."
                    )

                    st.write(
                        "**Driver guidance:** Proceed cautiously and "
                        "watch for uneven surfaces or nearby potholes."
                    )

            st.markdown("#### Model Confidence and Limitations")

            st.write(
                f"Average detection confidence: "
                f"{average_confidence:.1f}%. This represents the "
                "model's confidence in its predicted classes, not "
                "the percentage of road safety."
            )

            if average_confidence < 30:

                st.warning(
                    "Detection confidence is low. Predictions may "
                    "be unreliable. Low confidence does not mean "
                    "low danger."
                )

            elif average_confidence < 60:

                st.info(
                    "Detection confidence is moderate. Some "
                    "classifications or hazard boundaries may "
                    "be uncertain."
                )

            else:

                st.info(
                    "The model reports relatively high confidence, "
                    "but this does not guarantee correct detection "
                    "or establish actual road-damage severity."
                )

        st.caption(
            "The risk score is a prototype heuristic. It does not "
            "measure pothole depth, damaged area, vehicle speed, "
            "braking distance, or actual accident probability. "
            "Validate its weights and thresholds against labelled "
            "road-condition data."
        )
# ============================================================
# MULTIMODAL DETECTION
# ============================================================
elif selected_page == "Multimodal Detection":

    st.markdown("""
    <style>
        .mm-title {
            font-size: 34px;
            font-weight: 800;
            color: #102B50;
        }
        .mm-subtitle {
            font-size: 16px;
            color: #62748c;
            margin-bottom: 20px;
        }
        .mm-card {
            background: white;
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #DCE5EF;
            box-shadow: 0 3px 12px rgba(16,43,80,.06);
        }
        .mm-label {
            color: #52657D;
            font-size: 14px;
        }
        .mm-value {
            color: #102B50;
            font-size: 28px;
            font-weight: 800;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="mm-title">Multimodal Detection</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="mm-subtitle">'
        'Video-based road hazard detection, weather risk, '
        'vehicle telemetry and explainable risk fusion.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Upload a road video to analyze road hazards. Select a clear digital speedometer crop "
        "to enable OCR-based telemetry. The final AEMRF-Net score combines available modalities "
        "and explains the largest weighted contributor."
    )

    # --------------------------------------------------------
    # Optional speedometer OCR support. Install the Python package and
    # the separate Tesseract executable to enable digital speedometer OCR.
    # --------------------------------------------------------
    import re
    try:
        import pytesseract
        # Use the common Windows install path when present; otherwise let
        # pytesseract use the executable available on PATH.
        _tesseract_exe = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if _tesseract_exe.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(_tesseract_exe)
        OCR_IMPORT_AVAILABLE = True
    except ImportError:
        pytesseract = None
        OCR_IMPORT_AVAILABLE = False

    def _read_speed_from_crop(frame, crop_xywh):
        """Return a plausible digital speedometer reading, or None."""
        if pytesseract is None or crop_xywh is None:
            return None
        x, y, w, h = [int(v) for v in crop_xywh]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Enlarge small digits and improve contrast for digital displays.
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        try:
            raw = pytesseract.image_to_string(binary, config=config)
        except Exception:
            return None
        matches = re.findall(r"\d{1,3}", raw)
        if not matches:
            return None
        # Use the first digit group; reject values outside a plausible bike range.
        value = int(matches[0])
        return value if 0 <= value <= 180 else None

    def _speed_risk_from_readings(avg_speed, max_speed, limit):
        """Transparent heuristic, not an accident probability."""
        if avg_speed is None or max_speed is None or limit <= 0:
            return None
        # No speed penalty below 80% of the supplied limit. Risk rises above it.
        avg_excess = max(0.0, avg_speed / limit - 0.80)
        peak_excess = max(0.0, max_speed / limit - 0.80)
        avg_component = min(100.0, avg_excess / 0.80 * 100.0)
        peak_component = min(100.0, peak_excess / 0.80 * 100.0)
        return round(min(100.0, 0.60 * avg_component + 0.40 * peak_component), 1)

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------
    st.markdown("### 1. Road Video Input")

    uploaded_video = st.file_uploader(
        "Upload road video",
        type=["mp4", "avi", "mov", "m4v"],
        key="multimodal_video"
    )

    col_weather, col_telemetry = st.columns(2, gap="large")

    with col_weather:
        st.markdown("### 2. Weather Context")

        rain_risk_input = st.slider(
            "Rain / precipitation severity",
            min_value=0,
            max_value=100,
            value=50,
            help="0 = no precipitation concern; "
                 "100 = extreme precipitation concern."
        )

        visibility_km = st.slider(
            "Visibility (km)",
            min_value=0.1,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

        wet_road = st.checkbox(
            "Wet road surface confirmed",
            value=False
        )

        # Transparent prototype weather-risk formula.
        visibility_risk = max(
            0.0,
            min(100.0, (1.0 - visibility_km / 10.0) * 100.0)
        )

        wet_road_risk = 100.0 if wet_road else 0.0

        weather_risk = round(
            0.50 * rain_risk_input
            + 0.30 * visibility_risk
            + 0.20 * wet_road_risk,
            1
        )

        st.metric("Calculated Weather Risk", f"{weather_risk}/100")

        st.caption(
            "Prototype formula: 50% precipitation severity + "
            "30% visibility risk + 20% wet-road indicator. "
            "These weights require validation. The inputs are "
            "user-provided, not automatically observed weather."
        )

 
    with col_telemetry:
        st.markdown("### 3. Vehicle Telemetry")
        st.caption(
            "Select the bike's digital speedometer using your mouse. "
            "The selected digital speedometer region is processed with OCR during video analysis."
        )

        # Initialize session state only once.
        if "speed_crop" not in st.session_state:
            st.session_state["speed_crop"] = None

        if "speed_crop_confirmed" not in st.session_state:
            st.session_state["speed_crop_confirmed"] = False

        if "speed_video_id" not in st.session_state:
            st.session_state["speed_video_id"] = None

        speed_limit = st.number_input(
            "Speed limit (km/h)",
            min_value=5,
            max_value=150,
            value=40,
            step=5,
            key="speed_limit_ocr",
        )

        if uploaded_video is None:
            st.info(
                "Upload your bike video in the Road Video Input section first."
            )

            # Clear the previous selection when no video is available.
            st.session_state["speed_video_id"] = None
            st.session_state["speed_crop"] = None
            st.session_state["speed_crop_confirmed"] = False

        else:
            # Identify the currently uploaded video.
            current_video_id = (
                f"{uploaded_video.name}_{uploaded_video.size}"
            )

            # Reset selection only when the uploaded video changes.
            if st.session_state["speed_video_id"] != current_video_id:
                st.session_state["speed_video_id"] = current_video_id
                st.session_state["speed_crop"] = None
                st.session_state["speed_crop_confirmed"] = False

            preview_cap = None
            preview_path = None

            try:
                # Save the uploaded video temporarily for OpenCV.
                suffix = Path(uploaded_video.name).suffix or ".mp4"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp_video:
                    temp_video.write(uploaded_video.getvalue())
                    preview_path = temp_video.name

                preview_cap = cv2.VideoCapture(preview_path)

                if not preview_cap.isOpened():
                    st.error("Unable to open the uploaded video.")

                else:
                    # Read the first frame for speedometer selection.
                    ok, original_frame = preview_cap.read()

                    if not ok:
                        st.warning(
                            "Could not read a frame from the video. "
                            "Please upload a valid video file."
                        )

                    else:
                        original_height, original_width = (
                            original_frame.shape[:2]
                        )

                        # Resize the preview without enlarging small frames.
                        max_preview_width = 800
                        preview_scale = min(
                            1.0,
                            max_preview_width / original_width,
                        )

                        preview_width = max(
                            1, int(original_width * preview_scale)
                        )
                        preview_height = max(
                            1, int(original_height * preview_scale)
                        )

                        preview_frame = cv2.resize(
                            original_frame,
                            (preview_width, preview_height),
                        )

                        preview_rgb = cv2.cvtColor(
                            preview_frame,
                            cv2.COLOR_BGR2RGB,
                        )

                        preview_image = Image.fromarray(preview_rgb)

                        st.markdown("#### Select the speedometer area")
                        st.write(
                            "Drag a rectangle around the digital speed "
                            "digits. Include the digits clearly, but avoid "
                            "unnecessary dashboard areas."
                        )

                        # Use a video-specific key so a new video gets
                        # a fresh drawing canvas.
                        canvas_key = (
                            "bike_speedometer_canvas_"
                            + str(abs(hash(current_video_id)))
                        )

                        canvas_result = st_canvas(
                            fill_color="rgba(30, 136, 229, 0.18)",
                            stroke_width=3,
                            stroke_color="#1E88E5",
                            background_image=preview_image,
                            update_streamlit=True,
                            height=preview_height,
                            width=preview_width,
                            drawing_mode="rect",
                            key=canvas_key,
                        )

                        drawing = canvas_result.json_data
                        objects = (
                            drawing.get("objects", [])
                            if drawing is not None
                            else []
                        )

                        # Use the latest rectangle drawn on the canvas.
                        if objects:
                            selected = objects[-1]

                            left = float(selected.get("left", 0))
                            top = float(selected.get("top", 0))
                            rect_width = float(
                                selected.get("width", 0)
                            )
                            rect_height = float(
                                selected.get("height", 0)
                            )
                            scale_x = float(
                                selected.get("scaleX", 1)
                            )
                            scale_y = float(
                                selected.get("scaleY", 1)
                            )

                            # Rectangle boundaries in preview coordinates.
                            preview_x1 = max(0, int(left))
                            preview_y1 = max(0, int(top))

                            preview_x2 = min(
                                preview_width,
                                int(left + rect_width * scale_x),
                            )
                            preview_y2 = min(
                                preview_height,
                                int(top + rect_height * scale_y),
                            )

                            if (
                                preview_x2 > preview_x1
                                and preview_y2 > preview_y1
                            ):
                                # Convert preview coordinates to original
                                # video-frame coordinates.
                                original_x1 = max(
                                    0,
                                    min(
                                        original_width - 1,
                                        int(preview_x1 / preview_scale),
                                    ),
                                )
                                original_y1 = max(
                                    0,
                                    min(
                                        original_height - 1,
                                        int(preview_y1 / preview_scale),
                                    ),
                                )
                                original_x2 = max(
                                    original_x1 + 1,
                                    min(
                                        original_width,
                                        int(
                                            preview_x2 / preview_scale
                                        ),
                                    ),
                                )
                                original_y2 = max(
                                    original_y1 + 1,
                                    min(
                                        original_height,
                                        int(
                                            preview_y2 / preview_scale
                                        ),
                                    ),
                                )

                                new_crop = (
                                    original_x1,
                                    original_y1,
                                    original_x2 - original_x1,
                                    original_y2 - original_y1,
                                )

                                # Reset confirmation only if the user
                                # has changed the selected rectangle.
                                previous_crop = st.session_state.get(
                                    "speed_crop"
                                )

                                if new_crop != previous_crop:
                                    st.session_state["speed_crop"] = (
                                        new_crop
                                    )
                                    st.session_state[
                                        "speed_crop_confirmed"
                                    ] = False

                                # Preview crop using resized-frame coordinates.
                                selected_crop = preview_frame[
                                    preview_y1:preview_y2,
                                    preview_x1:preview_x2,
                                ]

                                if selected_crop.size > 0:
                                    st.image(
                                        cv2.cvtColor(
                                            selected_crop,
                                            cv2.COLOR_BGR2RGB,
                                        ),
                                        caption=(
                                            "Selected speedometer preview"
                                        ),
                                        width=350,
                                    )

                                    st.success(
                                        "Speedometer region selected. "
                                        "Confirm it below."
                                    )

                            else:
                                st.warning(
                                    "The rectangle is too small. "
                                    "Draw a larger region around the digits."
                                )

                        elif st.session_state.get("speed_crop") is None:
                            st.info(
                                "Draw a rectangle around the speedometer "
                                "to select the reading area."
                            )

            except Exception as error:
                st.error(
                    f"Speedometer selection failed: {error}"
                )

            finally:
                if preview_cap is not None:
                    preview_cap.release()

                if (
                    preview_path is not None
                    and Path(preview_path).exists()
                ):
                    Path(preview_path).unlink(missing_ok=True)

            # Keep the confirmation checkbox outside the canvas-object
            # block so it remains available during Streamlit reruns.
            if st.session_state.get("speed_crop") is not None:
                st.checkbox(
                    "Confirm this is the correct speedometer area",
                    key="speed_crop_confirmed",
                )

            if st.session_state.get("speed_crop_confirmed", False):
                st.success(
                    "Speedometer crop confirmed. OCR will run on sampled video frames."
                )
                st.caption(
                    f"Selected video coordinates (x, y, width, height): "
                    f"{st.session_state['speed_crop']}"
                )
            else:
                st.warning(
                    "Select the speedometer and confirm the region "
                    "before using it for speed extraction."
                )
    # --------------------------------------------------------
    # VIDEO ANALYSIS
    # --------------------------------------------------------
    st.markdown("### 4. Full Video Hazard Analysis")

    confidence_threshold = st.slider(
        "YOLO confidence threshold",
        min_value=0.10,
        max_value=0.80,
        value=0.25,
        step=0.05,
        help="Higher thresholds may reduce weak detections "
             "but can miss genuine hazards."
    )
    # Default telemetry values available to the whole page
    telemetry_available = False
    telemetry_risk = None
    average_speed = None
    maximum_speed = None
    average_acceleration = None

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button(
            "Analyze Video with XSafeNet",
            type="primary",
            use_container_width=True,
            key="run_multimodal_analysis"
        ):

            if not MODEL_PATH.is_file():
                st.error(f"YOLO weights not found: {MODEL_PATH}")

            else:
                input_path = None
                output_path = None
                cap = None
                writer = None

                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uploaded_video.name).suffix
                    ) as temp_input:
                        temp_input.write(uploaded_video.getbuffer())
                        input_path = temp_input.name

                    cap = cv2.VideoCapture(input_path)

                    if not cap.isOpened():
                        raise ValueError(
                            "The uploaded video could not be opened."
                        )

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(
                        cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    )

                    if (
                        width <= 0 or height <= 0
                        or not np.isfinite(fps) or fps <= 0
                    ):
                        raise ValueError(
                            "Video dimensions or frame rate are invalid."
                        )

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4"
                    ) as temp_output:
                        output_path = temp_output.name

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    writer = cv2.VideoWriter(
                        output_path,
                        fourcc,
                        fps,
                        (width, height)
                    )

                    if not writer.isOpened():
                        raise ValueError(
                            "Could not initialize annotated video writer."
                        )

                    model = load_yolo_model(MODEL_PATH)

                    progress = st.progress(0)
                    status = st.empty()

                    frame_count = 0
                    all_confidences = []
                    hazard_counts = {}
                    per_frame_risks = []
                    detection_rows = []
                    speed_readings = []
                    speed_sample_count = 0
                    speed_ocr_error = None
                    crop_for_ocr = (
                        st.session_state.get("speed_crop")
                        if st.session_state.get("speed_crop_confirmed", False)
                        else None
                    )
                    ocr_step = max(1, int(round(fps)))
                    if crop_for_ocr is not None and OCR_IMPORT_AVAILABLE:
                        try:
                            pytesseract.get_tesseract_version()
                        except Exception:
                            speed_ocr_error = (
                                "The Tesseract OCR executable was not found. Install Tesseract OCR "
                                "for Windows, then restart the Streamlit app. The Python package alone is not enough."
                            )

                    severity_points = {
                        "Longitudinal Crack": 1.0,
                        "Transverse Crack": 1.5,
                        "Alligator Crack": 2.5,
                        "Pothole": 4.0
                    }

                    while True:
                        ok, frame = cap.read()

                        if not ok:
                            break

                        # Read the selected digital speedometer at a limited rate.
                        if crop_for_ocr is not None and frame_count % ocr_step == 0:
                            speed_sample_count += 1
                            if OCR_IMPORT_AVAILABLE and not speed_ocr_error:
                                speed_value = _read_speed_from_crop(frame, crop_for_ocr)
                                if speed_value is not None:
                                    speed_readings.append(float(speed_value))
                            elif not OCR_IMPORT_AVAILABLE:
                                speed_ocr_error = (
                                    "pytesseract is not installed. Install it with "
                                    "python -m pip install pytesseract, then install "
                                    "the Tesseract OCR application separately."
                                )

                        predictions = model.predict(
                            source=frame,
                            imgsz=640,
                            conf=confidence_threshold,
                            verbose=False
                        )

                        result = predictions[0]
                        annotated = frame.copy()
                        frame_points = 0.0

                        if result.boxes is not None:
                            for box in result.boxes:
                                class_id = int(box.cls[0].item())
                                confidence = float(
                                    box.conf[0].item()
                                )

                                # Use the trained model's class name
                                # when available; the mapping is a fallback.
                                model_names = model.names
                                model_label = (
                                    model_names.get(class_id)
                                    if isinstance(model_names, dict)
                                    else (
                                        model_names[class_id]
                                        if class_id < len(model_names)
                                        else None
                                    )
                                )

                                label = HAZARD_LABELS.get(
                                    class_id,
                                    model_label or f"Class {class_id}"
                                )

                                x1, y1, x2, y2 = map(
                                    int,
                                    box.xyxy[0].tolist()
                                )

                                cv2.rectangle(
                                    annotated,
                                    (x1, y1),
                                    (x2, y2),
                                    (0, 0, 255),
                                    3
                                )

                                label_text = (
                                    f"{label} {confidence * 100:.1f}%"
                                )

                                font = cv2.FONT_HERSHEY_SIMPLEX
                                scale = 0.6
                                thickness = 2

                                (tw, th), baseline = cv2.getTextSize(
                                    label_text,
                                    font,
                                    scale,
                                    thickness
                                )

                                text_y = max(y1, th + 12)

                                cv2.rectangle(
                                    annotated,
                                    (x1, text_y - th - 10),
                                    (
                                        x1 + tw + 10,
                                        text_y + baseline - 5
                                    ),
                                    (0, 0, 255),
                                    -1
                                )

                                cv2.putText(
                                    annotated,
                                    label_text,
                                    (x1 + 5, text_y - 5),
                                    font,
                                    scale,
                                    (255, 255, 255),
                                    thickness,
                                    cv2.LINE_AA
                                )

                                all_confidences.append(confidence)

                                hazard_counts[label] = (
                                    hazard_counts.get(label, 0) + 1
                                )

                                severity = severity_points.get(
                                    label, 1.0
                                )

                                frame_points += severity * (0.50 + 0.50 * confidence)

                                detection_rows.append({
                                    "Frame": frame_count + 1,
                                    "Time (s)": round(
                                        frame_count / fps, 3
                                    ),
                                    "Hazard": label,
                                    "Confidence (%)": round(
                                        confidence * 100, 2
                                    ),
                                    "Severity Weight": severity,
                                    "Bounding Box": (
                                        f"({x1}, {y1}, {x2}, {y2})"
                                    )
                                })

                        # Per-frame risk: normalize weighted hazard
                        # points against four pothole-equivalent points.
                        frame_risk = min(
                            100.0,
                            frame_points / 4.0 * 100.0
                        )
                        per_frame_risks.append(frame_risk)

                        writer.write(annotated)
                        frame_count += 1

                        if frame_count % 5 == 0:
                            if total_frames > 0:
                                progress.progress(
                                    min(
                                        1.0,
                                        frame_count / total_frames
                                    )
                                )

                            status.text(
                                f"Analyzing frame {frame_count}"
                                + (
                                    f" of {total_frames}"
                                    if total_frames > 0
                                    else ""
                                )
                            )

                    cap.release()
                    cap = None
                    writer.release()
                    writer = None

                    progress.progress(1.0)
                    status.success(
                        f"Video analysis completed: {frame_count} frames."
                    )

                    if frame_count == 0:
                        raise ValueError(
                            "No readable frames were found."
                        )

                    average_confidence = (
                        float(np.mean(all_confidences)) * 100.0
                        if all_confidences
                        else 0.0
                    )

                    # Hazard risk uses detected frames rather than diluting
                    # severe damage across every empty frame in the video.
                    positive_frame_risks = [r for r in per_frame_risks if r > 0]
                    if positive_frame_risks:
                        mean_positive = float(np.mean(positive_frame_risks))
                        peak_positive = float(np.max(positive_frame_risks))
                        detection_coverage = len(positive_frame_risks) / max(1, frame_count)
                        persistence_factor = 0.70 + 0.30 * min(1.0, detection_coverage * 5.0)
                        hazard_risk = round(
                            min(100.0, (0.65 * mean_positive + 0.35 * peak_positive)
                                * persistence_factor), 1
                        )
                    else:
                        hazard_risk = 0.0

                    # Accept telemetry only when OCR produced several plausible
                    # readings; one accidental OCR number is not enough.
                    if len(speed_readings) >= 3:
                        average_speed = round(float(np.median(speed_readings)), 1)
                        maximum_speed = round(float(np.percentile(speed_readings, 90)), 1)
                        telemetry_risk = _speed_risk_from_readings(
                            average_speed, maximum_speed, float(speed_limit)
                        )
                        telemetry_available = telemetry_risk is not None
                    else:
                        average_speed = None
                        maximum_speed = None
                        telemetry_risk = None
                        telemetry_available = False

                    # AEMRF-Net heuristic fusion. Weights are renormalized
                    # when a modality is genuinely unavailable.
                    available_components = [("Road hazards", hazard_risk, 0.55),
                                            ("Weather", weather_risk, 0.20)]
                    if telemetry_available and telemetry_risk is not None:
                        available_components.append(("Vehicle speed", telemetry_risk, 0.25))
                    total_weight = sum(weight for _, _, weight in available_components)
                    aemrf_risk = round(sum(value * weight for _, value, weight in available_components)
                                       / total_weight, 1)
                    fusion_note = (
                        "Risk fusion includes road hazards, weather and validated speedometer OCR."
                        if telemetry_available else
                        "Risk fusion includes road hazards and user-provided weather. Speed was excluded because fewer than three reliable OCR readings were obtained."
                    )

                    aemrf_risk = round(
                        min(100.0, max(0.0, aemrf_risk)),
                        1
                    )

                    if not all_confidences:
                        confidence_status = "NO DETECTIONS"
                    elif average_confidence < 30:
                        confidence_status = "LOW"
                    elif average_confidence < 60:
                        confidence_status = "MODERATE"
                    else:
                        confidence_status = "RELATIVELY HIGH"

                    if aemrf_risk < 25:
                        risk_category = "LOW"
                    elif aemrf_risk < 50:
                        risk_category = "MODERATE"
                    elif aemrf_risk < 75:
                        risk_category = "HIGH"
                    else:
                        risk_category = "CRITICAL"

                    # Save analysis results across Streamlit reruns.
                    st.session_state["mm_output_path"] = output_path
                    st.session_state["mm_hazard_risk"] = hazard_risk
                    st.session_state["mm_weather_risk"] = weather_risk
                    st.session_state["mm_aemrf_risk"] = aemrf_risk
                    st.session_state["mm_avg_conf"] = average_confidence
                    st.session_state["mm_hazard_counts"] = hazard_counts
                    st.session_state["mm_detection_rows"] = detection_rows
                    st.session_state["mm_frame_count"] = frame_count
                    st.session_state["mm_fps"] = fps
                    st.session_state["mm_category"] = risk_category
                    st.session_state["mm_conf_status"] = confidence_status
                    st.session_state["mm_fusion_note"] = fusion_note
                    st.session_state["mm_telemetry_available"] = (
                        telemetry_available
                    )
                    st.session_state["mm_telemetry_risk"] = telemetry_risk
                    st.session_state["mm_average_speed"] = average_speed
                    st.session_state["mm_maximum_speed"] = maximum_speed
                    st.session_state["mm_speed_readings"] = speed_readings
                    st.session_state["mm_speed_sample_count"] = speed_sample_count
                    st.session_state["mm_speed_ocr_error"] = speed_ocr_error
                    st.session_state["mm_speed_limit"] = float(speed_limit)
                    st.session_state["mm_weather_inputs"] = {
                        "rain": rain_risk_input,
                        "visibility_km": visibility_km,
                        "wet_road": wet_road,
                    }

                    st.rerun()

                except Exception as error:
                    st.error(f"Multimodal analysis failed: {error}")

                finally:
                    if cap is not None:
                        cap.release()

                    if writer is not None:
                        writer.release()

                    if input_path and Path(input_path).exists():
                        Path(input_path).unlink(missing_ok=True)


    
    # ============================================================
    # XSafeNet — MULTIMODAL RESULTS DASHBOARD
    # Display saved results after video analysis
    # ============================================================

    import html

    if st.session_state.get("mm_aemrf_risk") is not None:

        # ---------------- SAVED RESULTS ----------------
        hazard_risk = float(
            st.session_state.get("mm_hazard_risk", 0)
        )
        weather_risk = float(
            st.session_state.get("mm_weather_risk", 0)
        )
        aemrf_risk = float(
            st.session_state.get("mm_aemrf_risk", 0)
        )
        average_confidence = float(
            st.session_state.get("mm_avg_conf", 0)
        )

        hazard_counts = st.session_state.get(
            "mm_hazard_counts", {}
        )
        telemetry_available = st.session_state.get(
            "mm_telemetry_available", False
        )
        telemetry_risk = st.session_state.get(
            "mm_telemetry_risk"
        )
        risk_category = st.session_state.get(
            "mm_category", "UNKNOWN"
        )
        fusion_note = st.session_state.get(
            "mm_fusion_note", ""
        )
        weather_inputs = st.session_state.get(
            "mm_weather_inputs", {}
        )

        total_hazards = sum(hazard_counts.values())

        average_speed = st.session_state.get(
            "mm_average_speed"
        )
        maximum_speed = st.session_state.get(
            "mm_maximum_speed"
        )

        # ---------------- COLORS ----------------
        if aemrf_risk < 25:
            risk_color = "#16a34a"
            risk_icon = "🟢"
        elif aemrf_risk < 50:
            risk_color = "#eab308"
            risk_icon = "🟡"
        elif aemrf_risk < 75:
            risk_color = "#f97316"
            risk_icon = "🟠"
        else:
            risk_color = "#ef4444"
            risk_icon = "🔴"

        if hazard_risk < 25:
            hazard_color = "#16a34a"
        elif hazard_risk < 50:
            hazard_color = "#eab308"
        elif hazard_risk < 75:
            hazard_color = "#f97316"
        else:
            hazard_color = "#ef4444"

        if weather_risk < 25:
            weather_color = "#16a34a"
        elif weather_risk < 50:
            weather_color = "#eab308"
        elif weather_risk < 75:
            weather_color = "#f97316"
        else:
            weather_color = "#ef4444"

        if telemetry_available and telemetry_risk is not None:
            speed_color = (
                "#ef4444" if telemetry_risk >= 50
                else "#f97316" if telemetry_risk >= 20
                else "#16a34a"
            )
            speed_risk_text = f"{float(telemetry_risk):.1f}/100"
        else:
            speed_color = "#64748b"
            speed_risk_text = "Unavailable"

        # ---------------- WEATHER DESCRIPTION ----------------
        rain_value = weather_inputs.get("rain")
        visibility = weather_inputs.get("visibility_km")
        wet_road = weather_inputs.get("wet_road", False)

        weather_conditions = []

        if wet_road:
            weather_conditions.append("Wet road reported")

        if visibility is not None:
            if float(visibility) < 2:
                weather_conditions.append("Very low visibility")
            elif float(visibility) < 5:
                weather_conditions.append("Reduced visibility")
            else:
                weather_conditions.append("Good visibility")

        if rain_value is not None:
            try:
                rain_number = float(rain_value)
                if rain_number >= 60:
                    weather_conditions.append("High rainfall risk")
                elif rain_number >= 30:
                    weather_conditions.append("Moderate rainfall risk")
                else:
                    weather_conditions.append("Low rainfall risk")
            except (TypeError, ValueError):
                pass

        weather_description = (
            " • ".join(weather_conditions)
            if weather_conditions
            else "Weather inputs not available"
        )

        # ---------------- DASHBOARD HEADER ----------------
        st.markdown("""
        <div style="
            background:linear-gradient(120deg,#0b1730,#12345a);
            padding:24px;
            border-radius:18px;
            border:1px solid #24476d;
            margin:10px 0 22px 0;">
            <div style="font-size:13px;color:#93c5fd;
                        letter-spacing:2px;font-weight:700;">
                XSAFENET • INTELLIGENT ROAD SAFETY
            </div>
            <div style="font-size:28px;color:white;
                        font-weight:800;margin-top:7px;">
                Multimodal Analysis Results
            </div>
            <div style="font-size:14px;color:#cbd5e1;margin-top:7px;">
                Road hazards · Speedometer telemetry · Weather · Explainable AI
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Explain telemetry availability instead of silently showing a dash.
        if telemetry_available and average_speed is not None:
            st.success(
                f"Speedometer OCR accepted {len(st.session_state.get('mm_speed_readings', []))} readings. "
                f"Median speed: {average_speed:.1f} km/h; 90th-percentile speed: "
                f"{float(maximum_speed):.1f} km/h."
            )
        else:
            ocr_error_msg = st.session_state.get("mm_speed_ocr_error")
            if ocr_error_msg:
                st.warning(str(ocr_error_msg))
            else:
                st.warning(
                    "Vehicle telemetry unavailable: at least 3 plausible OCR readings are required. "
                    "Confirm the crop contains clear digital digits and ensure Tesseract OCR is installed."
                )

        # ---------------- METRIC CARDS ----------------
        st.markdown("""
        <style>
        .xs-card {
            background: #ffffff;
            border: 1px solid #dbe4ef;
            border-radius: 16px;
            padding: 17px 16px;
            min-height: 148px;
            box-shadow: 0 4px 14px rgba(15,23,42,0.05);
        }
        .xs-label {
            color: #64748b;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
        }
        .xs-value {
            font-size: 27px;
            font-weight: 800;
            margin: 13px 0 7px 0;
            line-height: 1.2;
        }
        .xs-sub {
            font-size: 12px;
            color: #64748b;
            line-height: 1.5;
        }
        .xs-panel {
            background: #ffffff;
            border: 1px solid #dbe4ef;
            border-radius: 16px;
            padding: 19px;
            margin: 8px 0 14px 0;
        }
        .xs-panel-title {
            color: #0f172a;
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .xs-explain {
            background: #f8fafc;
            border-left: 4px solid #3b82f6;
            padding: 13px 15px;
            border-radius: 9px;
            margin: 9px 0;
            color: #334155;
            font-size: 13px;
            line-height: 1.6;
        }
        </style>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4, gap="small")

        speed_display = (
            f"{float(average_speed):.1f}"
            if telemetry_available and average_speed is not None
            else "—"
        )

        with c1:
            st.markdown(f"""
            <div class="xs-card" style="border-top:4px solid {hazard_color};">
                <div class="xs-label">🕳️ Road Hazards</div>
                <div class="xs-value" style="color:{hazard_color};">
                    {total_hazards}
                </div>
                <div class="xs-sub">
                    Detections · Risk {hazard_risk:.1f}/100
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="xs-card" style="border-top:4px solid {speed_color};">
                <div class="xs-label">🏍️ Vehicle Speed</div>
                <div class="xs-value" style="color:{speed_color};">
                    {speed_display}
                    <span style="font-size:13px;">km/h</span>
                </div>
                <div class="xs-sub">
                    Speed risk: {speed_risk_text}<br>
                    Peak/90th percentile: {f"{float(maximum_speed):.1f}" if telemetry_available and maximum_speed is not None else "—"} km/h
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="xs-card" style="border-top:4px solid {weather_color};">
                <div class="xs-label">🌧️ Weather Risk</div>
                <div class="xs-value" style="color:{weather_color};">
                    {weather_risk:.1f}<span style="font-size:13px;">/100</span>
                </div>
                <div class="xs-sub">
                    Rainfall · Visibility · Wet-road inputs
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="xs-card" style="border-top:4px solid {risk_color};">
                <div class="xs-label">🛡️ AEMRF-Net Risk</div>
                <div class="xs-value" style="color:{risk_color};">
                    {aemrf_risk:.1f}<span style="font-size:13px;">/100</span>
                </div>
                <div class="xs-sub">
                    {risk_icon} {html.escape(risk_category)} risk
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ---------------- OVERALL RISK PANEL ----------------
        st.markdown(f"""
        <div class="xs-panel" style="border-left:5px solid {risk_color};">
            <div class="xs-panel-title">
                {risk_icon} AEMRF-Net Road-Safety Assessment
            </div>
            <div style="font-size:15px;color:#334155;margin-bottom:12px;">
                Combined risk score:
                <b style="color:{risk_color};">{aemrf_risk:.1f}/100</b>
                &nbsp;·&nbsp; Category:
                <b style="color:{risk_color};">{html.escape(risk_category)}</b>
            </div>
            <div style="background:#e2e8f0;border-radius:20px;height:12px;
                        overflow:hidden;">
                <div style="width:{aemrf_risk}%;height:12px;
                            background:{risk_color};border-radius:20px;"></div>
            </div>
            <div style="font-size:12px;color:#64748b;margin-top:10px;">
                This is a project risk indicator, not a calibrated probability
                of an accident.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- EXPLAINABLE AI ----------------
        # Calculate the largest weighted contributor before displaying the explanation.
        factor_scores = [("road hazards", hazard_risk, 0.55), ("weather", weather_risk, 0.20)]
        if telemetry_available and telemetry_risk is not None:
            factor_scores.append(("vehicle speed", float(telemetry_risk), 0.25))
        active_weight_sum = sum(w for _, _, w in factor_scores)
        factor_contributions = [
            (name, value * weight / active_weight_sum)
            for name, value, weight in factor_scores
        ]
        major_factor, major_contribution = max(factor_contributions, key=lambda item: item[1])

        st.markdown("## 🧠 Explainable AI — Why these risks?")
        st.info(
            f"**Main contributing factor: {major_factor.title()}**. "
            f"Its weighted contribution indicator is {major_contribution:.1f} points. "
            "This identifies the largest contributor in the configured formula; it does not prove accident causation."
        )

        hazard_reasons = []
        if total_hazards > 0:
            hazard_reasons.append(
                f"The detector identified {total_hazards} road-hazard "
                "instances in the analysed video."
            )
        else:
            hazard_reasons.append(
                "No hazards were detected. This does not prove the road is hazard-free."
            )

        hazard_reasons.append(
            f"The current hazard-risk score is {hazard_risk:.1f}/100."
        )

        speed_reasons = []
        if telemetry_available and telemetry_risk is not None:
            speed_reasons.append(
                f"The speedometer OCR produced readings; the current "
                f"telemetry-risk indicator is {float(telemetry_risk):.1f}/100."
            )
            if average_speed is not None:
                speed_reasons.append(
                    f"Average extracted speed: {float(average_speed):.1f} km/h."
                )
            if maximum_speed is not None:
                speed_reasons.append(
                    f"Peak extracted speed: {float(maximum_speed):.1f} km/h."
                )
        else:
            speed_reasons.append(
                "No reliable speedometer reading was available, so telemetry "
                "was excluded from the current risk fusion."
            )

        speed_readings_saved = st.session_state.get("mm_speed_readings", [])
        speed_sample_count_saved = int(st.session_state.get("mm_speed_sample_count", 0) or 0)
        speed_ocr_error_saved = st.session_state.get("mm_speed_ocr_error")
        if speed_ocr_error_saved:
            speed_reasons.append(str(speed_ocr_error_saved))
        elif speed_sample_count_saved and not telemetry_available:
            speed_reasons.append(
                f"OCR sampled {speed_sample_count_saved} frames but found only "
                f"{len(speed_readings_saved)} valid speed readings. Improve lighting, "
                "crop the digits more tightly, or use a clearer digital speedometer video."
            )
        elif telemetry_available:
            speed_reasons.append(
                f"The configured speed limit is {float(st.session_state.get('mm_speed_limit', 40)):.0f} km/h. "
                "Speed risk is a transparent heuristic based on average and high-percentile speed."
            )

        weather_reasons = [
            f"Weather-risk score: {weather_risk:.1f}/100.",
            weather_description + ".",
        ]

        if total_hazards:
            top_hazard = max(hazard_counts.items(), key=lambda item: item[1])[0]
            hazard_reasons.append(
                f"Most frequently detected class: {top_hazard} ({hazard_counts[top_hazard]} detections). "
                "Potholes and alligator cracks are assigned higher severity weights than ordinary cracks."
            )

        explanations = [
            (
                "🕳️ Road Hazard Explanation",
                hazard_color,
                " ".join(hazard_reasons),
            ),
            (
                "🏍️ Speedometer Explanation",
                speed_color,
                " ".join(speed_reasons),
            ),
            (
                "🌧️ Weather Explanation",
                weather_color,
                " ".join(weather_reasons),
            ),
            (
                "🛡️ AEMRF-Net Risk Explanation",
                risk_color,
                f"AEMRF-Net Risk is {aemrf_risk:.1f}/100 ({risk_category}). "
                f"The largest weighted contributor in this run is {major_factor} "
                f"(contribution indicator {major_contribution:.1f} points). "
                f"{fusion_note} This is a rule-based risk indicator, not a verified accident probability. "
                "The system cannot conclude that an accident will occur or prove causation from video alone.",
            ),
        ]

        explain_cols = st.columns(2, gap="medium")

        for index, (title, color, reason) in enumerate(explanations):
            with explain_cols[index % 2]:
                st.markdown(f"""
                <div class="xs-panel" style="border-top:4px solid {color};">
                    <div class="xs-panel-title">
                        {html.escape(title)}
                    </div>
                    <div style="font-size:13px;color:#334155;line-height:1.7;">
                        {html.escape(reason)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ---------------- DETECTION SUMMARY ----------------
        with st.expander("🔎 View detected hazard breakdown"):
            if hazard_counts:
                st.dataframe(
                    [
                        {"Hazard Type": str(name), "Detections": int(count)}
                        for name, count in hazard_counts.items()
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No hazard-count summary is available.")

            st.metric(
                "Average Detection Confidence",
                f"{average_confidence:.1f}%"
            )

        # ---------------- PER-DETECTION DETAILS ----------------
        with st.expander("📋 View individual hazard predictions", expanded=False):
            _detection_rows = st.session_state.get("mm_detection_rows", [])
            if _detection_rows:
                import pandas as pd
                st.dataframe(
                    _detection_rows,
                    use_container_width=True,
                    hide_index=True,
                )
                _csv_bytes = pd.DataFrame(_detection_rows).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download detection details (CSV)",
                    data=_csv_bytes,
                    file_name="XSafeNet_Hazard_Detections.csv",
                    mime="text/csv",
                    key="download_xsafenet_detection_csv",
                )
            else:
                st.info("No individual hazard predictions were recorded for this video.")

        # ---------------- PROCESSED VIDEO ----------------
        output_path = st.session_state.get("mm_output_path")

        if output_path and Path(output_path).exists():
            with st.expander("🎬 View analysed video"):
                st.video(output_path)
                try:
                    with open(output_path, "rb") as video_file:
                        annotated_video_bytes = video_file.read()
                    st.download_button(
                        label="⬇️ Download Annotated Video (MP4)",
                        data=annotated_video_bytes,
                        file_name="XSafeNet_Annotated_Road_Hazards.mp4",
                        mime="video/mp4",
                        key="download_xsafenet_annotated_video",
                        use_container_width=True,
                    )
                except OSError as download_error:
                    st.error(f"Could not prepare annotated video download: {download_error}")

        st.caption(
            "XSafeNet analysis is based on the inputs and detections available "
            "for this video. OCR errors, missed hazards and unavailable weather "
            "or telemetry data can affect the results."
        )
elif selected_page == "Reports":
    render_reports_page()

elif selected_page == "Road Simulation":
    render_road_simulation()

elif selected_page == "Performance Monitoring":
    render_performance_monitoring()
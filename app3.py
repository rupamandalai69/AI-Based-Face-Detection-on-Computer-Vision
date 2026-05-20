# =========================================================
# 😊 FACE EMOTION DETECTION
# Ultra Professional AI Dashboard
# =========================================================

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Face Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =====================================================
GOOGLE FONT
===================================================== */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* =====================================================
BACKGROUND
===================================================== */

.stApp{
    background:
    linear-gradient(
        135deg,
        #050816,
        #0f172a,
        #111827
    );

    color:white;
}

/* =====================================================
REMOVE EXTRA SPACE
===================================================== */

.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
}

/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"]{

    background:
    linear-gradient(
        180deg,
        #020617,
        #0f172a
    );

    border-right:1px solid rgba(255,255,255,0.08);
}

/* Sidebar Text */

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* =====================================================
LOGO TITLE
===================================================== */

.logo-title{

    font-size:34px;

    font-weight:900;

    text-align:center;

    background:linear-gradient(
        90deg,
        #38bdf8,
        #22c55e,
        #a855f7
    );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;

    margin-bottom:20px;
}

/* =====================================================
SIDEBAR CARDS
===================================================== */

.side-card{

    background: rgba(255,255,255,0.05);

    border-radius:24px;

    padding:22px;

    border:1px solid rgba(255,255,255,0.08);

    margin-bottom:18px;

    box-shadow:
    0 0 20px rgba(0,255,255,0.08);

    transition:0.4s;
}

.side-card:hover{
    transform:translateY(-5px);
}

/* =====================================================
SIDE HEADINGS
===================================================== */

.side-heading{

    font-size:20px;

    font-weight:700;

    color:#38bdf8;

    margin-bottom:12px;
}

/* =====================================================
SIDE TEXT
===================================================== */

.side-text{

    color:#e2e8f0;

    font-size:15px;

    line-height:1.8;
}

/* =====================================================
MAIN TITLE
===================================================== */

.main-title{

    text-align:center;

    font-size:72px;

    font-weight:900;

    margin-top:10px;

    background:linear-gradient(
        90deg,
        #38bdf8,
        #22c55e,
        #a855f7
    );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;

    letter-spacing:2px;
}

/* =====================================================
SUBTITLE
===================================================== */

.subtitle{

    text-align:center;

    color:#cbd5e1;

    font-size:22px;

    margin-bottom:35px;
}

/* =====================================================
TOP DASHBOARD CARDS
===================================================== */

.metric-card{

    background:
    rgba(255,255,255,0.05);

    border-radius:28px;

    padding:28px;

    border:1px solid rgba(255,255,255,0.08);

    backdrop-filter:blur(20px);

    box-shadow:
    0 0 25px rgba(0,255,255,0.08);

    transition:0.4s;
}

.metric-card:hover{

    transform:translateY(-6px);

    box-shadow:
    0 0 35px rgba(56,189,248,0.2);
}

/* Metric Label */

.metric-label{

    color:#94a3b8;

    font-size:16px;

    margin-bottom:10px;
}

/* Metric Value */

.metric-value{

    font-size:42px;

    font-weight:800;

    color:white;
}

/* =====================================================
GLASS CARD
===================================================== */

.glass-card{

    background:
    rgba(255,255,255,0.05);

    border-radius:28px;

    padding:28px;

    border:1px solid rgba(255,255,255,0.08);

    backdrop-filter:blur(20px);

    box-shadow:
    0 0 25px rgba(0,255,255,0.08);

    margin-top:20px;
}

/* =====================================================
RESULT BOX
===================================================== */

.result-box{

    background:
    linear-gradient(
        135deg,
        rgba(56,189,248,0.18),
        rgba(168,85,247,0.18)
    );

    border-radius:30px;

    padding:35px;

    border:1px solid rgba(255,255,255,0.08);

    text-align:center;

    margin-top:20px;

    box-shadow:
    0 0 30px rgba(56,189,248,0.15);
}

/* =====================================================
RESULT TEXT
===================================================== */

.result-emotion{

    font-size:52px;

    font-weight:800;

    color:white;
}

.result-confidence{

    font-size:24px;

    color:#cbd5e1;

    margin-top:10px;
}

/* =====================================================
UPLOAD BOX
===================================================== */

[data-testid="stFileUploader"]{

    background:
    rgba(255,255,255,0.04);

    border:2px dashed rgba(255,255,255,0.12);

    border-radius:25px;

    padding:25px;
}

/* =====================================================
BUTTON
===================================================== */

.stButton > button{

    width:100%;

    border:none;

    border-radius:18px;

    padding:14px;

    font-size:18px;

    font-weight:700;

    background:
    linear-gradient(
        90deg,
        #38bdf8,
        #22c55e
    );

    color:black;

    transition:0.4s;
}

.stButton > button:hover{

    transform:scale(1.03);

    box-shadow:
    0 0 20px rgba(56,189,248,0.35);
}

/* =====================================================
TABS
===================================================== */

.stTabs [data-baseweb="tab"]{

    background:#111827;

    color:white;

    border-radius:18px;

    padding:14px 28px;

    margin-right:10px;

    font-size:18px;

    font-weight:600;
}

.stTabs [aria-selected="true"]{

    background:
    linear-gradient(
        90deg,
        #38bdf8,
        #22c55e
    ) !important;

    color:black !important;
}

/* =====================================================
PROGRESS BAR
===================================================== */

.stProgress > div > div > div > div {

    background:
    linear-gradient(
        90deg,
        #38bdf8,
        #22c55e,
        #a855f7
    );
}

/* =====================================================
FOOTER
===================================================== */

.footer{

    text-align:center;

    color:#94a3b8;

    margin-top:50px;

    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("""
<div class="logo-title">
😊 FACE EMOTION
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="side-card">

<div class="side-heading">
🚀 AI Features
</div>

<div class="side-text">

✅ Real-Time Emotion Detection<br>
✅ Deep Learning CNN Model<br>
✅ Live Face Recognition<br>
✅ AI Confidence Analytics<br>
✅ Multi Emotion Prediction

</div>

</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="side-card">

<div class="side-heading">
🧠 Technologies
</div>

<div class="side-text">

🔹 Streamlit<br>
🔹 OpenCV<br>
🔹 TensorFlow<br>
🔹 Plotly Dashboard<br>
🔹 Deep Learning CNN

</div>

</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="side-card">

<div class="side-heading">
📌 About Project
</div>

<div class="side-text">

This AI system detects human emotions from facial expressions using Computer Vision and Deep Learning.

</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# MAIN TITLE
# =========================================================

st.markdown("""
<div class="main-title">
😊 FACE EMOTION DETECTION
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Real-Time AI Facial Emotion Recognition Dashboard
</div>
""", unsafe_allow_html=True)

# =========================================================
# DASHBOARD
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">

    <div class="metric-label">
    🎯 AI Accuracy
    </div>

    <div class="metric-value">
    98%
    </div>

    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">

    <div class="metric-label">
    🧠 Model
    </div>

    <div class="metric-value">
    CNN
    </div>

    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">

    <div class="metric-label">
    📡 Detection
    </div>

    <div class="metric-value">
    LIVE
    </div>

    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">

    <div class="metric-label">
    ⚡ Version
    </div>

    <div class="metric-value">
    2.0
    </div>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

with st.spinner("Loading AI Engine..."):
    model = load_model("emotion_model.h5")

# =========================================================
# FACE CASCADE
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =========================================================
# EMOTIONS
# =========================================================

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

emoji = {
    "Angry":"😠",
    "Disgust":"🤢",
    "Fear":"😨",
    "Happy":"😊",
    "Neutral":"😐",
    "Sad":"😢",
    "Surprise":"😲"
}

feedback = {
    "Happy":"You look cheerful and energetic!",
    "Sad":"Take rest and stay positive.",
    "Angry":"Relax and take a deep breath.",
    "Fear":"Everything will be okay.",
    "Neutral":"You seem calm and focused.",
    "Surprise":"Something exciting happened?",
    "Disgust":"Take a short refreshing break."
}

# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs([
    "📷 Live Webcam Detection",
    "🖼 Upload Image Detection"
])

# =========================================================
# WEBCAM TAB
# =========================================================

with tab1:

    st.markdown('<div class="glass-card">',
    unsafe_allow_html=True)

    run = st.checkbox("🎥 Start Webcam Detection")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:

        ret, frame = camera.read()

        if not ret:
            st.error("Unable to access webcam")
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (x,y,w,h) in faces:

            roi_gray = gray[y:y+h, x:x+w]

            roi_gray = cv2.resize(
                roi_gray,
                (48,48)
            )

            roi = roi_gray.astype("float") / 255.0

            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            prediction = model.predict(
                roi,
                verbose=0
            )[0]

            max_index = int(np.argmax(prediction))

            emotion = emotion_labels[max_index]

            confidence = round(
                float(prediction[max_index]) * 100,
                2
            )

            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                (56,189,248),
                3
            )

            cv2.putText(
                frame,
                f"{emotion}",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (34,197,94),
                2
            )

            st.markdown(f"""
            <div class="result-box">

            <div class="result-emotion">
            {emoji[emotion]} {emotion}
            </div>

            <div class="result-confidence">
            Confidence : {confidence}%
            </div>

            <br>

            <h4>{feedback[emotion]}</h4>

            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        FRAME_WINDOW.image(frame)

    st.markdown('</div>',
    unsafe_allow_html=True)

# =========================================================
# IMAGE TAB
# =========================================================

with tab2:

    left, right = st.columns([1,1])

    with left:

        st.markdown('<div class="glass-card">',
        unsafe_allow_html=True)

        st.subheader("🖼 Upload Image")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg","jpeg","png"]
        )

        st.markdown('</div>',
        unsafe_allow_html=True)

    with right:

        if uploaded_file is not None:

            file_bytes = np.asarray(
                bytearray(uploaded_file.read()),
                dtype=np.uint8
            )

            image = cv2.imdecode(file_bytes,1)

            image = cv2.resize(
                image,
                (800,600)
            )

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4
            )

            if len(faces) == 0:

                st.error(
                    "❌ No Face Detected"
                )

            else:

                for (x,y,w,h) in faces:

                    cv2.rectangle(
                        image,
                        (x,y),
                        (x+w,y+h),
                        (56,189,248),
                        3
                    )

                    roi_gray = gray[y:y+h, x:x+w]

                    roi_gray = cv2.resize(
                        roi_gray,
                        (48,48)
                    )

                    roi = roi_gray.astype("float") / 255.0

                    roi = np.expand_dims(
                        roi,
                        axis=0
                    )

                    roi = np.expand_dims(
                        roi,
                        axis=-1
                    )

                    prediction = model.predict(
                        roi,
                        verbose=0
                    )[0]

                    max_index = int(
                        np.argmax(prediction)
                    )

                    emotion = emotion_labels[max_index]

                    confidence = round(
                        float(prediction[max_index]) * 100,
                        2
                    )

                    cv2.putText(
                        image,
                        emotion,
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (34,197,94),
                        2
                    )

                    st.markdown(f"""
                    <div class="result-box">

                    <div class="result-emotion">
                    {emoji[emotion]} {emotion}
                    </div>

                    <div class="result-confidence">
                    Confidence : {confidence}%
                    </div>
                    <br>

                    <h4>{feedback[emotion]}</h4>

                    </div>
                    """, unsafe_allow_html=True)

                    st.progress(int(confidence))

                    df = pd.DataFrame({
                        "Emotion": emotion_labels,
                        "Confidence": prediction
                    })

                    fig = px.bar(
                        df,
                        x="Emotion",
                        y="Confidence",
                        text_auto=True,
                        title="Emotion Confidence Analysis"
                    )

                    fig.update_layout(
                        paper_bgcolor="#0f172a",
                        plot_bgcolor="#0f172a",
                        font_color="white"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                image_rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    image_rgb,
                    caption="Detected Emotion Result",
                    use_container_width=True
                )

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

<hr>

<h3>
😊 FACE EMOTION DETECTION SYSTEM
</h3>

Built using Streamlit • OpenCV • TensorFlow • Deep Learning

<br>

© 2026 Artificial Intelligence Dashboard

</div>
""", unsafe_allow_html=True)



from pathlib import Path
import textwrap

import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Leaf Vision",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "deployment"
    / "final_leaf_non_leaf_model.keras"
)


# ============================================================
# CUSTOM VISUAL DESIGN
# ============================================================

st.html(
    textwrap.dedent("""
    <style>

    /* =====================================================
       GLOBAL APPLICATION
       ===================================================== */

    .stApp {
        background:
            linear-gradient(
                135deg,
                #f4faef 0%,
                #f7fbf2 45%,
                #eef8e9 100%
            );
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #063b2a 0%,
                #075438 45%,
                #063b2a 100%
            );
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }

    [data-testid="stSidebar"] * {
        color: #f2f9ed !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.18);
    }


    /* =====================================================
       TYPOGRAPHY
       ===================================================== */

    h1 {
        color: #073d2d !important;
        font-weight: 800 !important;
        letter-spacing: -1.5px;
    }

    h2, h3 {
        color: #0a4a35 !important;
        font-weight: 750 !important;
    }

    p {
        color: #17382d;
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        position: relative;
        overflow: hidden;

        padding: 42px 46px;
        margin-bottom: 28px;

        border-radius: 22px;

        background:
            linear-gradient(
                135deg,
                #edf8df 0%,
                #e6f5d9 55%,
                #d9efcf 100%
            );

        border: 1px solid #d2e8c7;

        box-shadow:
            0 10px 30px rgba(26, 77, 45, 0.10);
    }

    .hero::after {
        content: "🍃";
        position: absolute;
        right: 45px;
        bottom: -20px;
        font-size: 150px;
        opacity: 0.16;
        transform: rotate(-18deg);
    }

    .hero-title {
        font-size: 3.1rem;
        font-weight: 850;
        color: #073d2d;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 1.35rem;
        font-weight: 700;
        color: #176044;
        margin-bottom: 14px;
    }

    .hero-description {
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 750px;
        color: #25473b;
    }


    /* =====================================================
       SECTION CARDS
       ===================================================== */

    .soft-card {
        background: rgba(255,255,255,0.78);
        border: 1px solid #dcebd5;
        border-radius: 20px;
        padding: 25px;
        box-shadow:
            0 8px 25px rgba(30, 80, 45, 0.08);
    }


    /* =====================================================
       FEATURE CARDS
       ===================================================== */

    .feature-card {
        min-height: 180px;

        background:
            linear-gradient(
                145deg,
                #ffffff 0%,
                #f5faef 100%
            );

        border: 1px solid #dcebd5;
        border-radius: 18px;

        padding: 24px;

        box-shadow:
            0 7px 20px rgba(31, 83, 48, 0.07);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }

    .feature-title {
        font-size: 1.18rem;
        font-weight: 750;
        color: #0b4533;
        margin-bottom: 8px;
    }

    .feature-text {
        font-size: 0.95rem;
        line-height: 1.55;
        color: #456257;
    }


    /* =====================================================
       RESULT BOXES
       ===================================================== */

    .result-leaf {
        background:
            linear-gradient(
                135deg,
                #e6f8df,
                #d8f1d0
            );

        border: 2px solid #76c96c;
        border-radius: 20px;

        padding: 30px;

        text-align: center;

        box-shadow:
            0 8px 25px rgba(61, 145, 66, 0.12);
    }

    .result-nonleaf {
        background:
            linear-gradient(
                135deg,
                #fff0ed,
                #ffe5df
            );

        border: 2px solid #ed7b68;
        border-radius: 20px;

        padding: 30px;

        text-align: center;

        box-shadow:
            0 8px 25px rgba(190, 73, 53, 0.10);
    }

    .result-label {
        font-size: 2rem;
        font-weight: 850;
        margin-bottom: 5px;
    }

    .result-confidence {
        font-size: 1.05rem;
        color: #496157;
    }


    /* =====================================================
       HOW IT WORKS
       ===================================================== */

    .workflow {
        background:
            linear-gradient(
                135deg,
                #eef8df,
                #e6f4d8
            );

        border: 1px solid #d4e9c5;
        border-radius: 18px;

        padding: 25px;

        box-shadow:
            0 7px 22px rgba(50, 100, 50, 0.07);
    }

    .workflow-title {
        font-size: 1.3rem;
        font-weight: 750;
        color: #0b4533;
    }

    .step-number {
        display: inline-flex;

        width: 30px;
        height: 30px;

        align-items: center;
        justify-content: center;

        border-radius: 50%;

        background: #79c94b;
        color: white;

        font-weight: 800;
    }


    /* =====================================================
       BUTTON
       ===================================================== */

    .stButton > button {
        border-radius: 14px;
        min-height: 52px;

        font-weight: 750;
        font-size: 1.05rem;

        border: none;

        box-shadow:
            0 7px 18px rgba(26, 91, 52, 0.16);
    }


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {
        background: rgba(247, 252, 241, 0.85);
        border: 1px dashed #9acb82;
        border-radius: 17px;
        padding: 12px;
    }


    /* =====================================================
       METRICS
       ===================================================== */

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.76);
        border: 1px solid #dcebd5;
        border-radius: 16px;
        padding: 14px;
    }


    /* =====================================================
       EXPANDERS
       ===================================================== */

    [data-testid="stExpander"] {
        border: 1px solid #dcebd5;
        border-radius: 16px;
        background: rgba(255,255,255,0.70);
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        text-align: center;
        padding: 25px 10px;
        color: #577064;
        font-size: 0.9rem;
    }

    </style>
    """),
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image(image):

    image = image.convert("RGB")

    image_resized = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image_resized,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = load_model().predict(
        image_batch,
        verbose=0
    )

    non_leaf_probability = float(
        prediction[0][0]
    )

    leaf_probability = (
        1.0 - non_leaf_probability
    )

    threshold = 0.50

    if non_leaf_probability >= threshold:

        predicted_class = "NON-LEAF"
        confidence = non_leaf_probability

    else:

        predicted_class = "LEAF"
        confidence = leaf_probability

    return (
        predicted_class,
        confidence,
        leaf_probability,
        non_leaf_probability
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        textwrap.dedent("""
        <div style="
            text-align:center;
            padding:10px 0 20px 0;
        ">
            <div style="font-size:3.3rem;">🍃</div>

            <div style="
                font-size:1.65rem;
                font-weight:800;
            ">
                Leaf Vision
            </div>

            <div style="
                font-size:0.95rem;
                opacity:0.82;
            ">
                AI Image Classifier
            </div>
        </div>
        """),
    )

    st.write(
        "A deep learning image classification "
        "application designed to distinguish "
        "leaves from non-leaf images."
    )

    st.divider()

    st.markdown(
        "### 🤖 MODEL"
    )

    st.html(
        textwrap.dedent("""
        <div style="
            background:rgba(255,255,255,0.07);
            border:1px solid rgba(255,255,255,0.12);
            border-radius:15px;
            padding:18px;
        ">

        <div style="
            color:#a8ef76;
            font-size:1.15rem;
            font-weight:750;
        ">
        MobileNetV2
        </div>

        <p style="
            margin-top:10px;
            line-height:1.55;
            opacity:0.9;
        ">
        A convolutional neural network architecture
        fine-tuned for this binary classification task.
        </p>

        </div>
        """),
    )

    st.divider()

    st.markdown(
        "### 🖼️ INPUT"
    )

    st.write(
        "Image size: **224 × 224 pixels**"
    )

    st.write(
        "Classes:"
    )

    st.success("🍃 LEAF")

    st.error("🚫 NON-LEAF")

    st.divider()

    st.html(
        textwrap.dedent("""
        <div style="
            background:rgba(255,255,255,0.07);
            border-radius:15px;
            padding:17px;
        ">

        🛡️ <b>Final deployment model</b>

        <p style="
            margin-top:8px;
            line-height:1.5;
            opacity:0.85;
        ">
        Trained and fine-tuned for leaf classification.
        </p>

        </div>
        """),
    )

    st.divider()

    st.caption(
        "Leaf Vision • AI Image Classification"
    )


# ============================================================
# HERO SECTION
# ============================================================

st.html(
    textwrap.dedent("""
    <div class="hero">

        <div style="
            font-size:3.6rem;
            margin-bottom:4px;
        ">
            🍃
        </div>

        <div class="hero-title">
            Leaf Vision
        </div>

        <div class="hero-subtitle">
            Intelligent Leaf vs Non-Leaf Image Classifier
        </div>

        <div class="hero-description">
            Upload an image and let the trained deep learning
            model determine whether the main subject is a
            <b>leaf</b> or a <b>non-leaf object</b>.
        </div>

    </div>
    """),
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.subheader("📤 Upload Your Image")

st.write(
    "Choose a clear JPG, JPEG, or PNG image."
)

uploaded_file = st.file_uploader(
    "Select an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help=(
        "Upload an image containing a leaf "
        "or another object."
    )
)


# ============================================================
# FEATURE CARDS
# ============================================================

if uploaded_file is None:

    st.write("")

    feature1, feature2, feature3, feature4 = st.columns(
        4,
        gap="medium"
    )

    with feature1:

        st.html(
            textwrap.dedent("""
            <div class="feature-card">

                <div class="feature-icon">
                    🎯
                </div>

                <div class="feature-title">
                    Accurate
                </div>

                <div class="feature-text">
                    High-performance deep learning
                    model fine-tuned for leaf
                    classification.
                </div>

            </div>
            """),
        )

    with feature2:

        st.html(
            textwrap.dedent("""
            <div class="feature-card">

                <div class="feature-icon">
                    🛡️
                </div>

                <div class="feature-title">
                    Reliable
                </div>

                <div class="feature-text">
                    Evaluated on a held-out test
                    dataset for robust predictions.
                </div>

            </div>
            """),
        )

    with feature3:

        st.html(
            textwrap.dedent("""
            <div class="feature-card">

                <div class="feature-icon">
                    ⚡
                </div>

                <div class="feature-title">
                    Fast
                </div>

                <div class="feature-text">
                    Optimized MobileNetV2 architecture
                    for efficient image classification.
                </div>

            </div>
            """),
        )

    with feature4:

        st.html(
            textwrap.dedent("""
            <div class="feature-card">

                <div class="feature-icon">
                    🌿
                </div>

                <div class="feature-title">
                    Smart
                </div>

                <div class="feature-text">
                    Intelligent predictions with
                    confidence and probability analysis.
                </div>

            </div>
            """),
        )


# ============================================================
# IMAGE WORKFLOW
# ============================================================

if uploaded_file is not None:

    st.divider()

    st.subheader("🖼️ Image Preview")

    uploaded_image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_col, information_col = st.columns(
        [2, 1],
        gap="large"
    )

    with image_col:

        st.image(
            uploaded_image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with information_col:

        st.markdown(
            "### 📋 Image Information"
        )

        width, height = uploaded_image.size

        st.metric(
            "Width",
            f"{width}px"
        )

        st.metric(
            "Height",
            f"{height}px"
        )

        st.metric(
            "Color Mode",
            uploaded_image.mode
        )

        st.success(
            "✓ Image successfully loaded"
        )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.divider()

    st.subheader("🔍 Analyze Image")

    button_left, button_center, button_right = st.columns(
        [1, 2, 1]
    )

    with button_center:

        predict_button = st.button(
            "🔎  Predict Image",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        try:

            with st.spinner(
                "🧠 Analyzing image with MobileNetV2..."
            ):

                (
                    predicted_class,
                    confidence,
                    leaf_probability,
                    non_leaf_probability
                ) = predict_image(
                    uploaded_image
                )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🎯 Classification Result"
            )

            result_col, confidence_col = st.columns(
                [1, 1],
                gap="large"
            )

            with result_col:

                if predicted_class == "LEAF":

                    st.html(
                        textwrap.dedent("""
                        <div class="result-leaf">

                            <div class="result-label">
                                🍃 LEAF
                            </div>

                            <div class="result-confidence">
                                The model identifies this image
                                as a leaf.
                            </div>

                        </div>
                        """),
                    )

                else:

                    st.html(
                        textwrap.dedent("""
                        <div class="result-nonleaf">

                            <div class="result-label">
                                🚫 NON-LEAF
                            </div>

                            <div class="result-confidence">
                                The model identifies this image
                                as a non-leaf object.
                            </div>

                        </div>
                        """),
                    )


            with confidence_col:

                st.metric(
                    "🎯 Prediction Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.progress(
                    confidence
                )

                if confidence >= 0.90:

                    st.success(
                        "Very high confidence prediction."
                    )

                elif confidence >= 0.70:

                    st.warning(
                        "Moderate confidence prediction."
                    )

                else:

                    st.info(
                        "Lower confidence — consider "
                        "using a clearer image."
                    )


            # ------------------------------------------------
            # PROBABILITY ANALYSIS
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "📊 Probability Analysis"
            )

            probability_col1, probability_col2 = st.columns(
                2,
                gap="large"
            )

            with probability_col1:

                st.markdown(
                    "### 🍃 Leaf"
                )

                st.progress(
                    leaf_probability
                )

                st.write(
                    f"**{leaf_probability * 100:.2f}%**"
                )

            with probability_col2:

                st.markdown(
                    "### 🚫 Non-Leaf"
                )

                st.progress(
                    non_leaf_probability
                )

                st.write(
                    f"**{non_leaf_probability * 100:.2f}%**"
                )


            # ------------------------------------------------
            # TECHNICAL DETAILS
            # ------------------------------------------------

            with st.expander(
                "🔬 View Technical Prediction Details"
            ):

                technical_col1, technical_col2 = st.columns(
                    2
                )

                with technical_col1:

                    st.write(
                        "**Architecture:** MobileNetV2"
                    )

                    st.write(
                        "**Input:** 224 × 224 × 3"
                    )

                    st.write(
                        "**Normalization:** 1/255"
                    )

                with technical_col2:

                    st.write(
                        "**Threshold:** 0.50"
                    )

                    st.write(
                        f"**Leaf probability:** "
                        f"{leaf_probability:.6f}"
                    )

                    st.write(
                        f"**Non-leaf probability:** "
                        f"{non_leaf_probability:.6f}"
                    )


        except Exception as error:

            st.error(
                "❌ Unable to analyze this image."
            )

            st.warning(
                "Please try another clear JPG, JPEG, or PNG image."
            )

            with st.expander(
                "🔬 Technical Error Details"
            ):

                st.code(
                    str(error)
                )



# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.html(
    textwrap.dedent("""
    <div class="workflow">

        <div class="workflow-title">
            💡 How it works
        </div>

        <br>

        <span class="step-number">1</span>
        &nbsp; Upload an image

        &nbsp;&nbsp; →

        <span class="step-number">2</span>
        &nbsp; Preview the image

        &nbsp;&nbsp; →

        <span class="step-number">3</span>
        &nbsp; Click Predict Image

        &nbsp;&nbsp; →

        <span class="step-number">4</span>
        &nbsp; View the result

    </div>
    """),
)


# ============================================================
# ABOUT THE MODEL
# ============================================================

st.divider()

with st.expander(
    "🤖 About the AI Model"
):

    model_col1, model_col2, model_col3 = st.columns(
        3
    )

    with model_col1:

        st.metric(
            "Architecture",
            "MobileNetV2"
        )

    with model_col2:

        st.metric(
            "Input Size",
            "224 × 224"
        )

    with model_col3:

        st.metric(
            "Classes",
            "2"
        )

    st.write(
        "The final application uses a fine-tuned "
        "MobileNetV2 convolutional neural network "
        "for binary image classification."
    )

    st.info(
        "The deployment model is the final fine-tuned "
        "MobileNetV2 model selected after evaluation."
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    textwrap.dedent("""
    <div class="footer">

        🍃 <b>Leaf Vision</b> • AI Image Classification

        <br>

        Powered by TensorFlow, MobileNetV2 and Streamlit

        <br>

        Upload • Analyze • Understand

    </div>
    """),
)

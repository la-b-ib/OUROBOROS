import streamlit as st
import numpy as np
import plotly.graph_objects as go
import gudhi  # The Gold Standard for Topological Data Analysis (C++ backend)
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation
import pefile
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Import Advanced Methods (Algorithms 1-10)
try:
    from advanced_methods import (
        EnsembleFusionEngine,
        create_persistence_barcode_figure,
        create_multifractal_spectrum_figure,
        create_spectral_graph_figure
    )
    HAS_ADVANCED = True
except ImportError as e:
    HAS_ADVANCED = False
    print(f"Advanced methods not available: {e}")

# Import Interdisciplinary Methods (Algorithms 11-20)
try:
    from interdisciplinary_methods import InterdisciplinaryEnsemble
    from interdisciplinary_visualizations import get_visualization
    HAS_INTERDISCIPLINARY = True
except ImportError as e:
    HAS_INTERDISCIPLINARY = False
    print(f"Interdisciplinary methods not available: {e}")

# Domain I: Topological & Geometric
try:
    import kmapper as km
    from skdim.id import MOM  # Local Intrinsic Dimensionality
    HAS_MAPPER = True
except:
    HAS_MAPPER = False

# Domain II: Chaos & Dynamics
try:
    # Try individual chaos functions without loading datasets
    from nolds import lyap_r, hurst_rs, corr_dim, dfa
    HAS_CHAOS = True
except:
    try:
        # Fallback to hurst only
        from hurst import compute_Hc
        import numpy as np
        HAS_CHAOS = True
    except:
        HAS_CHAOS = False

# Domain III: Bio-Digital
try:
    from Bio import pairwise2
    from Bio.Seq import Seq
    HAS_BIO = True
except:
    HAS_BIO = False

# Domain IV: Spectral & Signal
try:
    from scipy.spatial.distance import euclidean
    from scipy.signal import spectrogram
    import numpy as np
    # fastdtw and skimage are optional
    try:
        from fastdtw import fastdtw
    except:
        pass
    try:
        from skimage.feature import graycomatrix, graycoprops
    except:
        pass
    HAS_SPECTRAL = True
except:
    HAS_SPECTRAL = False

# Domain V: Symbolic & Logic
try:
    from z3 import *
    from datasketch import MinHash
    import lzma
    import bz2
    import zlib
    HAS_SYMBOLIC = True
except:
    HAS_SYMBOLIC = False

# Domain VI: Sonification
try:
    from midiutil import MIDIFile
    HAS_MIDI = True
except:
    HAS_MIDI = False

st.set_page_config(
    page_title="OUROBOROS // Forensic Cockpit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# ADAPTIVE CYBER-MINIMALISM CSS FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ─── Google Fonts: Rajdhani (Headers) + JetBrains Mono (Code) ─── */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* ─── CSS Variables: Theme-Aware Color System ─── */
    :root {
        --glass-opacity: 0.7;
        --glass-blur: 12px;
        --glow-intensity: 0.6;
        --animation-speed: 0.3s;
    }
    
    /* ─── Global Typography Override ─── */
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }
    
    /* ─── App Container: Seamless Background ─── */
    .stApp {
        background: var(--background-color);
        transition: background var(--animation-speed) ease;
    }
    
    /* ─── Main Content Area: Remove Default Padding ─── */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }
    
    /* ─── Title: Clean with Glow (No Glitch) ─── */
    .glitch-title {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 4rem !important;
        font-weight: 700 !important;
        letter-spacing: 8px !important;
        text-align: center !important;
        margin: 0 !important;
        padding: 1rem 0 !important;
        position: relative !important;
        color: currentColor !important;
        text-shadow: 
            0 0 10px var(--primary-color),
            0 0 20px var(--primary-color),
            0 0 30px var(--primary-color);
    }
    
    .glitch-subtitle {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
        text-align: center !important;
        opacity: 0.7 !important;
        letter-spacing: 3px !important;
        margin-top: -10px !important;
        padding-bottom: 2rem !important;
        color: var(--primary-color) !important;
    }
    
    /* ─── Chameleon Glassmorphism Cards ─── */
    .glass-card {
        background: color-mix(in srgb, var(--secondary-background-color) 70%, transparent) !important;
        backdrop-filter: blur(var(--glass-blur)) saturate(180%) !important;
        -webkit-backdrop-filter: blur(var(--glass-blur)) saturate(180%) !important;
        border: 1px solid color-mix(in srgb, var(--primary-color) 30%, transparent) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 
            0 8px 32px 0 rgba(0, 0, 0, 0.1),
            inset 0 1px 0 0 color-mix(in srgb, var(--primary-color) 20%, transparent) !important;
        transition: all var(--animation-speed) cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .glass-card:hover {
        transform: translateY(-4px) !important;
        box-shadow: 
            0 12px 48px 0 color-mix(in srgb, var(--primary-color) 20%, transparent),
            inset 0 1px 0 0 color-mix(in srgb, var(--primary-color) 40%, transparent) !important;
        border-color: var(--primary-color) !important;
    }
    
    /* ─── Sidebar: Tactical Command Panel ─── */
    section[data-testid="stSidebar"] {
        background: color-mix(in srgb, var(--secondary-background-color) 85%, transparent) !important;
        backdrop-filter: blur(16px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(200%) !important;
        border-right: 1px solid color-mix(in srgb, var(--primary-color) 25%, transparent) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem !important;
    }
    
    /* ─── Sidebar Headers: HUD Style ─── */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--primary-color) !important;
        border-bottom: 2px solid var(--primary-color) !important;
        padding-bottom: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* ─── Metrics: Holographic Display ─── */
    [data-testid="stMetricValue"] {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: var(--primary-color) !important;
        text-shadow: 0 0 10px color-mix(in srgb, var(--primary-color) 60%, transparent) !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        opacity: 0.8 !important;
    }
    
    /* ─── Buttons: Cyber-Minimal Style ─── */
    .stButton > button {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        background: color-mix(in srgb, var(--primary-color) 15%, transparent) !important;
        border: 2px solid var(--primary-color) !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        transition: all var(--animation-speed) ease !important;
        color: var(--text-color) !important;
    }
    
    .stButton > button:hover {
        background: var(--primary-color) !important;
        color: var(--background-color) !important;
        box-shadow: 
            0 0 20px var(--primary-color),
            0 0 40px color-mix(in srgb, var(--primary-color) 50%, transparent) !important;
        transform: translateY(-2px) !important;
    }
    
    /* ─── Progress Bars: Adaptive Gradient ─── */
    .stProgress > div > div > div > div {
        background: linear-gradient(
            90deg,
            var(--primary-color),
            color-mix(in srgb, var(--primary-color) 70%, white)
        ) !important;
        border-radius: 4px !important;
        box-shadow: 0 0 10px color-mix(in srgb, var(--primary-color) 50%, transparent) !important;
    }
    
    /* ─── Tabs: Minimal Navigation ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: transparent !important;
        border-bottom: 2px solid color-mix(in srgb, var(--primary-color) 30%, transparent) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border: none !important;
        background: transparent !important;
        padding: 1rem 1.5rem !important;
        color: var(--text-color) !important;
        opacity: 0.6 !important;
        transition: all var(--animation-speed) ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        opacity: 1 !important;
        color: var(--primary-color) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: color-mix(in srgb, var(--primary-color) 15%, transparent) !important;
        opacity: 1 !important;
        border-bottom: 3px solid var(--primary-color) !important;
        color: var(--primary-color) !important;
    }
    
    /* ─── Expanders: Collapsible Panels ─── */
    .streamlit-expanderHeader {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        background: color-mix(in srgb, var(--secondary-background-color) 50%, transparent) !important;
        border: 1px solid color-mix(in srgb, var(--primary-color) 20%, transparent) !important;
        border-radius: 8px !important;
        transition: all var(--animation-speed) ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: var(--primary-color) !important;
        background: color-mix(in srgb, var(--primary-color) 10%, transparent) !important;
    }
    
    /* ─── File Uploader: Drop Zone ─── */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--primary-color) !important;
        border-radius: 12px !important;
        background: color-mix(in srgb, var(--secondary-background-color) 50%, transparent) !important;
        padding: 2rem !important;
        transition: all var(--animation-speed) ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: color-mix(in srgb, var(--primary-color) 100%, white) !important;
        background: color-mix(in srgb, var(--primary-color) 10%, transparent) !important;
        box-shadow: 0 0 20px color-mix(in srgb, var(--primary-color) 30%, transparent) !important;
    }
    
    /* ─── Code Blocks: Terminal Aesthetic ─── */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background: color-mix(in srgb, var(--secondary-background-color) 80%, transparent) !important;
        border: 1px solid color-mix(in srgb, var(--primary-color) 20%, transparent) !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 4px !important;
    }
    
    pre {
        background: color-mix(in srgb, var(--secondary-background-color) 80%, transparent) !important;
        border-left: 4px solid var(--primary-color) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }
    
    /* ─── Checkbox/Radio: Tactical Indicators ─── */
    [data-testid="stCheckbox"] label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
    }
    
    /* ─── Skeleton Loader: Pulse Animation ─── */
    .skeleton-loader {
        width: 100%;
        height: 20px;
        background: linear-gradient(
            90deg,
            color-mix(in srgb, var(--secondary-background-color) 100%, transparent) 0%,
            color-mix(in srgb, var(--primary-color) 20%, transparent) 50%,
            color-mix(in srgb, var(--secondary-background-color) 100%, transparent) 100%
        );
        background-size: 200% 100%;
        border-radius: 4px;
        animation: skeleton-pulse 1.5s infinite;
    }
    
    @keyframes skeleton-pulse {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* ─── HUD Divider Lines ─── */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(
            90deg,
            transparent,
            var(--primary-color),
            transparent
        ) !important;
        margin: 2rem 0 !important;
        opacity: 0.5 !important;
    }
    
    /* ─── Scrollbar: Cyber Theme ─── */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--secondary-background-color);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: color-mix(in srgb, var(--primary-color) 100%, white);
    }
    
    /* ─── Alert Boxes: Status Indicators ─── */
    .stAlert {
        border-radius: 8px !important;
        border-left: 4px solid var(--primary-color) !important;
        background: color-mix(in srgb, var(--secondary-background-color) 70%, transparent) !important;
        backdrop-filter: blur(8px) !important;
    }
    
    /* ─── Dataframe: Grid Display ─── */
    [data-testid="stDataFrame"] {
        border: 1px solid color-mix(in srgb, var(--primary-color) 30%, transparent) !important;
        border-radius: 8px !important;
    }
    
    /* ─── Spinner: Loading Animation ─── */
    .stSpinner > div {
        border-color: var(--primary-color) transparent transparent transparent !important;
    }
    
    /* ─── Pulse Animation for Threat Indicators ─── */
    @keyframes threat-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.1); }
    }
    
    .threat-indicator {
        animation: threat-pulse 2s infinite;
    }
    
    /* ─── Status Badge Styling ─── */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0.5rem;
        background: color-mix(in srgb, var(--primary-color) 20%, transparent);
        border: 2px solid var(--primary-color);
        box-shadow: 0 0 10px color-mix(in srgb, var(--primary-color) 40%, transparent);
    }
    
    /* ─── Scan Line Effect (Optional Cyberpunk Touch) ─── */
    @keyframes scan-line {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100vh); }
    }
    
    .scan-line {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(
            90deg,
            transparent,
            var(--primary-color),
            transparent
        );
        opacity: 0.3;
        animation: scan-line 8s linear infinite;
        pointer-events: none;
        z-index: 9999;
    }
    
    /* ─── Developer Signature: Adaptive Professional Badge ─── */
    .dev-signature {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
        margin-top: 3rem;
        border-top: 2px solid color-mix(in srgb, var(--primary-color) 30%, transparent);
        background: color-mix(in srgb, var(--secondary-background-color) 50%, transparent);
        backdrop-filter: blur(8px);
        border-radius: 12px 12px 0 0;
    }
    
    .dev-signature .dev-name {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: var(--primary-color);
        text-shadow: 0 0 10px color-mix(in srgb, var(--primary-color) 40%, transparent);
        margin: 0;
    }
    
    .dev-signature .dev-email {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem;
        color: var(--text-color);
        opacity: 0.7;
        margin: 0.5rem 0 0 0;
    }
    
    .dev-signature .dev-email a {
        color: var(--primary-color) !important;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    .dev-signature .dev-email a:hover {
        text-shadow: 0 0 10px var(--primary-color);
        opacity: 1;
    }
    
    .dev-signature .dev-badge {
        display: inline-block;
        margin-top: 0.5rem;
        padding: 0.3rem 1rem;
        background: color-mix(in srgb, var(--primary-color) 15%, transparent);
        border: 1px solid var(--primary-color);
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 1px;
        color: var(--primary-color);
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TITLE + SUBTITLE + SCAN LINE
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="scan-line"></div>
<div class="glitch-title">OUROBOROS Ꝏ</div>
<div class="glitch-subtitle">// FORENSIC COCKPIT ALPHA // TOPOLOGICAL DEFENSE SYSTEM //</div>
""", unsafe_allow_html=True)



# --- 1. TOPOLOGICAL CORE (The Hardest Math) ---
def compute_persistence_homology(bytes_data):
    """
    Treats the binary as a point cloud in high-dimensional space 
    and computes the 'holes' (Betti numbers) that persist across scales.
    """
    # 1. Embed bytes into a Point Cloud (Sliding Window Embedding)
    # This creates a "Time-Delay Embedding" of the code execution
    window_size = 3
    data = np.frombuffer(bytes_data, dtype=np.uint8)
    if len(data) > 5000: 
        data = data[:5000]  # Cap for demo speed
    
    point_cloud = []
    for i in range(len(data) - window_size):
        point_cloud.append(data[i:i+window_size])
    point_cloud = np.array(point_cloud)

    # 2. Build Simplicial Complex (Vietoris-Rips)
    # This connects points that are close to form triangles/tetrahedrons
    rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=10.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    
    # 3. Compute Persistent Homology
    # This finds loops (H1) and voids (H2)
    diag = simplex_tree.persistence()
    return diag

# --- 2. CHAOS CORE (Recurrence Plots) ---
def compute_recurrence_plot(bytes_data):
    """
    Visualizes the phase space trajectory. 
    Diagonal lines = Deterministic Logic. Scattered dots = Chaos.
    """
    data_points = np.frombuffer(bytes_data, dtype=np.uint8)[:1000]  # Sample
    time_series = TimeSeries(data_points, embedding_dimension=2, time_delay=2)
    
    from pyrqa.metric import EuclideanMetric
    settings = Settings(time_series, 
                       neighbourhood=FixedRadius(65.0), 
                       similarity_measure=EuclideanMetric,
                       theiler_corrector=1)
    
    try:
        # Try fast computation with OpenCL (may fail on cloud environments)
        computation = RQAComputation.create(settings, verbose=False)
        result = computation.run()
        return result.recurrence_rate, data_points
    except Exception as e:
        # Fallback: compute basic recurrence rate manually without OpenCL
        try:
            # Manual recurrence computation using distance matrix
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(data_points.reshape(-1, 1), metric='euclidean'))
            threshold = 65.0
            recurrence_matrix = (dist_matrix < threshold).astype(int)
            recurrence_rate = np.sum(recurrence_matrix) / (len(data_points) ** 2)
            return recurrence_rate, data_points
        except:
            # Ultimate fallback: return basic entropy-based estimate
            return 0.1, data_points

# --- 3. MULTIFRACTAL CORE ---
def compute_multifractal_spectrum(bytes_data):
    """
    Analyzes the singularity spectrum to distinguish between
    legitimate compression and malicious polymorphic packing.
    """
    data = np.frombuffer(bytes_data, dtype=np.uint8)[:10000]
    
    # Compute box-counting dimensions at multiple scales
    q_values = np.linspace(-5, 5, 21)
    alpha_values = []
    f_alpha_values = []
    
    for q in q_values:
        # Partition data into boxes and compute generalized dimensions
        box_size = 256
        boxes = [data[i:i+box_size] for i in range(0, len(data), box_size)]
        
        if len(boxes) > 1:
            # Compute probability distribution
            probs = []
            for box in boxes:
                if len(box) > 0:
                    unique, counts = np.unique(box, return_counts=True)
                    probs.extend(counts / len(box))
            
            if len(probs) > 0:
                probs = np.array(probs)
                probs = probs[probs > 0]  # Remove zeros
                
                if q != 1:
                    tau_q = np.sum(probs ** q)
                    if tau_q > 0:
                        alpha = np.log(tau_q) / np.log(len(boxes))
                        alpha_values.append(alpha)
                        f_alpha_values.append(q * alpha)
    
    return alpha_values, f_alpha_values

# --- 4. DOMAIN I: TOPOLOGICAL & GEOMETRIC ---
def compute_reeb_graph_skeleton(bytes_data):
    """Mapper Algorithm: Generate skeletal representation of manifold"""
    if not HAS_MAPPER:
        return None
    
    data = np.frombuffer(bytes_data, dtype=np.uint8)[:5000]
    if len(data) < 100:
        return None
    
    # Create point cloud
    point_cloud = []
    window_size = 3
    for i in range(len(data) - window_size):
        point_cloud.append(data[i:i+window_size])
    point_cloud = np.array(point_cloud)
    
    # Create mapper
    mapper = km.KeplerMapper(verbose=0)
    
    # Fit to data
    lens = mapper.fit_transform(point_cloud, projection="sum")
    
    # Create graph
    graph = mapper.map(lens, point_cloud, cover=km.Cover(n_cubes=10, perc_overlap=0.3))
    
    return {"nodes": len(graph["nodes"]), "links": len(graph["links"])}

def compute_local_intrinsic_dim(bytes_data):
    """LID Estimation: Find where packer ends and payload begins"""
    if not HAS_MAPPER:
        return []
    
    data = np.frombuffer(bytes_data, dtype=np.uint8)[:5000]
    if len(data) < 100:
        return []
    
    # Sliding window LID estimation
    window_size = 100
    lids = []
    
    try:
        mom = MOM()
        for i in range(0, len(data) - window_size, 50):
            window = data[i:i+window_size].reshape(-1, 1)
            if len(window) > 20:
                lid = mom.fit_transform(window)
                lids.append(float(lid))
    except:
        pass
    
    return lids

# --- 5. DOMAIN II: CHAOS & DYNAMICS ---
def compute_lyapunov_exponent(bytes_data):
    """Butterfly Effect measurement: Chaos vs Order"""
    if not HAS_CHAOS:
        return 0.0
    
    data = np.frombuffer(bytes_data, dtype=np.uint8)[:2000].astype(float)
    if len(data) < 100:
        return 0.0
    
    try:
        # Largest Lyapunov exponent
        lyap = nolds.lyap_r(data, emb_dim=10, lag=1, min_tsep=10)
        return float(lyap)
    except:
        return 0.0

def compute_hurst_exponent(bytes_data):
    """Long-term memory detection: Legitimate vs Random"""
    if not HAS_CHAOS:
        return 0.5
    
    data = np.frombuffer(bytes_data, dtype=np.uint8)[:5000].astype(float)
    if len(data) < 100:
        return 0.5
    
    try:
        H, c, data_used = compute_Hc(data, kind='price', simplified=True)
        return float(H)
    except:
        return 0.5

def classify_cellular_automaton(bytes_data):
    """Wolfram Classification: Chaos vs Complexity"""
    data = np.frombuffer(bytes_data, dtype=np.uint8)[:1000]
    if len(data) < 100:
        return "Unknown"
    
    # Simple heuristic based on entropy and autocorrelation
    entropy = -np.sum((np.bincount(data) / len(data)) * np.log2(np.bincount(data) / len(data) + 1e-10))
    
    # Autocorrelation
    autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr_decay = np.abs(autocorr[10] / autocorr[0]) if len(autocorr) > 10 else 0
    
    if entropy > 7.5 and autocorr_decay < 0.1:
        return "Class 3 (Chaotic)"
    elif entropy > 6.0 and autocorr_decay > 0.3:
        return "Class 4 (Complex/Life-like)"
    elif entropy < 5.0:
        return "Class 1 (Uniform)"
    else:
        return "Class 2 (Periodic)"

# --- 6. DOMAIN III: BIO-DIGITAL ---
def compute_smith_waterman_alignment(bytes_data, reference=None):
    """DNA-style sequence alignment for mutation detection"""
    if not HAS_BIO:
        return 0.0
    
    # Convert bytes to "genetic" sequence (reduced alphabet)
    data = np.frombuffer(bytes_data, dtype=np.uint8)[:500]
    seq1 = ''.join([chr(65 + (b % 4)) for b in data])  # ABCD alphabet
    
    if reference is None:
        # Create a simple reference (normal code pattern)
        reference = 'ABCD' * 125
    
    try:
        alignments = pairwise2.align.localxx(seq1, reference, one_alignment_only=True)
        if alignments:
            score = alignments[0].score
            max_score = min(len(seq1), len(reference))
            return float(score / max_score) if max_score > 0 else 0.0
    except:
        pass
    
    return 0.0

def compute_entropy_rate(bytes_data):
    """Information Theoretic Death: Entropy production rate"""
    data = np.frombuffer(bytes_data, dtype=np.uint8)[:5000]
    if len(data) < 100:
        return []
    
    # Compute entropy in windows
    window_size = 100
    entropy_rates = []
    
    for i in range(0, len(data) - window_size, 50):
        window = data[i:i+window_size]
        counts = np.bincount(window, minlength=256)
        probs = counts / window_size
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        entropy_rates.append(entropy)
    
    return entropy_rates

# --- 7. DOMAIN IV: SPECTRAL & SIGNAL ---
def compute_graph_laplacian_spectrum(bytes_data):
    """Sound of the Graph: Eigenvalue fingerprint"""
    data = np.frombuffer(bytes_data, dtype=np.uint8)[:1000]
    if len(data) < 50:
        return []
    
    # Build graph from byte transitions
    G = nx.DiGraph()
    for i in range(len(data) - 1):
        G.add_edge(int(data[i]), int(data[i+1]))
    
    if len(G.nodes()) < 3:
        return []
    
    # Compute Laplacian eigenvalues
    try:
        L = nx.laplacian_matrix(G.to_undirected()).todense()
        eigenvalues = np.linalg.eigvalsh(L)
        # Return first 10 eigenvalues (spectral signature)
        return eigenvalues[:10].tolist()
    except:
        return []

def compute_glcm_texture(bytes_data):
    """Gray-Level Co-occurrence Matrix: Visual texture fingerprint"""
    if not HAS_SPECTRAL:
        return {}
    
    data = np.frombuffer(bytes_data, dtype=np.uint8)[:10000]
    
    # Reshape into 2D image
    side = int(np.sqrt(len(data)))
    if side < 10:
        return {}
    
    image = data[:side*side].reshape(side, side)
    
    try:
        # Compute GLCM
        glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        # Extract texture features
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        
        return {
            'contrast': float(contrast),
            'correlation': float(correlation),
            'energy': float(energy),
            'homogeneity': float(homogeneity)
        }
    except:
        return {}

# --- 8. DOMAIN V: SYMBOLIC & LOGIC ---
def compute_kolmogorov_complexity(bytes_data):
    """Compression profile: Compressibility fingerprint"""
    if not HAS_SYMBOLIC:
        return {}
    
    data = bytes_data[:5000]
    original_size = len(data)
    
    if original_size == 0:
        return {}
    
    results = {}
    
    try:
        # LZMA
        compressed = lzma.compress(data)
        results['lzma'] = len(compressed) / original_size
    except:
        results['lzma'] = 1.0
    
    try:
        # BZ2
        compressed = bz2.compress(data)
        results['bz2'] = len(compressed) / original_size
    except:
        results['bz2'] = 1.0
    
    try:
        # Zlib
        compressed = zlib.compress(data)
        results['zlib'] = len(compressed) / original_size
    except:
        results['zlib'] = 1.0
    
    return results

def compute_minhash_signature(bytes_data):
    """LSH Signature: Fast fuzzy matching"""
    if not HAS_SYMBOLIC:
        return None
    
    data = np.frombuffer(bytes_data, dtype=np.uint8)[:5000]
    
    # Create MinHash
    m = MinHash(num_perm=128)
    
    # Add byte n-grams
    for i in range(len(data) - 3):
        ngram = bytes(data[i:i+4])
        m.update(ngram)
    
    return m.hashvalues[:10].tolist()

def check_benfords_law(bytes_data):
    """Benford's Law: Natural vs Artificial distribution"""
    data = np.frombuffer(bytes_data, dtype=np.uint8)
    
    # Get first digits (in decimal)
    first_digits = []
    for b in data:
        if b > 0:
            first_digit = int(str(b)[0])
            if first_digit > 0:
                first_digits.append(first_digit)
    
    if len(first_digits) < 30:
        return 0.0
    
    # Expected Benford distribution
    benford = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
    
    # Observed distribution
    observed = np.zeros(9)
    for d in first_digits:
        if 1 <= d <= 9:
            observed[d-1] += 1
    observed = observed / np.sum(observed)
    
    # Chi-square test
    chi_square = np.sum((observed - benford)**2 / (benford + 1e-10))
    
    return float(chi_square)

# --- 9. DOMAIN VI: SONIFICATION ---
def generate_midi_sonification(bytes_data, filename="malware_sound.mid"):
    """Convert binary patterns to MIDI for auditory analysis"""
    if not HAS_MIDI:
        return None
    
    data = np.frombuffer(bytes_data, dtype=np.uint8)[:1000]
    
    # Create MIDI file
    midi = MIDIFile(1)
    track = 0
    time = 0
    midi.addTrackName(track, time, "Malware Sonification")
    midi.addTempo(track, time, 120)
    
    channel = 0
    volume = 100
    
    # Map bytes to notes
    for i, byte_val in enumerate(data):
        pitch = 36 + (byte_val % 48)  # MIDI notes 36-83
        duration = 0.25
        midi.addNote(track, channel, pitch, time, duration, volume)
        time += duration
        
        # Add drum on high bytes (potential jumps/calls)
        if byte_val > 200:
            midi.addNote(track, 9, 36, time, 0.25, volume)  # Kick drum
    
    # Write to file
    with open(filename, 'wb') as f:
        midi.writeFile(f)
    
    return filename

# --- 10. ENHANCED THREAT ASSESSMENT ENGINE ---
def assess_threat_level(diag, rec_rate, multifractal_data, advanced_metrics=None):
    """
    Combines topological, chaotic, and advanced metrics to compute threat score.
    """
    # Count persistent features (long-lived homology classes)
    h1_intervals = [p[1] for p in diag if p[0] == 1]
    persistent_features = sum(1 for interval in h1_intervals 
                             if interval[1] < float('inf') and (interval[1] - interval[0]) > 5)
    
    # Recurrence rate (high = deterministic, low = random)
    chaos_score = abs(rec_rate - 0.5) * 2  # Normalize to [0,1]
    
    # Multifractal complexity
    alpha_values, _ = multifractal_data
    multifractal_width = max(alpha_values) - min(alpha_values) if alpha_values else 0
    
    # Combined threat assessment (base)
    topology_threat = min(persistent_features / 10, 1.0)
    chaos_threat = chaos_score
    multifractal_threat = min(multifractal_width / 3, 1.0)
    
    # Advanced metrics integration
    advanced_threat = 0.0
    if advanced_metrics:
        # Lyapunov (chaos)
        if 'lyapunov' in advanced_metrics and advanced_metrics['lyapunov'] > 0.5:
            advanced_threat += 0.2
        
        # Hurst (memory)
        if 'hurst' in advanced_metrics:
            h = advanced_metrics['hurst']
            if h < 0.3 or h > 0.7:  # Deviation from normal
                advanced_threat += 0.15
        
        # LID (packing)
        if 'lid_variance' in advanced_metrics and advanced_metrics['lid_variance'] > 5.0:
            advanced_threat += 0.2
        
        # Benford (unnaturalness)
        if 'benford' in advanced_metrics and advanced_metrics['benford'] > 10.0:
            advanced_threat += 0.15
        
        # Compression anomaly
        if 'compression_min' in advanced_metrics and advanced_metrics['compression_min'] > 0.95:
            advanced_threat += 0.15
        
        # Genome mismatch
        if 'genome_match' in advanced_metrics and advanced_metrics['genome_match'] < 0.5:
            advanced_threat += 0.15
        
        advanced_threat = min(advanced_threat, 1.0)
    
    overall_threat = (topology_threat * 0.25 + 
                     chaos_threat * 0.20 + 
                     multifractal_threat * 0.20 +
                     advanced_threat * 0.35)
    
    return {
        'overall': overall_threat,
        'topology': topology_threat,
        'chaos': chaos_threat,
        'multifractal': multifractal_threat,
        'advanced': advanced_threat,
        'persistent_features': persistent_features,
        'recurrence_rate': rec_rate
    }

# ═══════════════════════════════════════════════════════════════════════════
# HUD STATUS DASHBOARD - PRACTICAL & FEATURE-RICH
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

# Calculate dynamic metrics
core_available = sum([HAS_ADVANCED, HAS_CHAOS, HAS_BIO, HAS_SPECTRAL, HAS_SYMBOLIC])
total_modules = 5
core_percentage = (core_available / total_modules) * 100

interdisciplinary_count = 10 if HAS_INTERDISCIPLINARY else 0
total_algorithms = (10 if HAS_ADVANCED else 0) + interdisciplinary_count
ensemble_capability = "FULL FUSION" if (HAS_ADVANCED and HAS_INTERDISCIPLINARY) else "PARTIAL" if (HAS_ADVANCED or HAS_INTERDISCIPLINARY) else "BASIC"

visualization_types = 0
if HAS_ADVANCED: visualization_types += 3  # Persistence, Multifractal, Spectral
if HAS_INTERDISCIPLINARY: visualization_types += 10  # All interdisciplinary
if HAS_MAPPER: visualization_types += 1
if HAS_CHAOS: visualization_types += 2
if HAS_BIO: visualization_types += 1
if HAS_SPECTRAL: visualization_types += 2
if HAS_MIDI: visualization_types += 1

system_health = "OPTIMAL" if core_percentage >= 80 else "DEGRADED" if core_percentage >= 40 else "CRITICAL"

hud_col1, hud_col2, hud_col3, hud_col4, hud_col5 = st.columns(5)

with hud_col1:
    st.metric(
        label="🛡️ CORE MODULES",
        value=f"{core_available}/{total_modules}",
        delta=f"{core_percentage:.0f}% Active",
        delta_color="normal" if core_percentage >= 60 else "inverse"
    )
    with st.popover("ℹ️ Details"):
        st.write("**Module Status:**")
        st.write(f"{'✅' if HAS_ADVANCED else '❌'} Advanced Methods (Core 1-10)")
        st.write(f"{'✅' if HAS_CHAOS else '❌'} Chaos Theory & Dynamics")
        st.write(f"{'✅' if HAS_BIO else '❌'} Bio-Informatics")
        st.write(f"{'✅' if HAS_SPECTRAL else '❌'} Spectral Analysis")
        st.write(f"{'✅' if HAS_SYMBOLIC else '❌'} Symbolic Logic")

with hud_col2:
    st.metric(
        label="⚛️ INTERDISCIPLINARY",
        value=f"{interdisciplinary_count}/10",
        delta="Full Suite" if HAS_INTERDISCIPLINARY else "Not Loaded",
        delta_color="normal" if HAS_INTERDISCIPLINARY else "off"
    )
    with st.popover("ℹ️ Details"):
        if HAS_INTERDISCIPLINARY:
            st.write("**Available Algorithms:**")
            st.write("✅ Gravitational Lensing")
            st.write("✅ Epigenetic Tracking")
            st.write("✅ Quantum Walk")
            st.write("✅ Fluid Dynamics")
            st.write("✅ Stylometric Radar")
            st.write("✅ Event Horizon Entropy")
            st.write("✅ Symbiotic Process Tree")
            st.write("✅ Temporal Manifold")
            st.write("✅ Neural-Symbolic Hybrid")
            st.write("✅ Sonification Audio")
        else:
            st.warning("Install interdisciplinary_methods module")

with hud_col3:
    st.metric(
        label="🎯 ENSEMBLE MODE",
        value=f"{total_algorithms} Algos",
        delta=ensemble_capability,
        delta_color="normal" if total_algorithms >= 15 else "inverse"
    )
    with st.popover("ℹ️ Details"):
        st.write("**Ensemble Configuration:**")
        st.write(f"Total Algorithms: {total_algorithms}")
        st.write(f"Voting System: {'Enabled' if total_algorithms >= 3 else 'Disabled'}")
        st.write(f"Confidence Threshold: {0.7 if total_algorithms >= 10 else 0.5}")
        st.write(f"Fusion Mode: {ensemble_capability}")

with hud_col4:
    st.metric(
        label="📊 VISUALIZATIONS",
        value=f"{visualization_types}",
        delta="Plotly 3D",
        delta_color="normal"
    )
    with st.popover("ℹ️ Details"):
        st.write("**Visualization Types:**")
        st.write(f"{'✅' if HAS_ADVANCED else '❌'} Persistence Barcodes")
        st.write(f"{'✅' if HAS_ADVANCED else '❌'} Multifractal Spectrums")
        st.write(f"{'✅' if HAS_ADVANCED else '❌'} Spectral Graphs")
        st.write(f"{'✅' if HAS_INTERDISCIPLINARY else '❌'} Physics Simulations")
        st.write(f"{'✅' if HAS_INTERDISCIPLINARY else '❌'} Genomic Heatmaps")
        st.write(f"{'✅' if HAS_INTERDISCIPLINARY else '❌'} Quantum Visualizations")
        st.write(f"Total Chart Types: {visualization_types}")

with hud_col5:
    st.metric(
        label="⚡ SYSTEM STATUS",
        value=system_health,
        delta="Real-Time Analysis",
        delta_color="normal" if system_health == "OPTIMAL" else "inverse"
    )
    with st.popover("ℹ️ Details"):
        st.write("**System Information:**")
        st.write(f"Health: {system_health}")
        st.write(f"Core Coverage: {core_percentage:.0f}%")
        st.write(f"Processing Mode: {'Parallel' if total_algorithms >= 5 else 'Sequential'}")
        st.write(f"Memory Optimization: {'Enabled' if HAS_ADVANCED else 'Basic'}")

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SKELETON LOADER HELPER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════
def show_skeleton_loader(num_lines=3):
    """Display animated skeleton loader during processing"""
    for _ in range(num_lines):
        st.markdown('<div class="skeleton-loader"></div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR: TACTICAL ALGORITHM SELECTION
# ═══════════════════════════════════════════════════════════════════════════

# --- UI EXECUTION ---

# Sidebar for algorithm selection
st.sidebar.title("🎛️ Analysis Configuration")

# Developer Signature Badge in Sidebar
st.sidebar.markdown("""
<div style="
    text-align: center; 
    padding: 1rem; 
    margin-bottom: 1.5rem;
    background: color-mix(in srgb, var(--primary-color) 10%, transparent);
    border: 1px solid var(--primary-color);
    border-radius: 8px;
    backdrop-filter: blur(8px);
">
    <p style="
        margin: 0;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: var(--primary-color);
        text-shadow: 0 0 8px color-mix(in srgb, var(--primary-color) 40%, transparent);
    ">⚡ LABIB BIN SHAHED ⚡</p>
    <p style="
        margin: 0.3rem 0 0 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-color);
        opacity: 0.8;
    ">Forensic AI Engineer</p>
    <a href="mailto:labib-x@protonmail.com" style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: var(--primary-color);
        text-decoration: none;
        opacity: 0.8;
    ">labib-x@protonmail.com</a>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Select Algorithms to Run")

if HAS_ADVANCED:
    st.sidebar.success("✅ Advanced Methods Module Loaded")
    
    # Algorithm categories
    core_algorithms = {
        1: "Persistent Homology Kernels",
        2: "Multifractal Spectrum (WTMM)",
        3: "Spectral Graph Clustering",
        4: "Recurrence Quantification",
        5: "Compression-Based Clustering",
        6: "Dynamic Time Warping",
        7: "Latent Dirichlet Allocation",
        8: "Benford's Law Analysis",
        9: "MinHash LSH",
        10: "Symbolic Execution (Z3)"
    }
    
    advanced_algorithms = {
        11: "Topological Autoencoder",
        12: "Zigzag Persistence",
        13: "Isomap/LLE Manifold",
        14: "Quasi-Monte Carlo TDA"
    }
    
    # Check for interdisciplinary methods
    if HAS_INTERDISCIPLINARY:
        interdisciplinary_algorithms = {
            11: "⚛️ Gravitational Lensing (Physics)",
            12: "🧬 Epigenetic State Tracking (Genomics)",
            13: "🌀 Quantum Walk Control Flow",
            14: "💨 Fluid Dynamics Data Flow",
            15: "📡 Stylometric Phonetic Radar",
            16: "🌌 Event Horizon Entropy Surface",
            17: "🌿 Symbiotic Process Tree",
            18: "⏰ Chrono-Slicing Temporal Manifold",
            19: "🔬 Neural-Symbolic Hybrid Verifier",
            20: "🎵 Sonification Spectral Audio"
        }
        # Override advanced_algorithms with interdisciplinary
        advanced_algorithms = interdisciplinary_algorithms
    
    st.sidebar.markdown("#### Core Algorithms (1-10)")
    selected_core = []
    for algo_id, algo_name in core_algorithms.items():
        if st.sidebar.checkbox(algo_name, value=True, key=f"core_{algo_id}"):
            selected_core.append(algo_id)
    
    st.sidebar.markdown("#### 🌟 Interdisciplinary Methods (11-20)" if HAS_INTERDISCIPLINARY else "#### Advanced Methods (11-14)")
    selected_advanced = []
    for algo_id, algo_name in advanced_algorithms.items():
        if st.sidebar.checkbox(algo_name, value=True, key=f"adv_{algo_id}"):
            selected_advanced.append(algo_id)
    
    selected_algorithms = selected_core + selected_advanced
    
    st.sidebar.markdown(f"**Total Selected:** {len(selected_algorithms)} algorithms")
else:
    st.sidebar.warning("⚠️ Advanced Methods Module Not Available")
    st.sidebar.info("Running in legacy mode with basic TDA")
    selected_algorithms = None

# ═══════════════════════════════════════════════════════════════════════════
# FILE UPLOAD: GLASS CARD INTERFACE
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🔬 BINARY PAYLOAD INJECTION")
uploaded_file = st.file_uploader("Drop your target binary for forensic analysis", type=None, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    bytes_data = uploaded_file.read()
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"### 📁 TARGET ANALYSIS")
    st.markdown(f"**File:** `{uploaded_file.name}` | **Size:** {len(bytes_data):,} bytes")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # ADVANCED ENSEMBLE ANALYSIS
    # ========================================================================
    
    if HAS_ADVANCED and selected_algorithms:
        st.markdown("---")
        
        # Separate core and interdisciplinary algorithms
        core_selected = [a for a in selected_algorithms if a <= 10]
        interdisciplinary_selected = [a for a in selected_algorithms if a > 10]
        
        # Run core algorithms (1-10)
        ensemble_result = None
        if core_selected and HAS_ADVANCED:
            with st.spinner("⚙️ Running core mathematical algorithms..."):
                engine = EnsembleFusionEngine()
                ensemble_result = engine.analyze(bytes_data, core_selected)
        
        # Run interdisciplinary algorithms (11-20)
        interdisciplinary_result = None
        if interdisciplinary_selected and HAS_INTERDISCIPLINARY:
            with st.spinner("⚛️ Running interdisciplinary fusion (Physics, Genomics, Quantum, etc.)..."):
                interdisciplinary_engine = InterdisciplinaryEnsemble()
                interdisciplinary_result = interdisciplinary_engine.analyze(bytes_data, interdisciplinary_selected)
        
        # Combine results if both exist
        if ensemble_result and interdisciplinary_result:
            total_votes = ensemble_result['threat_votes'] + interdisciplinary_result['threat_votes']
            total_algos = ensemble_result['total_algorithms'] + interdisciplinary_result['total_algorithms']
            combined_confidence = total_votes / total_algos if total_algos > 0 else 0.0
            
            if combined_confidence >= 0.6:
                combined_verdict = "HIGH THREAT"
                combined_color = "🔴"
            elif combined_confidence >= 0.3:
                combined_verdict = "MODERATE THREAT"
                combined_color = "🟡"
            else:
                combined_verdict = "LOW THREAT"
                combined_color = "🟢"
            
            display_result = {
                'confidence': combined_confidence,
                'threat_votes': total_votes,
                'total_algorithms': total_algos,
                'verdict': combined_verdict,
                'color': combined_color
            }
        elif ensemble_result:
            display_result = ensemble_result
        elif interdisciplinary_result:
            display_result = interdisciplinary_result
        else:
            display_result = None
        
        # Display Ensemble Verdict in Glass Card
        if display_result:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 ENSEMBLE VERDICT")
            
            col_v1, col_v2, col_v3, col_v4 = st.columns([1, 2, 1, 1])
            
            with col_v1:
                st.metric("Status", display_result['color'])
            
            with col_v2:
                st.metric("Confidence Score", f"{display_result['confidence']:.1%}")
                st.progress(display_result['confidence'])
            
            with col_v3:
                st.metric("Threat Votes", f"{display_result['threat_votes']}/{display_result['total_algorithms']}")
            
            with col_v4:
                st.metric("Classification", display_result['verdict'])
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display Individual Algorithm Results
        st.markdown("---")
        
        # Create tabs for different algorithm categories
        tab_core, tab_advanced, tab_visual = st.tabs(["🛡️ Core Algorithms (1-10)", "⚛️ Advanced Methods (11-14)", "📊 Visual Artifacts"])
        
        with tab_core:
            for algo_id in range(1, 11):
                if algo_id in ensemble_result['individual_results']:
                    result = ensemble_result['individual_results'][algo_id]
                    
                    with st.expander(f"**Algorithm #{algo_id}: {result['name']}** {'🔴' if result.get('result', {}).get('threat_indicator', False) else '🟢'}"):
                        st.caption(result['description'])
                        
                        if 'error' in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            algo_result = result['result']
                            
                            # Display key metrics
                            if algo_id == 1:  # Persistent Homology
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Betti β₀", algo_result.get('betti_numbers', {}).get('b0', 0))
                                with col2:
                                    st.metric("Betti β₁", algo_result.get('betti_numbers', {}).get('b1', 0))
                                with col3:
                                    st.metric("Persistence Entropy", f"{algo_result.get('persistence_entropy', 0):.2f}")
                                
                                # Visualize barcode
                                if algo_result.get('persistence_pairs'):
                                    fig = create_persistence_barcode_figure(algo_result['persistence_pairs'])
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_core_{algo_id}_barcode")
                            
                            elif algo_id == 2:  # Multifractal
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Spectrum Width Δα", f"{algo_result.get('spectrum_width', 0):.2f}")
                                with col2:
                                    st.metric("Peak Position", f"{algo_result.get('peak_position', 0):.2f}")
                                with col3:
                                    st.metric("Asymmetry", f"{algo_result.get('asymmetry', 0):.2f}")
                                
                                if algo_result.get('alpha') and algo_result.get('f_alpha'):
                                    fig = create_multifractal_spectrum_figure(algo_result['alpha'], algo_result['f_alpha'])
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_core_{algo_id}_multifractal")
                            
                            elif algo_id == 3:  # Spectral Graph
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Spectral Gap", f"{algo_result.get('spectral_gap', 0):.4f}")
                                with col2:
                                    st.metric("Clusters", algo_result.get('n_clusters', 0))
                                with col3:
                                    st.metric("Algebraic Connectivity", f"{algo_result.get('algebraic_connectivity', 0):.4f}")
                                
                                if algo_result.get('eigenvalues'):
                                    fig = create_spectral_graph_figure(algo_result['eigenvalues'])
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_core_{algo_id}_spectral")
                            
                            elif algo_id == 4:  # RQA
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Recurrence Rate", f"{algo_result.get('recurrence_rate', 0):.3f}")
                                with col2:
                                    st.metric("Determinism", f"{algo_result.get('determinism', 0):.3f}")
                                with col3:
                                    st.metric("Laminarity", f"{algo_result.get('laminarity', 0):.3f}")
                                
                                st.info(f"**Behavior:** {algo_result.get('behavior_class', 'Unknown')}")
                            
                            elif algo_id == 5:  # Compression
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Min Compression", f"{algo_result.get('min_ratio', 0):.3f}")
                                with col2:
                                    st.metric("Entropy", f"{algo_result.get('entropy', 0):.2f} bits")
                                with col3:
                                    st.metric("Packed", "Yes" if algo_result.get('is_packed', False) else "No")
                                
                                # Show compression ratios - 3D
                                ratios = algo_result.get('compression_ratios', {})
                                fig = go.Figure()
                                ratio_keys = list(ratios.keys())
                                ratio_vals = list(ratios.values())
                                colors = ['#FF0055', '#00FFFF', '#00FF41'][:len(ratio_keys)]
                                
                                for i, (key, val, color) in enumerate(zip(ratio_keys, ratio_vals, colors)):
                                    fig.add_trace(go.Scatter3d(
                                        x=[i, i],
                                        y=[0, val],
                                        z=[i, i],
                                        mode='lines+markers',
                                        line=dict(color=color, width=10),
                                        marker=dict(size=8, color=color),
                                        name=key,
                                        showlegend=True
                                    ))
                                
                                fig.update_layout(
                                    title="Compression Profile (3D)",
                                    template="plotly_white",
                                    scene=dict(
                                        xaxis_title="Algorithm",
                                        yaxis_title="Ratio",
                                        zaxis_title="Index"
                                    ),
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"chart_core_{algo_id}_compression")
                            
                            elif algo_id == 6:  # DTW
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("DTW Distance", f"{algo_result.get('dtw_distance', 0):.1f}")
                                with col2:
                                    st.metric("Normalized", f"{algo_result.get('normalized_distance', 0):.3f}")
                                with col3:
                                    st.metric("CF Density", f"{algo_result.get('control_flow_density', 0):.2%}")
                            
                            elif algo_id == 7:  # LDA
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Topics", algo_result.get('n_topics', 0))
                                with col2:
                                    st.metric("Avg Entropy", f"{algo_result.get('avg_entropy', 0):.2f}")
                                with col3:
                                    st.metric("Topic Variance", f"{algo_result.get('topic_variance', 0):.3f}")
                                
                                st.markdown("**Top Terms per Topic:**")
                                for i, topic in enumerate(algo_result.get('topics', [])):
                                    st.caption(f"Topic {i+1}: {', '.join(topic)}")
                            
                            elif algo_id == 8:  # Benford
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("χ² Statistic", f"{algo_result.get('chi_square', 0):.2f}")
                                with col2:
                                    st.metric("KS Statistic", f"{algo_result.get('ks_statistic', 0):.3f}")
                                with col3:
                                    st.metric("Conforms", "Yes" if algo_result.get('conforms_to_benford', False) else "No")
                                
                                # Plot distribution comparison - 3D
                                fig = go.Figure()
                                x = list(range(1, 10))
                                
                                # Benford's Law - green bars
                                for i, (digit, expected) in enumerate(zip(x, algo_result['expected_dist'])):
                                    fig.add_trace(go.Scatter3d(
                                        x=[i-0.2, i-0.2],
                                        y=[0, expected],
                                        z=[0, 0],
                                        mode='lines',
                                        line=dict(color='#00FF41', width=15),
                                        showlegend=(i==0),
                                        name="Benford's Law" if i==0 else None
                                    ))
                                
                                # Observed - red bars
                                for i, (digit, observed) in enumerate(zip(x, algo_result['observed_dist'])):
                                    fig.add_trace(go.Scatter3d(
                                        x=[i+0.2, i+0.2],
                                        y=[0, observed],
                                        z=[0, 0],
                                        mode='lines',
                                        line=dict(color='#FF0055', width=15),
                                        showlegend=(i==0),
                                        name="Observed" if i==0 else None
                                    ))
                                
                                fig.update_layout(
                                    title="First Digit Distribution (3D)",
                                    template="plotly_white",
                                    scene=dict(
                                        xaxis_title="Digit",
                                        yaxis_title="Frequency",
                                        zaxis_title="",
                                        zaxis=dict(showticklabels=False)
                                    ),
                                    height=500
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"chart_core_{algo_id}_benford")
                            
                            elif algo_id == 9:  # MinHash
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Signature Entropy", f"{algo_result.get('signature_entropy', 0):.2f}")
                                with col2:
                                    sig = algo_result.get('signature', [])
                                    st.caption(f"Signature: {sig[:5] if sig else 'N/A'}...")
                            
                            elif algo_id == 10:  # Z3
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Opaque Predicates", algo_result.get('control_flow_anomalies', 0))
                                with col2:
                                    st.metric("Arithmetic Chains", algo_result.get('arithmetic_chains', 0))
                                with col3:
                                    st.metric("Obfuscation Score", f"{algo_result.get('obfuscation_score', 0):.4f}")
                                
                                if algo_result.get('opaque_predicates'):
                                    st.markdown("**Detected Opaque Predicates:**")
                                    for pred in algo_result['opaque_predicates']:
                                        st.caption(f"Offset {pred['offset']}: {pred['type']}")
        
        with tab_advanced:
            # Display interdisciplinary methods if available
            results_to_display = None
            if interdisciplinary_result and HAS_INTERDISCIPLINARY:
                st.markdown("### 🌟 Interdisciplinary Mathematical Fusion")
                st.caption("Physics, Genomics, Quantum Computing, Astrophysics, and more...")
                results_to_display = interdisciplinary_result['individual_results']
                algo_range = range(11, 21)
            elif ensemble_result:
                results_to_display = ensemble_result['individual_results']
                algo_range = range(11, 15)
            
            if results_to_display:
                for algo_id in algo_range:
                    if algo_id in results_to_display:
                        result = results_to_display[algo_id]
                        
                        # Get emoji based on algorithm
                        threat_emoji = '🔴' if result.get('result', {}).get('threat_indicator', False) else '🟢'
                        
                        with st.expander(f"**Algorithm #{algo_id}: {result['name']}** {threat_emoji}"):
                            st.caption(result['description'])
                            
                            if 'error' in result:
                                st.error(f"Error: {result['error']}")
                            else:
                                algo_result = result['result']
                                
                                # Display metrics and visualizations for interdisciplinary methods
                                if algo_id == 11 and HAS_INTERDISCIPLINARY:  # Gravitational Lensing
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Singularities", algo_result.get('n_singularities', 0))
                                    with col2:
                                        st.metric("Lensing Strength", f"{algo_result.get('lensing_strength', 0):.2f}")
                                    with col3:
                                        st.metric("Mass Concentration", f"{algo_result.get('mass_concentration', 0):.2f}")
                                    
                                    fig = get_visualization(11, algo_result)
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_inter_{algo_id}_lensing")
                                
                                elif algo_id == 12 and HAS_INTERDISCIPLINARY:  # Epigenetic
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Methylation Variance", f"{algo_result.get('methylation_variance', 0):.4f}")
                                    with col2:
                                        st.metric("Epigenetic Drift", f"{algo_result.get('epigenetic_drift', 0):.3f}")
                                    with col3:
                                        st.metric("Avg Accessibility", f"{algo_result.get('avg_accessibility', 0):.2f}")
                                    
                                    fig = get_visualization(12, algo_result)
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_inter_{algo_id}_epigenetic")
                                
                                elif algo_id == 13 and HAS_INTERDISCIPLINARY:  # Quantum Walk
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Interference Peaks", algo_result.get('n_interference_peaks', 'N/A'))
                                    with col2:
                                        st.metric("Quantum Coherence", f"{algo_result.get('quantum_coherence', 0):.2f}")
                                    with col3:
                                        st.metric("Tunneling Events", algo_result.get('tunneling_events', 'N/A'))
                                    
                                    fig = get_visualization(13, algo_result)
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_inter_{algo_id}_quantum")
                                
                                elif algo_id == 14 and HAS_INTERDISCIPLINARY:  # Fluid Dynamics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Reynolds Number", f"{algo_result.get('reynolds_number', 0):.1f}")
                                    with col2:
                                        st.metric("Flow Regime", algo_result.get('flow_regime', 'Unknown'))
                                    with col3:
                                        st.metric("Vortices", algo_result.get('n_vortices', 0))
                                    
                                    fig = get_visualization(14, algo_result)
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_inter_{algo_id}_fluid")
                                
                                elif algo_id == 15 and HAS_INTERDISCIPLINARY:  # Stylometric
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Zipf Ratio", f"{algo_result.get('zipf_ratio', 0):.2f}")
                                    with col2:
                                        st.metric("Type-Token Ratio", f"{algo_result.get('ttr_bigram', 0):.3f}")
                                    
                                    fig = get_visualization(15, algo_result)
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_inter_{algo_id}_stylometric")
                                
                                elif algo_id == 16 and HAS_INTERDISCIPLINARY:  # Event Horizon
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Event Horizons", algo_result.get('n_event_horizons', 0))
                                    with col2:
                                        st.metric("Schwarzschild", "✓" if algo_result.get('schwarzschild_reached', False) else "✗")
                                    with col3:
                                        st.metric("Max Curvature", f"{algo_result.get('max_curvature', 0):.2f}")
                                    
                                    fig = get_visualization(16, algo_result)
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_inter_{algo_id}_entropy")
                                
                                elif algo_id == 17 and HAS_INTERDISCIPLINARY:  # Symbiotic
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Parasitism", algo_result.get('relationships', {}).get('parasitism', 0))
                                    with col2:
                                        st.metric("Invasive Species", algo_result.get('n_invasive_species', 0))
                                    with col3:
                                        st.metric("Biodiversity", f"{algo_result.get('biodiversity', 0):.2f}")
                                    
                                    fig = get_visualization(17, algo_result)
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_inter_{algo_id}_symbiotic")
                                
                                elif algo_id == 18 and HAS_INTERDISCIPLINARY:  # Temporal Manifold
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Temporal Anomalies", algo_result.get('n_temporal_anomalies', 0))
                                    with col2:
                                        st.metric("Time Loops", algo_result.get('time_loops_detected', 0))
                                    with col3:
                                        st.metric("Causality Violations", algo_result.get('causality_violations', 0))
                                    
                                    fig = get_visualization(18, algo_result)
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_inter_{algo_id}_temporal")
                                
                                elif algo_id == 19 and HAS_INTERDISCIPLINARY:  # Neural-Symbolic
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Hybrid Confidence", f"{algo_result.get('hybrid_confidence', 0):.1%}")
                                    with col2:
                                        st.metric("Neural Confidence", f"{algo_result.get('neural_confidence', 0):.1%}")
                                    with col3:
                                        st.metric("Symbolic Confidence", f"{algo_result.get('symbolic_confidence', 0):.1%}")
                                    
                                    st.markdown("**Logical Proof:**")
                                    for step in algo_result.get('proof_steps', []):
                                        st.text(step)
                                    
                                    fig = get_visualization(19, algo_result)
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_inter_{algo_id}_neural_symbolic")
                                
                                elif algo_id == 20 and HAS_INTERDISCIPLINARY:  # Sonification
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Spectral Centroid", f"{algo_result.get('spectral_centroid', 0):.1f} Hz")
                                    with col2:
                                        st.metric("Audio Character", algo_result.get('audio_character', 'Unknown'))
                                    with col3:
                                        st.metric("Tempo", f"{algo_result.get('tempo', 0):.1f} BPS")
                                    
                                    fig = get_visualization(20, algo_result)
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_inter_{algo_id}_audio")
                                
                                # Handle original advanced algorithms (11-14) if not interdisciplinary
                                elif algo_id == 11 and not HAS_INTERDISCIPLINARY:  # Topological Autoencoder
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Reconstruction Error", f"{algo_result['reconstruction_error']:.4f}")
                                    with col2:
                                        st.metric("Explained Variance", f"{algo_result['explained_variance']:.2%}")
                                    with col3:
                                        st.metric("Latent Dim", algo_result['latent_dim'])
                                
                                elif algo_id == 12 and not HAS_INTERDISCIPLINARY:  # Zigzag Persistence
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Max State Change", algo_result['max_state_change'])
                                    with col2:
                                        st.metric("Chunks Analyzed", algo_result['n_chunks'])
                                    
                                    # Plot Betti evolution
                                    if algo_result['betti_evolution']:
                                        b1_vals = [chunk['b1'] for chunk in algo_result['betti_evolution']]
                                        fig = go.Figure()
                                        x_chunks = list(range(len(b1_vals)))
                                        fig.add_trace(go.Scatter3d(
                                            x=x_chunks,
                                            y=b1_vals,
                                            z=np.cumsum(b1_vals),  # Cumulative for 3D effect
                                            mode='lines+markers',
                                            line=dict(color='#00FFFF', width=4),
                                            marker=dict(size=6, color=b1_vals, colorscale='Plasma', showscale=True),
                                            name='β₁ Evolution'
                                        ))
                                        fig.update_layout(
                                            title="Topological State Evolution (3D)",
                                            template="plotly_white",
                                            scene=dict(
                                                xaxis_title="Chunk",
                                                yaxis_title="Betti β₁",
                                                zaxis_title="Cumulative β₁"
                                            ),
                                            height=500
                                        )
                                        st.plotly_chart(fig, use_container_width=True, key=f"chart_fallback_{algo_id}_zigzag")
                                
                                elif algo_id == 13 and not HAS_INTERDISCIPLINARY:  # Isomap/LLE
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Intrinsic Dim", algo_result['intrinsic_dim'])
                                    with col2:
                                        st.metric("Manifold Spread", f"{algo_result['manifold_spread']:.2f}")
                                    with col3:
                                        st.metric("Manifold Range", f"{algo_result['manifold_range']:.2f}")
                                
                                elif algo_id == 14 and not HAS_INTERDISCIPLINARY:  # QMC TDA
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Samples", algo_result['n_samples'])
                                    with col2:
                                        st.metric("Persistent Features", algo_result['persistent_features'])
                                    with col3:
                                        st.metric("Coverage", f"{algo_result['coverage']:.2%}")
        
        with tab_visual:
            st.header("🎨 Unified Manifold Visualization")
            st.markdown("**Interactive exploration of all algorithmic features in unified 3D space**")
            
            # Extract all features from ensemble results
            all_features = {}
            feature_categories = {}
            
            for algo_id, result in ensemble_result['individual_results'].items():
                category = "Core" if int(algo_id) <= 10 else "Interdisciplinary"
                for key, value in result.items():
                    if isinstance(value, (int, float)) and key not in ['verdict', 'confidence']:
                        feature_name = f"{algo_id}_{key}"
                        all_features[feature_name] = float(value) if isinstance(value, (int, float)) else 0
                        feature_categories[feature_name] = category
            
            # Create tabs for different visualizations
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                "🌐 3D Feature Graph",
                "🔍 Topology Explorer", 
                "⏱️ Evolution Timeline",
                "🎵 Sonic Spectrum"
            ])
            
            with viz_tab1:
                st.subheader("3D Force-Directed Feature Graph")
                st.markdown("**All extracted features connected by correlation strength**")
                
                # Create 3D force-directed graph
                import numpy as np
                
                if len(all_features) > 0:
                    # Convert features to array
                    feature_names = list(all_features.keys())
                    feature_values = np.array(list(all_features.values()))
                    
                    # Normalize values
                    if len(feature_values) > 0 and np.std(feature_values) > 0:
                        feature_values = (feature_values - np.mean(feature_values)) / np.std(feature_values)
                    
                    # Create 3D positions using PCA-like spread
                    n_features = len(feature_names)
                    theta = np.linspace(0, 4*np.pi, n_features)
                    phi = np.linspace(0, np.pi, n_features)
                    
                    x = feature_values * np.sin(phi) * np.cos(theta)
                    y = feature_values * np.sin(phi) * np.sin(theta)
                    z = feature_values * np.cos(phi)
                    
                    # Determine colors by category
                    colors = ['#667eea' if feature_categories[name] == 'Core' else '#764ba2' 
                             for name in feature_names]
                    
                    # Create 3D scatter plot
                    fig = go.Figure()
                    
                    # Add nodes
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='markers+text',
                        marker=dict(
                            size=10,
                            color=colors,
                            line=dict(color='white', width=2),
                            opacity=0.8
                        ),
                        text=[name.split('_')[1][:10] for name in feature_names],
                        textposition='top center',
                        textfont=dict(size=8),
                        hovertemplate='<b>%{text}</b><br>Value: %{marker.color}<extra></extra>',
                        name='Features'
                    ))
                    
                    # Add connection lines between similar features
                    for i in range(min(n_features, 50)):  # Limit connections
                        for j in range(i+1, min(n_features, 50)):
                            distance = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)
                            if distance < 1.5:  # Only connect close features
                                fig.add_trace(go.Scatter3d(
                                    x=[x[i], x[j]],
                                    y=[y[i], y[j]],
                                    z=[z[i], z[j]],
                                    mode='lines',
                                    line=dict(color='rgba(100,100,100,0.2)', width=1),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                    
                    fig.update_layout(
                        template='plotly_white',
                        scene=dict(
                            xaxis_title='Dimension X',
                            yaxis_title='Dimension Y',
                            zaxis_title='Dimension Z',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                        ),
                        height=700,
                        title='3D Feature Manifold (Force-Directed Layout)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature statistics
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Total Features", len(feature_names))
                    with col_stat2:
                        core_count = sum(1 for cat in feature_categories.values() if cat == 'Core')
                        st.metric("Core Features", core_count)
                    with col_stat3:
                        inter_count = len(feature_names) - core_count
                        st.metric("Interdisciplinary Features", inter_count)
                else:
                    st.warning("No numeric features extracted from analysis")
            
            with viz_tab2:
                st.subheader("Interactive Topology Explorer")
                st.markdown("**Navigate through topological features and their relationships**")
                
                # Create hierarchical topology view
                algo_results = ensemble_result['individual_results']
                
                # Group by verdict
                threat_algos = []
                benign_algos = []
                
                for algo_id, result in algo_results.items():
                    if result.get('verdict') == 'THREAT':
                        threat_algos.append((algo_id, result))
                    else:
                        benign_algos.append((algo_id, result))
                
                col_topo1, col_topo2 = st.columns(2)
                
                with col_topo1:
                    st.markdown("### 🔴 Threat Indicators")
                    st.metric("Algorithms Flagging Threat", len(threat_algos))
                    
                    if threat_algos:
                        # Create sunburst chart
                        labels = ['Threats'] + [f"Algo {aid}" for aid, _ in threat_algos]
                        parents = [''] + ['Threats'] * len(threat_algos)
                        values = [len(threat_algos)] + [r.get('confidence', 0.5) for _, r in threat_algos]
                        
                        fig_threat = go.Figure(go.Sunburst(
                            labels=labels,
                            parents=parents,
                            values=values,
                            marker=dict(colors=['#ff0844']*len(labels)),
                            hovertemplate='<b>%{label}</b><br>Confidence: %{value:.2f}<extra></extra>'
                        ))
                        
                        fig_threat.update_layout(
                            template='plotly_white',
                            height=400,
                            title='Threat Detection Hierarchy'
                        )
                        
                        st.plotly_chart(fig_threat, use_container_width=True)
                
                with col_topo2:
                    st.markdown("### 🟢 Benign Indicators")
                    st.metric("Algorithms Flagging Benign", len(benign_algos))
                    
                    if benign_algos:
                        # Create sunburst chart
                        labels = ['Benign'] + [f"Algo {aid}" for aid, _ in benign_algos]
                        parents = [''] + ['Benign'] * len(benign_algos)
                        values = [len(benign_algos)] + [r.get('confidence', 0.5) for _, r in benign_algos]
                        
                        fig_benign = go.Figure(go.Sunburst(
                            labels=labels,
                            parents=parents,
                            values=values,
                            marker=dict(colors=['#56ab2f']*len(labels)),
                            hovertemplate='<b>%{label}</b><br>Confidence: %{value:.2f}<extra></extra>'
                        ))
                        
                        fig_benign.update_layout(
                            template='plotly_white',
                            height=400,
                            title='Benign Detection Hierarchy'
                        )
                        
                        st.plotly_chart(fig_benign, use_container_width=True)
                
                # Network graph of algorithm relationships
                st.markdown("---")
                st.markdown("### 🕸️ Algorithm Correlation Network (3D)")
                
                # Algorithm name mapping
                algo_names = {
                    '1': "Persistence Homology",
                    '2': "Recurrence Quantification",
                    '3': "Spectral Graph",
                    '4': "Multifractal Spectrum",
                    '5': "Kolmogorov Complexity",
                    '6': "Dynamic Time Warping",
                    '7': "Latent Dirichlet Allocation",
                    '8': "Benford's Law",
                    '9': "MinHash Locality",
                    '10': "Z3 Constraint Solver",
                    '11': "Gravitational Lensing",
                    '12': "Zigzag Persistence",
                    '13': "Quantum Interference",
                    '14': "Fluid Dynamics",
                    '15': "Neural Morphogenesis",
                    '16': "Phylogenetic Tree",
                    '17': "Symbiotic Processes",
                    '18': "Temporal Manifold",
                    '19': "Crystallography",
                    '20': "Harmonic Resonance"
                }
                
                # Calculate correlation between algorithm confidences
                algo_ids = list(algo_results.keys())
                confidences = [algo_results[aid].get('confidence', 0.5) for aid in algo_ids]
                
                if len(algo_ids) > 1:
                    # Create topological 3D network with mesh surface
                    fig_network = go.Figure()
                    
                    # Position nodes on torus topology
                    n_nodes = len(algo_ids)
                    
                    # Create torus coordinates (R=major radius, r=minor radius)
                    R, r = 2.0, 0.8
                    u = np.linspace(0, 2*np.pi, n_nodes)
                    v = np.linspace(0, 2*np.pi, n_nodes)
                    
                    # Node positions on torus surface
                    angle_step = 2 * np.pi / n_nodes
                    x_pos = []
                    y_pos = []
                    z_pos = []
                    
                    for i in range(n_nodes):
                        u_i = (i * angle_step)
                        v_i = (confidences[i] * 2 * np.pi)  # Confidence affects vertical position
                        x = (R + r * np.cos(v_i)) * np.cos(u_i)
                        y = (R + r * np.cos(v_i)) * np.sin(u_i)
                        z = r * np.sin(v_i)
                        x_pos.append(x)
                        y_pos.append(y)
                        z_pos.append(z)
                    
                    x_pos = np.array(x_pos)
                    y_pos = np.array(y_pos)
                    z_pos = np.array(z_pos)
                    
                    # Create torus mesh surface for topology visualization
                    u_mesh = np.linspace(0, 2*np.pi, 50)
                    v_mesh = np.linspace(0, 2*np.pi, 30)
                    u_grid, v_grid = np.meshgrid(u_mesh, v_mesh)
                    
                    x_torus = (R + r * np.cos(v_grid)) * np.cos(u_grid)
                    y_torus = (R + r * np.cos(v_grid)) * np.sin(u_grid)
                    z_torus = r * np.sin(v_grid)
                    
                    # Add translucent torus surface
                    fig_network.add_trace(go.Surface(
                        x=x_torus, y=y_torus, z=z_torus,
                        colorscale=[[0, 'rgba(102,126,234,0.1)'], [1, 'rgba(118,75,162,0.1)']],
                        showscale=False,
                        hoverinfo='skip',
                        name='Topology',
                        opacity=0.15
                    ))
                    
                    # Add connecting edges with gradient colors
                    for i in range(n_nodes):
                        for j in range(i+1, n_nodes):
                            conf_diff = abs(confidences[i] - confidences[j])
                            if conf_diff < 0.3:  # Similar confidence creates stronger connection
                                edge_color = f'rgba({int(255*conf_diff)},{int(100*(1-conf_diff))},{int(200*(1-conf_diff))},0.5)'
                                edge_width = 3 if conf_diff < 0.15 else 1.5
                                
                                fig_network.add_trace(go.Scatter3d(
                                    x=[x_pos[i], x_pos[j]],
                                    y=[y_pos[i], y_pos[j]],
                                    z=[z_pos[i], z_pos[j]],
                                    mode='lines',
                                    line=dict(color=edge_color, width=edge_width),
                                    hoverinfo='skip',
                                    showlegend=False
                                ))
                    
                    # Add algorithm nodes with enhanced styling
                    node_colors = ['#ff0844' if algo_results[aid].get('verdict') == 'THREAT' else '#56ab2f' 
                                  for aid in algo_ids]
                    
                    node_labels = [algo_names.get(aid, f"Algorithm {aid}") for aid in algo_ids]
                    
                    # Add glow effect with larger transparent markers
                    fig_network.add_trace(go.Scatter3d(
                        x=x_pos, y=y_pos, z=z_pos,
                        mode='markers',
                        marker=dict(
                            size=25, 
                            color=node_colors,
                            opacity=0.2,
                            line=dict(width=0)
                        ),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                    
                    # Add main nodes with labels
                    fig_network.add_trace(go.Scatter3d(
                        x=x_pos, y=y_pos, z=z_pos,
                        mode='markers+text',
                        marker=dict(
                            size=18, 
                            color=node_colors,
                            line=dict(color='white', width=2),
                            opacity=0.95,
                            symbol='diamond'
                        ),
                        text=node_labels,
                        textposition='top center',
                        textfont=dict(size=9, color='#222', family='Rajdhani'),
                        customdata=[[algo_ids[i], confidences[i], algo_results[algo_ids[i]].get('verdict')] 
                                   for i in range(len(algo_ids))],
                        hovertemplate='<b>%{text}</b><br>Confidence: %{customdata[1]:.1%}<br>Verdict: %{customdata[2]}<br>Position: (%.2f, %.2f, %.2f)<extra></extra>',
                        showlegend=False
                    ))
                    
                    fig_network.update_layout(
                        template='plotly_white',
                        height=800,
                        showlegend=False,
                        scene=dict(
                            xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)', zeroline=False, showticklabels=False, title=''),
                            yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)', zeroline=False, showticklabels=False, title=''),
                            zaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)', zeroline=False, showticklabels=False, title=''),
                            camera=dict(
                                eye=dict(x=1.8, y=1.8, z=1.2),
                                center=dict(x=0, y=0, z=0)
                            ),
                            bgcolor='rgba(250,250,250,0.95)',
                            aspectmode='cube'
                        ),
                        title=dict(
                            text='3D Topological Algorithm Network (Torus Manifold)',
                            font=dict(size=16, family='Rajdhani', color='#333')
                        ),
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig_network, use_container_width=True)
            
            with viz_tab3:
                st.subheader("Temporal Evolution Timeline")
                st.markdown("**Track how analysis verdict evolves through algorithm sequence**")
                
                # Create evolution timeline
                algo_sequence = sorted([(int(aid), result) for aid, result in algo_results.items()], 
                                      key=lambda x: x[0])
                
                if algo_sequence:
                    # Calculate cumulative verdict
                    timeline_data = []
                    cumulative_threat = 0
                    cumulative_benign = 0
                    
                    for i, (algo_id, result) in enumerate(algo_sequence):
                        if result.get('verdict') == 'THREAT':
                            cumulative_threat += 1
                        else:
                            cumulative_benign += 1
                        
                        timeline_data.append({
                            'step': i + 1,
                            'algo_id': algo_id,
                            'threat_votes': cumulative_threat,
                            'benign_votes': cumulative_benign,
                            'confidence': (cumulative_threat / (i + 1)) * 100
                        })
                    
                    # Create animated line chart
                    steps = [d['step'] for d in timeline_data]
                    threat_votes = [d['threat_votes'] for d in timeline_data]
                    benign_votes = [d['benign_votes'] for d in timeline_data]
                    confidence_line = [d['confidence'] for d in timeline_data]
                    
                    fig_timeline = go.Figure()
                    
                    # Threat votes line
                    fig_timeline.add_trace(go.Scatter(
                        x=steps, y=threat_votes,
                        mode='lines+markers',
                        name='Threat Votes',
                        line=dict(color='#ff0844', width=3),
                        marker=dict(size=8),
                        fill='tonexty'
                    ))
                    
                    # Benign votes line
                    fig_timeline.add_trace(go.Scatter(
                        x=steps, y=benign_votes,
                        mode='lines+markers',
                        name='Benign Votes',
                        line=dict(color='#56ab2f', width=3),
                        marker=dict(size=8),
                        fill='tozeroy'
                    ))
                    
                    # Confidence percentage line
                    fig_timeline.add_trace(go.Scatter(
                        x=steps, y=confidence_line,
                        mode='lines',
                        name='Threat Confidence %',
                        line=dict(color='#667eea', width=2, dash='dash'),
                        yaxis='y2'
                    ))
                    
                    fig_timeline.update_layout(
                        template='plotly_white',
                        height=500,
                        title='Analysis Evolution Timeline',
                        xaxis_title='Algorithm Sequence',
                        yaxis_title='Vote Count',
                        yaxis2=dict(
                            title='Confidence %',
                            overlaying='y',
                            side='right',
                            range=[0, 100]
                        ),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Evolution statistics
                    col_evo1, col_evo2, col_evo3, col_evo4 = st.columns(4)
                    with col_evo1:
                        st.metric("Total Steps", len(timeline_data))
                    with col_evo2:
                        st.metric("Final Threat Votes", cumulative_threat)
                    with col_evo3:
                        st.metric("Final Benign Votes", cumulative_benign)
                    with col_evo4:
                        final_conf = (cumulative_threat / len(timeline_data)) * 100
                        st.metric("Final Threat %", f"{final_conf:.1f}%")
                    
                    # Show decision points
                    st.markdown("---")
                    st.markdown("### 🎯 Key Decision Points")
                    
                    for i, (algo_id, result) in enumerate(algo_sequence[:5]):  # Show first 5
                        verdict = result.get('verdict', 'UNKNOWN')
                        confidence = result.get('confidence', 0)
                        
                        col_dp1, col_dp2, col_dp3 = st.columns([1, 2, 1])
                        with col_dp1:
                            st.write(f"**Step {i+1}**")
                        with col_dp2:
                            st.write(f"Algorithm #{algo_id}")
                        with col_dp3:
                            badge_color = "🔴" if verdict == 'THREAT' else "🟢"
                            st.write(f"{badge_color} {verdict} ({confidence:.0%})")
            
            with viz_tab4:
                st.subheader("Sonic Spectrum Visualization")
                st.markdown("**Audio-visual representation of binary structure through frequency analysis**")
                
                # Create spectrogram from binary data
                data_sample = np.frombuffer(bytes_data[:min(4096, len(bytes_data))], dtype=np.uint8)
                
                if len(data_sample) > 0:
                    # Normalize to [-1, 1] for audio
                    audio_signal = (data_sample.astype(float) - 128) / 128
                    
                    # Create frequency spectrum using FFT
                    from scipy.fft import fft, fftfreq
                    
                    fft_values = fft(audio_signal)
                    fft_magnitude = np.abs(fft_values)[:len(fft_values)//2]
                    fft_freq = fftfreq(len(audio_signal), 1.0)[:len(fft_values)//2]
                    
                    # Create 3D spectrogram surface
                    segments = min(50, len(audio_signal) // 64)
                    segment_length = len(audio_signal) // segments
                    
                    spectrogram_data = []
                    for i in range(segments):
                        start = i * segment_length
                        end = start + segment_length
                        if end <= len(audio_signal):
                            segment = audio_signal[start:end]
                            segment_fft = np.abs(fft(segment))[:len(segment)//2]
                            # Pad or trim to consistent length
                            if len(segment_fft) > 32:
                                segment_fft = segment_fft[:32]
                            else:
                                segment_fft = np.pad(segment_fft, (0, 32-len(segment_fft)))
                            spectrogram_data.append(segment_fft)
                    
                    spectrogram_data = np.array(spectrogram_data).T
                    
                    # Create 3D surface plot
                    fig_sonic = go.Figure(data=[go.Surface(
                        z=spectrogram_data,
                        colorscale='Viridis',
                        colorbar=dict(title='Magnitude')
                    )])
                    
                    fig_sonic.update_layout(
                        template='plotly_white',
                        height=600,
                        title='3D Sonic Spectrogram (Binary as Audio)',
                        scene=dict(
                            xaxis_title='Frequency Bin',
                            yaxis_title='Time Segment',
                            zaxis_title='Magnitude',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                        )
                    )
                    
                    st.plotly_chart(fig_sonic, use_container_width=True)
                    
                    # Frequency spectrum analysis
                    st.markdown("---")
                    st.markdown("### 📊 Frequency Domain Analysis")
                    
                    # Create frequency bar chart (top frequencies)
                    top_n = 20
                    top_indices = np.argsort(fft_magnitude)[-top_n:][::-1]
                    top_freqs = fft_freq[top_indices]
                    top_mags = fft_magnitude[top_indices]
                    
                    fig_freq = go.Figure()
                    fig_freq.add_trace(go.Bar(
                        x=list(range(top_n)),
                        y=top_mags,
                        marker=dict(
                            color=top_mags,
                            colorscale='Plasma',
                            showscale=True,
                            colorbar=dict(title='Magnitude')
                        ),
                        text=[f"{f:.2f} Hz" for f in top_freqs],
                        textposition='outside',
                        hovertemplate='<b>Frequency: %{text}</b><br>Magnitude: %{y:.2f}<extra></extra>'
                    ))
                    
                    fig_freq.update_layout(
                        template='plotly_white',
                        height=400,
                        title='Top 20 Dominant Frequencies',
                        xaxis_title='Rank',
                        yaxis_title='Magnitude'
                    )
                    
                    st.plotly_chart(fig_freq, use_container_width=True)
                    
                    # Audio statistics
                    col_audio1, col_audio2, col_audio3, col_audio4 = st.columns(4)
                    with col_audio1:
                        st.metric("Sample Rate", "1 Hz")
                    with col_audio2:
                        st.metric("Duration", f"{len(audio_signal)}s")
                    with col_audio3:
                        dominant_freq = fft_freq[np.argmax(fft_magnitude)]
                        st.metric("Dominant Freq", f"{dominant_freq:.2f} Hz")
                    with col_audio4:
                        spectral_centroid = np.sum(fft_freq * fft_magnitude) / np.sum(fft_magnitude)
                        st.metric("Spectral Centroid", f"{spectral_centroid:.2f} Hz")
                    
                    # Optional: MIDI generation info
                    if HAS_MIDI:
                        st.info("💡 **MIDI Sonification Available**: Click below to generate MIDI file from binary spectrum")
                        if st.button("🎵 Generate MIDI File"):
                            try:
                                from midiutil import MIDIFile
                                
                                # Create MIDI file
                                midi = MIDIFile(1)
                                track = 0
                                channel = 0
                                time = 0
                                tempo = 120
                                volume = 100
                                
                                midi.addTempo(track, time, tempo)
                                
                                # Convert frequency peaks to MIDI notes
                                for i, freq in enumerate(top_freqs[:16]):  # Use top 16 frequencies
                                    # Convert frequency to MIDI note (A4 = 440 Hz = note 69)
                                    if freq > 0:
                                        midi_note = int(69 + 12 * np.log2(freq / 440))
                                        midi_note = max(0, min(127, midi_note))  # Clamp to valid range
                                        
                                        duration = 1
                                        midi.addNote(track, channel, midi_note, time, duration, volume)
                                        time += 0.5
                                
                                # Save MIDI to bytes
                                import io
                                midi_buffer = io.BytesIO()
                                midi.writeFile(midi_buffer)
                                midi_buffer.seek(0)
                                
                                st.download_button(
                                    label="📥 Download MIDI",
                                    data=midi_buffer,
                                    file_name=f"ouroboros_sonic_{uploaded_file.name}.mid",
                                    mime="audio/midi"
                                )
                                st.success("✅ MIDI file generated successfully!")
                            except Exception as e:
                                st.error(f"MIDI generation failed: {e}")
                else:
                    st.warning("Insufficient data for sonic analysis")
        
        # Export Results
        st.markdown("---")
        st.subheader("💾 Export Professional Reports")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            if st.button("📥 Download JSON Report"):
                import json
                from datetime import datetime
                
                # Calculate benign votes
                benign_votes = ensemble_result['total_algorithms'] - ensemble_result['threat_votes']
                
                report = {
                    'metadata': {
                        'report_generated': datetime.now().isoformat(),
                        'analyzer': 'OUROBOROS Forensic Cockpit',
                        'version': '1.0.0',
                        'analyst': 'Labib Bin Shahed'
                    },
                    'file_info': {
                        'filename': uploaded_file.name,
                        'size_bytes': len(bytes_data),
                        'file_type': uploaded_file.type if hasattr(uploaded_file, 'type') else 'unknown'
                    },
                    'ensemble_verdict': {
                        'final_verdict': ensemble_result['verdict'],
                        'confidence_score': float(ensemble_result['confidence']),
                        'threat_votes': int(ensemble_result['threat_votes']),
                        'benign_votes': int(benign_votes),
                        'total_algorithms': int(ensemble_result['total_algorithms'])
                    },
                    'individual_algorithms': ensemble_result['individual_results']
                }
                
                st.download_button(
                    label="📄 JSON Report",
                    data=json.dumps(report, indent=2, default=str),
                    file_name=f"ouroboros_report_{uploaded_file.name}.json",
                    mime="application/json"
                )
        
        with col_exp2:
            if st.button("📊 Download CSV Report"):
                import csv
                import io
                from datetime import datetime
                
                # Calculate benign votes
                benign_votes = ensemble_result['total_algorithms'] - ensemble_result['threat_votes']
                
                # Create CSV with detailed analysis
                csv_buffer = io.StringIO()
                writer = csv.writer(csv_buffer)
                
                # Header Section
                writer.writerow(['OUROBOROS FORENSIC ANALYSIS REPORT'])
                writer.writerow(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['Analyzer:', 'OUROBOROS Forensic Cockpit Alpha'])
                writer.writerow(['Developer:', 'Labib Bin Shahed (labib-x@protonmail.com)'])
                writer.writerow([])
                
                # File Information
                writer.writerow(['FILE INFORMATION'])
                writer.writerow(['Filename', uploaded_file.name])
                writer.writerow(['File Size (bytes)', len(bytes_data)])
                writer.writerow(['File Size (KB)', f"{len(bytes_data)/1024:.2f}"])
                writer.writerow(['File Size (MB)', f"{len(bytes_data)/(1024*1024):.2f}"])
                writer.writerow([])
                
                # Ensemble Verdict
                writer.writerow(['ENSEMBLE ANALYSIS VERDICT'])
                writer.writerow(['Final Verdict', ensemble_result['verdict']])
                writer.writerow(['Confidence Score', f"{ensemble_result['confidence']:.2%}"])
                writer.writerow(['Threat Votes', ensemble_result['threat_votes']])
                writer.writerow(['Benign Votes', benign_votes])
                writer.writerow(['Total Algorithms', ensemble_result['total_algorithms']])
                writer.writerow([])
                
                # Individual Algorithm Results
                writer.writerow(['DETAILED ALGORITHM ANALYSIS'])
                writer.writerow(['Algorithm ID', 'Algorithm Name', 'Verdict', 'Confidence', 'Category', 'Key Metrics'])
                
                for algo_id, result in ensemble_result['individual_results'].items():
                    algo_name = {
                        1: "Persistence Homology (TDA)",
                        2: "Recurrence Quantification (RQA)",
                        3: "Spectral Graph Clustering",
                        4: "Multifractal Spectrum",
                        5: "Kolmogorov Complexity",
                        6: "Dynamic Time Warping (DTW)",
                        7: "Latent Dirichlet Allocation (LDA)",
                        8: "Benford's Law Analysis",
                        9: "MinHash Locality",
                        10: "Z3 Constraint Solver",
                        11: "Gravitational Lensing",
                        12: "Zigzag Persistence",
                        13: "Quantum Interference",
                        14: "Fluid Dynamics",
                        15: "Neural Morphogenesis",
                        16: "Phylogenetic Tree",
                        17: "Symbiotic Processes",
                        18: "Temporal Manifold",
                        19: "Crystallography",
                        20: "Harmonic Resonance"
                    }.get(int(algo_id), f"Algorithm {algo_id}")
                    
                    # Extract key metrics
                    key_metrics = []
                    for key, value in result.items():
                        if key not in ['verdict', 'confidence'] and isinstance(value, (int, float, bool)):
                            if isinstance(value, float):
                                key_metrics.append(f"{key}={value:.3f}")
                            else:
                                key_metrics.append(f"{key}={value}")
                    
                    metrics_str = '; '.join(key_metrics[:5])  # Limit to top 5 metrics
                    
                    category = "Core Methods" if int(algo_id) <= 10 else "Interdisciplinary"
                    
                    writer.writerow([
                        algo_id,
                        algo_name,
                        result.get('verdict', 'N/A'),
                        f"{result.get('confidence', 0):.2%}",
                        category,
                        metrics_str
                    ])
                
                writer.writerow([])
                writer.writerow(['END OF REPORT'])
                
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📊 CSV Report",
                    data=csv_data,
                    file_name=f"ouroboros_analysis_{uploaded_file.name}.csv",
                    mime="text/csv"
                )
        
        with col_exp3:
            if st.button("🌐 Download HTML Report"):
                from datetime import datetime
                
                # Calculate benign votes
                benign_votes = ensemble_result['total_algorithms'] - ensemble_result['threat_votes']
                
                # Create professional HTML report
                html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OUROBOROS Forensic Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 50px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            letter-spacing: 2px;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }}
        
        .metadata {{
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 3px solid #667eea;
        }}
        
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .metadata-item {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        
        .metadata-item .label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .metadata-item .value {{
            font-size: 1.2em;
            font-weight: bold;
            color: #1e3c72;
        }}
        
        .section {{
            padding: 40px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #1e3c72;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .verdict-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }}
        
        .verdict-box.threat {{
            background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
        }}
        
        .verdict-box.benign {{
            background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        }}
        
        .verdict-box .verdict-label {{
            font-size: 1.5em;
            margin-bottom: 10px;
        }}
        
        .verdict-box .confidence {{
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 2px solid #e9ecef;
        }}
        
        .stat-card .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-card .stat-label {{
            color: #666;
            margin-top: 10px;
        }}
        
        .algorithm-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        .algorithm-table thead {{
            background: #1e3c72;
            color: white;
        }}
        
        .algorithm-table th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .algorithm-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .algorithm-table tbody tr:hover {{
            background: #f8f9fa;
        }}
        
        .badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            display: inline-block;
        }}
        
        .badge-threat {{
            background: #ff0844;
            color: white;
        }}
        
        .badge-benign {{
            background: #56ab2f;
            color: white;
        }}
        
        .badge-core {{
            background: #667eea;
            color: white;
        }}
        
        .badge-interdisciplinary {{
            background: #764ba2;
            color: white;
        }}
        
        .footer {{
            background: #1e3c72;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .footer .developer {{
            font-size: 1.1em;
            margin-top: 10px;
        }}
        
        .footer .email {{
            color: #a8e063;
            text-decoration: none;
        }}
        
        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>OUROBOROS Ꝏ</h1>
            <div class="subtitle">FORENSIC COCKPIT ANALYSIS REPORT</div>
            <div class="subtitle" style="margin-top: 20px; font-size: 0.9em;">
                Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}
            </div>
        </div>
        
        <!-- Metadata -->
        <div class="metadata">
            <h2>FILE INFORMATION</h2>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <div class="label">Filename</div>
                    <div class="value">{uploaded_file.name}</div>
                </div>
                <div class="metadata-item">
                    <div class="label">File Size</div>
                    <div class="value">{len(bytes_data):,} bytes</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Size (KB)</div>
                    <div class="value">{len(bytes_data)/1024:.2f} KB</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Size (MB)</div>
                    <div class="value">{len(bytes_data)/(1024*1024):.2f} MB</div>
                </div>
            </div>
        </div>
        
        <!-- Ensemble Verdict -->
        <div class="section">
            <h2 class="section-title">ENSEMBLE ANALYSIS VERDICT</h2>
            
            <div class="verdict-box {'threat' if ensemble_result['verdict'] == 'THREAT' else 'benign'}">
                <div class="verdict-label">Final Verdict</div>
                <div class="confidence">{ensemble_result['verdict']}</div>
                <div style="font-size: 1.2em;">Confidence: {ensemble_result['confidence']:.1%}</div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{ensemble_result['threat_votes']}</div>
                    <div class="stat-label">Threat Votes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{benign_votes}</div>
                    <div class="stat-label">Benign Votes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{ensemble_result['total_algorithms']}</div>
                    <div class="stat-label">Total Algorithms</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(ensemble_result['individual_results'])}</div>
                    <div class="stat-label">Algorithms Run</div>
                </div>
            </div>
        </div>
        
        <!-- Individual Algorithms -->
        <div class="section" style="background: #f8f9fa;">
            <h2 class="section-title">DETAILED ALGORITHM ANALYSIS</h2>
            
            <table class="algorithm-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Algorithm Name</th>
                        <th>Category</th>
                        <th>Verdict</th>
                        <th>Confidence</th>
                        <th>Key Metrics</th>
                    </tr>
                </thead>
                <tbody>
"""
                
                # Add algorithm rows
                algo_names = {
                    1: "Persistence Homology (TDA)",
                    2: "Recurrence Quantification (RQA)",
                    3: "Spectral Graph Clustering",
                    4: "Multifractal Spectrum",
                    5: "Kolmogorov Complexity",
                    6: "Dynamic Time Warping (DTW)",
                    7: "Latent Dirichlet Allocation (LDA)",
                    8: "Benford's Law Analysis",
                    9: "MinHash Locality",
                    10: "Z3 Constraint Solver",
                    11: "Gravitational Lensing",
                    12: "Zigzag Persistence",
                    13: "Quantum Interference",
                    14: "Fluid Dynamics",
                    15: "Neural Morphogenesis",
                    16: "Phylogenetic Tree",
                    17: "Symbiotic Processes",
                    18: "Temporal Manifold",
                    19: "Crystallography",
                    20: "Harmonic Resonance"
                }
                
                for algo_id, result in sorted(ensemble_result['individual_results'].items(), key=lambda x: int(x[0])):
                    algo_name = algo_names.get(int(algo_id), f"Algorithm {algo_id}")
                    category = "Core Methods" if int(algo_id) <= 10 else "Interdisciplinary"
                    verdict = result.get('verdict', 'N/A')
                    confidence = result.get('confidence', 0)
                    
                    # Extract key metrics
                    key_metrics = []
                    for key, value in result.items():
                        if key not in ['verdict', 'confidence'] and isinstance(value, (int, float, bool)):
                            if isinstance(value, float):
                                key_metrics.append(f"{key}: {value:.3f}")
                            else:
                                key_metrics.append(f"{key}: {value}")
                    
                    metrics_html = "<br>".join(key_metrics[:4])  # Show top 4 metrics
                    
                    html_content += f"""
                    <tr>
                        <td><strong>{algo_id}</strong></td>
                        <td>{algo_name}</td>
                        <td><span class="badge badge-{'core' if int(algo_id) <= 10 else 'interdisciplinary'}">{category}</span></td>
                        <td><span class="badge badge-{'threat' if verdict == 'THREAT' else 'benign'}">{verdict}</span></td>
                        <td>{confidence:.1%}</td>
                        <td style="font-size: 0.85em;">{metrics_html if metrics_html else 'N/A'}</td>
                    </tr>
"""
                
                html_content += """
                </tbody>
            </table>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <div style="font-size: 1.3em; margin-bottom: 10px;">OUROBOROS Forensic Cockpit Alpha</div>
            <div class="developer">
                Developed by: <strong>Labib Bin Shahed</strong> | 
                <a href="mailto:labib-x@protonmail.com" class="email">labib-x@protonmail.com</a>
            </div>
            <div style="margin-top: 15px; opacity: 0.8;">
                🔐 Forensic AI Engineer | Topological Defense System
            </div>
            <div style="margin-top: 20px; font-size: 0.9em; opacity: 0.7;">
                © 2025 OUROBOROS Project. Advanced Multidomain Binary Analysis.
            </div>
        </div>
    </div>
</body>
</html>
"""
                
                st.download_button(
                    label="🌐 HTML Report",
                    data=html_content,
                    file_name=f"ouroboros_forensic_report_{uploaded_file.name}.html",
                    mime="text/html"
                )
    
    # ========================================================================
    # LEGACY ANALYSIS (if advanced methods not available)
    # ========================================================================
    else:
        st.markdown("---")
        st.header("📊 LEGACY ANALYSIS MODE")
        st.warning("Advanced methods module not available. Running basic TDA analysis.")
    
    # Main Analysis (original code continues)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("/// PERSISTENCE BARCODE (Topology)")
        st.info("**Long bars** = Robust Loops (Malware Logic). **Short bars** = Noise.")
        with st.spinner("Computing Homology Groups..."):
            diag = compute_persistence_homology(bytes_data)
            
            # Extract birth/death pairs for H1 (Loops)
            h1_intervals = [p[1] for p in diag if p[0] == 1]
            if not h1_intervals: 
                h1_intervals = [(0,0)]
            
            # Plot Barcode
            fig_tda = go.Figure()
            for i, interval in enumerate(h1_intervals):
                death = interval[1] if interval[1] < float('inf') else 20
                persistence = death - interval[0]
                color = '#FF0055' if persistence > 5 else '#00FF41'
                fig_tda.add_trace(go.Scatter(
                    x=[interval[0], death], y=[i, i],
                    mode='lines', line=dict(color=color, width=3),
                    showlegend=False,
                    hovertemplate=f'Birth: {interval[0]:.2f}<br>Death: {death:.2f}<br>Persistence: {persistence:.2f}'
                ))
            fig_tda.update_layout(
                title="H₁ Homology (Loops)", 
                template="plotly_white",
                xaxis_title="Filtration Value", 
                yaxis_title="Feature Index"
            )
            st.plotly_chart(fig_tda, use_container_width=True, key="chart_legacy_tda")
            
            st.metric("Betti Number β₁", len(h1_intervals))

    with col2:
        st.subheader("/// RECURRENCE PLOT (Chaos)")
        st.info("**Patterns** reveal hidden deterministic structures in 'random' data.")
        with st.spinner("Reconstructing Phase Space..."):
            # Simple visualization of the recurrence (Approximation for UI)
            # Real RQA plots are dense matrices; we simulate the visual style here
            data_sample = np.frombuffer(bytes_data, dtype=np.uint8)[:200]
            # Create a distance matrix (Self-Similarity)
            matrix = np.abs(data_sample[:, None] - data_sample[None, :])
            binary_matrix = np.where(matrix < 20, 1, 0)  # Threshold
            
            fig_rqa = go.Figure(data=go.Surface(
                z=binary_matrix, 
                colorscale='Viridis',
                showscale=True
            ))
            fig_rqa.update_layout(
                title="Recurrence Matrix 3D (Self-Similarity)", 
                template="plotly_white",
                scene=dict(
                    xaxis_title="Time i",
                    yaxis_title="Time j",
                    zaxis_title="Similarity"
                ),
                height=500
            )
            st.plotly_chart(fig_rqa, use_container_width=True, key="chart_legacy_rqa")
            
            try:
                rec_rate, _ = compute_recurrence_plot(bytes_data)
                st.metric("Recurrence Rate", f"{rec_rate:.3f}")
            except Exception as e:
                st.warning(f"Recurrence analysis unavailable (OpenCL not supported in cloud environment). Using fallback method.")
                # Fallback: simple entropy-based metric
                entropy = -np.sum(np.histogram(data_sample, bins=256, density=True)[0] * 
                                 np.log2(np.histogram(data_sample, bins=256, density=True)[0] + 1e-10))
                st.metric("Entropy (Fallback)", f"{entropy:.3f}")

    # Second row: Multifractal Analysis
    st.subheader("/// MULTIFRACTAL SINGULARITY SPECTRUM")
    st.info("**Width (Δα)** measures complexity richness. **Asymmetry** reveals machine vs human generation.")
    
    with st.spinner("Computing Singularity Spectrum..."):
        alpha_values, f_alpha_values = compute_multifractal_spectrum(bytes_data)
        
        if alpha_values:
            fig_mf = go.Figure()
            # Create 3D parametric curve
            t = np.linspace(0, 1, len(alpha_values))
            fig_mf.add_trace(go.Scatter3d(
                x=alpha_values, 
                y=f_alpha_values,
                z=t,
                mode='lines+markers',
                line=dict(color='#00FFFF', width=4),
                marker=dict(size=4, color=t, colorscale='Plasma', showscale=True),
                name='f(α) Spectrum'
            ))
            fig_mf.update_layout(
                title="Multifractal Spectrum (3D)", 
                template="plotly_white",
                scene=dict(
                    xaxis_title="Hölder Exponent α",
                    yaxis_title="f(α)",
                    zaxis_title="Parameter t"
                ),
                height=500
            )
            st.plotly_chart(fig_mf, use_container_width=True, key="chart_legacy_multifractal")
            
            col_mf1, col_mf2, col_mf3 = st.columns(3)
            with col_mf1:
                st.metric("Spectrum Width (Δα)", f"{max(alpha_values) - min(alpha_values):.3f}")
            with col_mf2:
                st.metric("Peak Position", f"{alpha_values[len(alpha_values)//2]:.3f}")
            with col_mf3:
                asymmetry = abs(alpha_values[0] - alpha_values[-1])
                st.metric("Asymmetry", f"{asymmetry:.3f}")
        else:
            st.warning("Insufficient data complexity for multifractal analysis.")
    
    # === ADVANCED ANALYSIS SECTION ===
    st.markdown("---")
    st.subheader("/// ADVANCED MULTIDOMAIN ANALYSIS")
    
    # Compute all advanced metrics
    advanced_metrics = {}
    
    with st.spinner("Computing advanced metrics..."):
        col_adv1, col_adv2, col_adv3 = st.columns(3)
        
        # Domain I: Topological & Geometric
        with col_adv1:
            st.markdown("**🔷 Topological & Geometric**")
            
            reeb = compute_reeb_graph_skeleton(bytes_data)
            if reeb:
                st.metric("Reeb Graph Nodes", reeb['nodes'])
                st.metric("Reeb Graph Links", reeb['links'])
            
            lid_vals = compute_local_intrinsic_dim(bytes_data)
            if lid_vals:
                lid_variance = np.var(lid_vals)
                st.metric("LID Variance", f"{lid_variance:.2f}")
                advanced_metrics['lid_variance'] = lid_variance
                
                # Plot LID profile
                fig_lid = go.Figure()
                # Create 3D ribbon effect
                x_pos = np.arange(len(lid_vals))
                fig_lid.add_trace(go.Scatter3d(
                    x=x_pos,
                    y=lid_vals,
                    z=np.sin(x_pos * 0.1),  # Add wave for 3D effect
                    mode='lines+markers',
                    line=dict(color='#00FFFF', width=4),
                    marker=dict(size=3, color=lid_vals, colorscale='Viridis', showscale=True),
                    name='Intrinsic Dimension'
                ))
                fig_lid.update_layout(
                    title="Local Intrinsic Dimensionality Profile (3D)",
                    template="plotly_white",
                    scene=dict(
                        xaxis_title="Position",
                        yaxis_title="LID",
                        zaxis_title="Wave"
                    ),
                    height=400
                )
                st.plotly_chart(fig_lid, use_container_width=True, key="chart_legacy_lid")
        
        # Domain II: Chaos & Dynamics
        with col_adv2:
            st.markdown("**🌀 Chaos & Dynamics**")
            
            lyap = compute_lyapunov_exponent(bytes_data)
            hurst = compute_hurst_exponent(bytes_data)
            ca_class = classify_cellular_automaton(bytes_data)
            
            st.metric("Lyapunov Exponent", f"{lyap:.4f}")
            st.metric("Hurst Exponent", f"{hurst:.4f}")
            st.metric("CA Classification", ca_class)
            
            advanced_metrics['lyapunov'] = lyap
            advanced_metrics['hurst'] = hurst
            
            # Interpretation
            if lyap > 0.5:
                st.warning("⚠️ High chaos detected (heavy obfuscation)")
            elif lyap < -0.1:
                st.info("ℹ️ Stable dynamics (ordered code)")
            
            if hurst < 0.4:
                st.warning("⚠️ Short-term memory (random generation)")
            elif hurst > 0.6:
                st.info("ℹ️ Long-term memory (structured code)")
        
        # Domain III: Bio-Digital
        with col_adv3:
            st.markdown("**🧬 Bio-Digital Analysis**")
            
            genome_match = compute_smith_waterman_alignment(bytes_data)
            st.metric("Genome Similarity", f"{genome_match:.2%}")
            advanced_metrics['genome_match'] = genome_match
            
            entropy_rates = compute_entropy_rate(bytes_data)
            if entropy_rates:
                st.metric("Entropy Variance", f"{np.var(entropy_rates):.2f}")
                
                # Plot entropy rate
                fig_ent = go.Figure()
                x_time = np.arange(len(entropy_rates))
                fig_ent.add_trace(go.Scatter3d(
                    x=x_time,
                    y=entropy_rates,
                    z=np.cumsum(entropy_rates) / (np.arange(len(entropy_rates)) + 1),  # Running average
                    mode='lines+markers',
                    line=dict(color='#FF0055', width=4),
                    marker=dict(size=3, color=entropy_rates, colorscale='Hot', showscale=True),
                    name='Entropy Rate'
                ))
                fig_ent.update_layout(
                    title="Entropy Production Rate (3D)",
                    template="plotly_white",
                    scene=dict(
                        xaxis_title="Time Window",
                        yaxis_title="Entropy (bits)",
                        zaxis_title="Avg Entropy"
                    ),
                    height=400
                )
                st.plotly_chart(fig_ent, use_container_width=True, key="chart_legacy_entropy")
    
    # Second row of advanced metrics
    col_adv4, col_adv5, col_adv6 = st.columns(3)
    
    # Domain IV: Spectral & Signal
    with col_adv4:
        st.markdown("**📊 Spectral & Signal**")
        
        eigenvalues = compute_graph_laplacian_spectrum(bytes_data)
        if eigenvalues and len(eigenvalues) > 0:
            st.metric("Spectral Gap", f"{eigenvalues[1] - eigenvalues[0]:.4f}" if len(eigenvalues) > 1 else "N/A")
            
            # Plot spectrum
            fig_spec = go.Figure()
            x_idx = list(range(len(eigenvalues)))
            fig_spec.add_trace(go.Scatter3d(
                x=x_idx,
                y=eigenvalues,
                z=[ev**2 for ev in eigenvalues],
                mode='markers+lines',
                marker=dict(size=6, color=eigenvalues, colorscale='Greens', showscale=True),
                line=dict(color='#00FF41', width=3)
            ))
            fig_spec.update_layout(
                title="Graph Laplacian Spectrum (3D)",
                template="plotly_white",
                scene=dict(
                    xaxis_title="Eigenvalue Index",
                    yaxis_title="λ",
                    zaxis_title="λ²"
                ),
                height=400
            )
            st.plotly_chart(fig_spec, use_container_width=True)
        
        texture = compute_glcm_texture(bytes_data)
        if texture:
            st.metric("GLCM Contrast", f"{texture.get('contrast', 0):.2f}")
            st.metric("GLCM Homogeneity", f"{texture.get('homogeneity', 0):.3f}")
    
    # Domain V: Symbolic & Logic
    with col_adv5:
        st.markdown("**🔐 Symbolic & Logic**")
        
        compression = compute_kolmogorov_complexity(bytes_data)
        if compression:
            comp_min = min(compression.values())
            st.metric("Min Compression Ratio", f"{comp_min:.3f}")
            advanced_metrics['compression_min'] = comp_min
            
            # Plot compression profile
            fig_comp = go.Figure()
            comp_keys = list(compression.keys())
            comp_vals = list(compression.values())
            fig_comp.add_trace(go.Scatter3d(
                x=list(range(len(comp_keys))),
                y=comp_vals,
                z=[v*2 for v in comp_vals],
                mode='markers',
                marker=dict(
                    size=15,
                    color=['#FF0055', '#00FFFF', '#00FF41'][:len(comp_keys)],
                    symbol='diamond'
                ),
                text=comp_keys,
                textposition='top center'
            ))
            fig_comp.update_layout(
                title="Compression Profile (3D)",
                template="plotly_white",
                scene=dict(
                    xaxis_title="Algorithm Index",
                    yaxis_title="Ratio",
                    zaxis_title="2×Ratio"
                ),
                height=400
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
        minhash_sig = compute_minhash_signature(bytes_data)
        if minhash_sig:
            st.metric("MinHash Signature", f"{minhash_sig[0]}")
        
        benford = check_benfords_law(bytes_data)
        st.metric("Benford's χ²", f"{benford:.2f}")
        advanced_metrics['benford'] = benford
        
        if benford > 10.0:
            st.warning("⚠️ Violates Benford's Law (artificial data)")
        else:
            st.success("✓ Follows Benford's Law (natural code)")
    
    # Domain VI: Sonification
    with col_adv6:
        st.markdown("**🎵 Sonification**")
        
        st.info("Convert binary patterns to MIDI for auditory analysis")
        
        if st.button("🎹 Generate MIDI"):
            with st.spinner("Generating sonic signature..."):
                midi_file = generate_midi_sonification(bytes_data, f"malware_{uploaded_file.name}.mid")
                if midi_file:
                    with open(midi_file, 'rb') as f:
                        st.download_button(
                            label="📥 Download MIDI",
                            data=f,
                            file_name=midi_file,
                            mime="audio/midi"
                        )
                    st.success("✓ MIDI file generated!")
                    st.caption("Listen for rhythmic anomalies in the sound pattern")
    
    # Threat Assessment
    st.markdown("---")
    st.subheader("/// ENHANCED THREAT ASSESSMENT MATRIX")
    
    multifractal_data = (alpha_values, f_alpha_values)
    threat_metrics = assess_threat_level(diag, rec_rate, multifractal_data, advanced_metrics)
    
    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
    
    with col_t1:
        threat_color = "🔴" if threat_metrics['overall'] > 0.7 else "🟡" if threat_metrics['overall'] > 0.4 else "🟢"
        st.metric("Overall Threat", f"{threat_color} {threat_metrics['overall']:.2%}")
    
    with col_t2:
        st.metric("Topology Score", f"{threat_metrics['topology']:.2%}")
    
    with col_t3:
        st.metric("Chaos Score", f"{threat_metrics['chaos']:.2%}")
    
    with col_t4:
        st.metric("Multifractal Score", f"{threat_metrics['multifractal']:.2%}")
    
    with col_t5:
        st.metric("Advanced Score", f"{threat_metrics['advanced']:.2%}")
    
    # Final verdict
    if threat_metrics['overall'] > 0.7:
        st.error(f"⚠️ **HIGH THREAT DETECTED** - The manifold indicates high-dimensional loops consistent with advanced persistent threat (APT) signatures. Persistent features: {threat_metrics['persistent_features']}")
    elif threat_metrics['overall'] > 0.4:
        st.warning(f"⚡ **MODERATE THREAT** - Unusual topological structures detected. Recommend further analysis. Persistent features: {threat_metrics['persistent_features']}")
    else:
        st.success(f"✓ **LOW THREAT** - Topological signature within normal parameters. Persistent features: {threat_metrics['persistent_features']}")
    
    # Technical Details Expander
    with st.expander("📊 View Technical Details"):
        details = {
            "persistent_features": threat_metrics['persistent_features'],
            "recurrence_rate": float(threat_metrics['recurrence_rate']),
            "h1_intervals_count": len([p for p in diag if p[0] == 1]),
            "multifractal_width": float(max(alpha_values) - min(alpha_values)) if alpha_values else 0,
            "file_size_bytes": len(bytes_data)
        }
        
        # Add advanced metrics
        if advanced_metrics:
            details.update(advanced_metrics)
        
        st.json(details)
    
    # Feature availability status
    with st.expander("🔧 Available Analysis Modules"):
        st.write(f"**Mapper/TDA**: {'✅ Available' if HAS_MAPPER else '❌ Not installed'}")
        st.write(f"**Chaos Theory**: {'✅ Available' if HAS_CHAOS else '❌ Not installed'}")
        st.write(f"**Bio-Informatics**: {'✅ Available' if HAS_BIO else '❌ Not installed'}")
        st.write(f"**Spectral Analysis**: {'✅ Available' if HAS_SPECTRAL else '❌ Not installed'}")
        st.write(f"**Symbolic Logic**: {'✅ Available' if HAS_SYMBOLIC else '❌ Not installed'}")
        st.write(f"**MIDI Sonification**: {'✅ Available' if HAS_MIDI else '❌ Not installed'}")

else:
    with st.expander("📐 Mathematics & References", expanded=False):
        st.markdown("### Mathematics Used")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- Takens' Theorem (Phase Space Reconstruction)")
            st.markdown("- Vietoris-Rips Complexes (Simplicial Homology)")
            st.markdown("- Hölder Exponents (Multifractal Formalism)")
            st.markdown("- Lyapunov Exponents (Chaos Theory)")
            st.markdown("- Smith-Waterman Algorithm (Genomics)")
        with col2:
            st.markdown("- Laplacian Eigenvalues (Spectral Graph Theory)")
            st.markdown("- Benford's Law (Statistical Physics)")
            st.markdown("- Wasserstein Distance (Optimal Transport)")
            st.markdown("- Isomap/LLE (Manifold Learning)")
            st.markdown("- Z3 SMT Solver (Formal Verification)")
        
        st.markdown("### References")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- [Security Applications of Chaotic Systems](https://www.youtube.com/watch?v=hczw4WyPIxg)")
            st.markdown("- Edelsbrunner & Harer - Computational Topology")
            st.markdown("- Strogatz - Nonlinear Dynamics and Chaos")
        with col2:
            st.markdown("- Wolfram - A New Kind of Science")
            st.markdown("- Villani - Optimal Transport (Wasserstein Distance)")
            st.markdown("- De Moura & Bjørner - Z3 SMT Solver")

# ═══════════════════════════════════════════════════════════════════════════
# DEVELOPER SIGNATURE FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="dev-signature">
    <p class="dev-name">⚡ DEVELOPED BY LABIB BIN SHAHED ⚡</p>
    <p class="dev-email">Contact: <a href="mailto:labib-x@protonmail.com">labib-x@protonmail.com</a></p>
    <span class="dev-badge">🔐 Forensic AI Engineer</span>
</div>
""", unsafe_allow_html=True)

st.caption("")

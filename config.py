# ---------------------------------------------------------
# config.py — Global configuration and constants
# ---------------------------------------------------------
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------
JSON_FOLDER       = '.'
INFO_DATEI        = 'dataset_info.xlsx'
VIDEO_SOURCE_PATH = r'data\processed'
RESULTS_FOLDER    = 'results'
SUMMARIZED_FILE   = os.path.join(RESULTS_FOLDER, 'results_summarized.xlsx')

# Top-level output directory for all plots
base_plot_folder = 'plots'
os.makedirs(base_plot_folder, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

# Maps technical model names to human-readable display names
BASE_MODEL_DISPLAY_NAMES = {
    'gemini-3-flash-preview': 'Gemini 3 Flash',
    'gpt5_2':                 'GPT-5.2',
    'qwen3.5-397b-a17b':      'Qwen 3.5',
    'kimi-k2.5':              'Kimi k2.5',
    'seed-2.0-lite':          'Seed 2.0 Lite',
}

# Ordering of prompt variants used in all tables and plots
VARIANT_ORDER = ['', '+I', '+T', '+I+T']

# Marker shapes for prompt variants in PCA scatter plots
VARIANT_MARKERS = {'': 'o', '+I': 's', '+T': '^', '+I+T': 'D'}

# Run identifiers
RUN_SUFFIXES = ["_1", "_2", "_3"]

# Metadata columns used across all analyses
META_COLS = ['dataset', 'gender', 'video_length', 'deepfake_category', 'deepfake_type', 'audio']

# ---------------------------------------------------------
# KEYWORD ANALYSIS CONFIGURATION
# ---------------------------------------------------------

# Domain-specific stop words for keyword analysis (extends sklearn English stop words)
DOMAIN_STOPS = list(ENGLISH_STOP_WORDS) + [
    'video', 'clip', 'footage', 'appears', 'seems', 'looks', 'shows',
    'detected', 'generated', 'ai', 'content', 'likely', 'probability',
    'confidence', 'based', 'analysis', 'observe', 'observed', 'visual',
    'anomalies', 'artifacts', 'signs', 'potential', 'deepfake'
]

# Anchor keyword definitions: shared by cluster_keywords() and analyze_region_coverage()
KEYWORD_ANCHORS = {
    # MARE facial regions
    'Skin':              'skin, cheek, forehead, complexion, dermal, face',
    'Nose':              'nose, nostril, nasal',
    'Mouth':             'mouth, lip, lips',
    'Teeth':             'tooth, teeth',
    'Eye':               'right-eye, left-eye, eye, ocular',
    'Eyebrow':           'right-eyebrow, left-eyebrow, eyebrow, brow',
    'Chin':              'chin, jaw, jawline, lower face',
    'Beard':             'beard, mustache, moustache, goatee',
    'Hairline':          'hairline, hair line, hair',
    'Ear':               'ear, ears',
    # Body regions
    'Head_Neck':         'neck, head, throat',
    'Torso':             'shoulder, torso, chest, arm, posture',
    'Hands':             'hand, hands, finger, fingers',
    # Background and temporal
    'Lighting':          'lighting, illumination, shadow, brightness, light source',
    'Scene':             'scene, background, environment, setting',
    'Temporal':          'flickering, temporal, inconsistency, frame rate',
    'General_Artifacts': 'edge, noise, blur, blending, compression, artifact, consistent, natural',
    # Audio features
    'Voice':             'voice, speech, pronunciation, accent, audio, sound, tone',
    'Lip_Sync':          'lip sync, synchronization, mouth movement, lipsync',
}

# Static hierarchy: maps Level-3 anchor → (Level-2, Level-1)
KEYWORD_HIERARCHY = {
    'Skin':              ('Face',       'Frame'),
    'Nose':              ('Face',       'Frame'),
    'Mouth':             ('Face',       'Frame'),
    'Teeth':             ('Face',       'Frame'),
    'Eye':               ('Face',       'Frame'),
    'Eyebrow':           ('Face',       'Frame'),
    'Chin':              ('Face',       'Frame'),
    'Beard':             ('Face',       'Frame'),
    'Hairline':          ('Face',       'Frame'),
    'Ear':               ('Face',       'Frame'),
    'Head_Neck':         ('Body',       'Frame'),
    'Torso':             ('Body',       'Frame'),
    'Hands':             ('Body',       'Frame'),
    'Lighting':          ('Background', 'Frame'),
    'Scene':             ('Background', 'Frame'),
    'Temporal':          ('Background', 'Frame'),
    'Voice':             ('Audio',      'Audio'),
    'Lip_Sync':          ('Audio',      'Audio'),
    'General_Artifacts': ('Background', 'Frame'),
}

# ---------------------------------------------------------
# GLOBAL PLOT STYLE
# ---------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.max_open_warning': 0,
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial'],
    'font.size':        15,
    'axes.titlesize':   15,
    'axes.labelsize':   15,
    'xtick.labelsize':  15,
    'ytick.labelsize':  15,
    'legend.fontsize':  15,
})

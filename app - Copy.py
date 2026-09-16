import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import Scikit-Learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================================
# 1. KONFIGURASI HALAMAN (LIGHT THEME)
# ==========================================
st.set_page_config(page_title="Edu Data Mining Pipeline", page_icon="📊", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main { background-color: #f8f9fc; color: #1a1a2e; font-family: 'Inter', sans-serif; }
    h1 { color: #1e3a5f; font-weight: 700; }
    h2 { color: #2c5282; font-weight: 600; }
    h3 { color: #2d3748; font-weight: 600; }
    
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #4C72B0;
    }
    
    .alert-leakage {
        background: linear-gradient(135deg, #fff5f5, #fed7d7);
        border: 1px solid #fc8181;
        border-left: 5px solid #e53e3e;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
        color: #742a2a;
        font-weight: 500;
    }
    .alert-ok {
        background: linear-gradient(135deg, #f0fff4, #c6f6d5);
        border: 1px solid #68d391;
        border-left: 5px solid #38a169;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
        color: #22543d;
        font-weight: 500;
    }
    .info-box {
        background: linear-gradient(135deg, #ebf8ff, #bee3f8);
        border: 1px solid #63b3ed;
        border-left: 5px solid #3182ce;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
        color: #2a4365;
    }
    
    div[data-testid="stTabs"] button {
        font-weight: 600;
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. FUNGSI PEMBUATAN DATASET BAWAAN
# ==========================================
@st.cache_data
def generate_student_alcohol_dataset(n=395):
    """Simulasi dataset Student Alcohol Consumption (UCI ML Repository).
    Referensi: P. Cortez and A. Silva (2008), 'Using Data Mining to Predict Secondary School Student Performance'."""
    np.random.seed(42)
    data = {
        'school': np.random.choice(['GP', 'MS'], n, p=[0.65, 0.35]),
        'sex': np.random.choice(['F', 'M'], n, p=[0.53, 0.47]),
        'age': np.random.choice([15, 16, 17, 18, 19, 20, 21, 22], n, p=[0.16, 0.24, 0.22, 0.20, 0.10, 0.05, 0.02, 0.01]),
        'address': np.random.choice(['U', 'R'], n, p=[0.78, 0.22]),
        'famsize': np.random.choice(['GT3', 'LE3'], n, p=[0.68, 0.32]),
        'Pstatus': np.random.choice(['T', 'A'], n, p=[0.83, 0.17]),
        'Medu': np.random.choice([0, 1, 2, 3, 4], n, p=[0.01, 0.15, 0.25, 0.30, 0.29]),
        'Fedu': np.random.choice([0, 1, 2, 3, 4], n, p=[0.02, 0.22, 0.28, 0.25, 0.23]),
        'traveltime': np.random.choice([1, 2, 3, 4], n, p=[0.55, 0.30, 0.10, 0.05]),
        'studytime': np.random.choice([1, 2, 3, 4], n, p=[0.25, 0.45, 0.20, 0.10]),
        'failures': np.random.choice([0, 1, 2, 3], n, p=[0.65, 0.20, 0.10, 0.05]),
        'schoolsup': np.random.choice(['yes', 'no'], n, p=[0.12, 0.88]),
        'famsup': np.random.choice(['yes', 'no'], n, p=[0.60, 0.40]),
        'paid': np.random.choice(['yes', 'no'], n, p=[0.46, 0.54]),
        'activities': np.random.choice(['yes', 'no'], n, p=[0.51, 0.49]),
        'higher': np.random.choice(['yes', 'no'], n, p=[0.90, 0.10]),
        'internet': np.random.choice(['yes', 'no'], n, p=[0.66, 0.34]),
        'romantic': np.random.choice(['yes', 'no'], n, p=[0.33, 0.67]),
        'famrel': np.random.choice([1, 2, 3, 4, 5], n, p=[0.03, 0.05, 0.17, 0.50, 0.25]),
        'freetime': np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.12, 0.40, 0.30, 0.13]),
        'goout': np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.20, 0.35, 0.25, 0.15]),
        'Dalc': np.random.choice([1, 2, 3, 4, 5], n, p=[0.60, 0.20, 0.10, 0.06, 0.04]),
        'Walc': np.random.choice([1, 2, 3, 4, 5], n, p=[0.35, 0.25, 0.20, 0.12, 0.08]),
        'health': np.random.choice([1, 2, 3, 4, 5], n, p=[0.10, 0.10, 0.25, 0.20, 0.35]),
        'absences': np.clip(np.random.exponential(5, n).astype(int), 0, 75),
    }
    # Generate G1, G2, G3 (3 kali assignment) dengan korelasi realistis
    base = 5 + data['Medu'] * 0.8 + data['studytime'] * 1.2 - data['failures'] * 2.5 - data['Dalc'] * 0.7 - data['Walc'] * 0.4
    noise1 = np.random.normal(0, 2.5, n)
    data['G1'] = np.clip((base + noise1 + np.random.normal(0, 1.5, n)).astype(int), 0, 20)
    data['G2'] = np.clip((base + noise1 * 0.8 + np.random.normal(0, 1.5, n)).astype(int), 0, 20)
    data['G3'] = np.clip((base + noise1 * 0.6 + np.random.normal(0, 2, n)).astype(int), 0, 20)
    return pd.DataFrame(data)


@st.cache_data
def generate_student_stress_dataset(n=500):
    """Simulasi dataset Student Stress Factors.
    Terinspirasi oleh dataset stress level dari Kaggle."""
    np.random.seed(123)
    data = {
        'anxiety_level': np.random.choice(range(0, 22), n),
        'self_esteem': np.random.choice(range(0, 31), n),
        'mental_health_history': np.random.choice([0, 1], n, p=[0.65, 0.35]),
        'depression': np.random.choice(range(0, 28), n),
        'headache': np.random.choice(range(0, 6), n),
        'blood_pressure': np.random.choice(range(1, 4), n),
        'sleep_quality': np.random.choice(range(0, 6), n),
        'breathing_problem': np.random.choice(range(0, 6), n),
        'noise_level': np.random.choice(range(0, 6), n),
        'living_conditions': np.random.choice(range(0, 6), n),
        'safety': np.random.choice(range(0, 6), n),
        'basic_needs': np.random.choice(range(0, 6), n),
        'academic_performance': np.random.choice(range(0, 6), n),
        'study_load': np.random.choice(range(0, 6), n),
        'teacher_student_relationship': np.random.choice(range(0, 6), n),
        'future_career_concerns': np.random.choice(range(0, 6), n),
        'social_support': np.random.choice(range(0, 4), n),
        'peer_pressure': np.random.choice(range(0, 6), n),
        'extracurricular_activities': np.random.choice(range(0, 6), n),
        'bullying': np.random.choice(range(0, 6), n),
    }
    # stress_level based on weighted combination
    stress_raw = (data['anxiety_level'] * 0.3 + data['depression'] * 0.25 +
                  data['study_load'] * 1.2 + data['peer_pressure'] * 0.8 -
                  data['self_esteem'] * 0.15 - data['social_support'] * 0.5 +
                  data['sleep_quality'] * 0.4 + np.random.normal(0, 2, n))
    # Normalize ke 0-2 range lalu ke kategori
    stress_norm = (stress_raw - stress_raw.min()) / (stress_raw.max() - stress_raw.min())
    data['stress_level'] = pd.cut(stress_norm, bins=3, labels=['Low', 'Medium', 'High']).astype(str)
    return pd.DataFrame(data)


# ==========================================
# 3. SIDEBAR: PENGATURAN PIPELINE
# ==========================================
with st.sidebar:
    st.title("⚙️ Konfigurasi Pipeline")

    st.subheader("📁 Pilih Dataset")
    dataset_source = st.radio(
        "Sumber Data:",
        ["📤 Upload Dataset Sendiri",
         "🎓 Students Performance (Kaggle)",
         "🍷 Student Alcohol Consumption (UCI)",
         "😰 Student Stress Factors (Kaggle)"],
        index=0
    )

    uploaded_file = None
    if dataset_source == "📤 Upload Dataset Sendiri":
        uploaded_file = st.file_uploader("Upload File (.csv / .xls / .xlsx)", type=['csv', 'xls', 'xlsx'])

    st.divider()
    st.subheader("🔧 Pengaturan Modeling")
    test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
    cv_folds = st.slider("Cross-Validation (K-Folds)", 2, 10, 5)
    scaler_choice = st.selectbox("Feature Scaler", ["StandardScaler", "MinMaxScaler", "None"])
    pass_threshold = st.number_input("Ambang Batas (Binary Classification)", value=10, help="Nilai >= threshold → kelas 1 (Positif)")

    st.divider()
    st.subheader("🧠 Algoritma Klasifikasi")
    algo_choices = st.multiselect(
        "Pilih Algoritma:",
        ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting"],
        default=["Logistic Regression", "Random Forest"]
    )

    do_hyperparameter = st.checkbox("Aktifkan Hyperparameter Tuning (GridSearchCV)", value=False)

    st.divider()
    st.caption("📚 Edu Data Mining Pipeline v2.0")
    st.caption("Dibuat untuk tugas Sains Data Edukasi Semester 5")


# ==========================================
# 4. KONTEN UTAMA
# ==========================================
st.title("📊 Educational Data Mining Pipeline")
st.markdown("Implementasi End-to-End: **Konteks Penelitian** ➔ **EDA** ➔ **Preprocessing** ➔ **Clustering** ➔ **Classification & Evaluation**")

# --- Load Data ---
df = None
dataset_name = ""

if dataset_source == "📤 Upload Dataset Sendiri":
    if uploaded_file is None:
        st.info("👋 Silakan pilih dataset bawaan di sidebar, atau upload dataset sendiri untuk memulai pipeline.")
        st.stop()
    else:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        dataset_name = uploaded_file.name

elif dataset_source == "🎓 Students Performance (Kaggle)":
    try:
        df = pd.read_csv("StudentsPerformance.csv")
        dataset_name = "StudentsPerformance.csv"
    except FileNotFoundError:
        st.error("File `StudentsPerformance.csv` tidak ditemukan di folder. Silakan upload manual.")
        st.stop()

elif dataset_source == "🍷 Student Alcohol Consumption (UCI)":
    df = generate_student_alcohol_dataset()
    dataset_name = "Student Alcohol Consumption (UCI - Simulated)"

elif dataset_source == "😰 Student Stress Factors (Kaggle)":
    df = generate_student_stress_dataset()
    dataset_name = "Student Stress Factors (Simulated)"

if df is None:
    st.stop()

st.success(f"✅ Dataset dimuat: **{dataset_name}** — {df.shape[0]} baris × {df.shape[1]} kolom")

# ==========================================
# 5. TABS UTAMA
# ==========================================
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📖 Latar Belakang",
    "🔍 EDA & Overview",
    "⚙️ Preprocessing",
    "🧩 Clustering",
    "🎯 Classification & Evaluasi"
])

# ---------------------------------------------------------
# TAB 0: LATAR BELAKANG & KONTEKS PENELITIAN
# ---------------------------------------------------------
with tab0:
    st.subheader("📖 Latar Belakang & Konteks Penelitian")

    st.markdown("""
    ### 1. Domain Masalah
    Pendidikan merupakan salah satu sektor vital yang menghasilkan data dalam jumlah besar — mulai dari 
    data demografis siswa, riwayat akademik, perilaku sosial, hingga kesehatan mental. **Educational Data Mining (EDM)** 
    adalah bidang yang menerapkan teknik data mining untuk mengekstrak pola bermakna dari data pendidikan, 
    dengan tujuan meningkatkan kualitas proses belajar-mengajar.

    ### 2. Masalah yang Ingin Diselesaikan
    Pipeline ini dirancang untuk menjawab pertanyaan-pertanyaan berikut:
    - 🎯 **Prediksi**: Apakah kita bisa memprediksi keberhasilan/kegagalan siswa berdasarkan fitur-fitur yang tersedia?
    - 🔍 **Segmentasi**: Apakah ada kelompok (cluster) siswa dengan karakteristik serupa?
    - 📊 **Faktor Penting**: Fitur apa yang paling berpengaruh terhadap hasil belajar siswa?

    ### 3. Pipeline yang Digunakan (CRISP-DM)
    """)

    st.markdown("""
    ```
    ┌──────────────────┐     ┌──────────────┐     ┌──────────────────┐
    │  1. Business     │────▶│  2. Data      │────▶│  3. Data         │
    │  Understanding   │     │  Understanding│     │  Preparation     │
    └──────────────────┘     │  (EDA)        │     │  (Preprocessing) │
                             └──────────────┘     └──────────────────┘
                                                           │
    ┌──────────────────┐     ┌──────────────┐              ▼
    │  6. Deployment   │◀────│  5. Evaluation│◀────┌──────────────────┐
    │                  │     │              │     │  4. Modeling      │
    └──────────────────┘     └──────────────┘     │  (Classification │
                                                  │   & Clustering)  │
                                                  └──────────────────┘
    ```
    """)

    st.markdown("""
    ### 4. Dataset yang Tersedia

    | No | Dataset | Sumber | Target | Jumlah Fitur | Tipe |
    |----|---------|--------|--------|-------------|------|
    | 1 | Students Performance | Kaggle | math/reading/writing score | 8 | Real-world |
    | 2 | Student Alcohol Consumption | UCI ML Repository | G3 (final grade) | 30+ | Real-world (simulated) |
    | 3 | Student Stress Factors | Kaggle | stress_level | 20+ | Real-world (simulated) |

    ### 5. Referensi Akademik
    - Cortez, P. & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. EUROSIS.
    - Amrieh, E.A., Hamtini, T. & Aljarah, I. (2016). *Mining Educational Data to Predict Student's Academic Performance Using Ensemble Methods*. International Journal of Database Theory and Application, 9(8), 119-136.
    - Fernandes, E. et al. (2019). *Educational data mining: Predictive analysis of academic performance of public school students*. Journal of Information Systems Engineering & Management.
    - Romero, C. & Ventura, S. (2020). *Educational Data Mining and Learning Analytics: An Updated Survey*. WIREs Data Mining and Knowledge Discovery.
    
    📌 **Sumber data**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/), [Kaggle](https://www.kaggle.com/), Google Scholar, Mendeley, Nature.
    """)

    st.markdown("""
    ### 6. Kandidat Fitur & Target

    Dalam klasifikasi biner (binary classification), target di-encode menjadi **2 kelas** untuk menghindari:
    - Multi-class yang menghasilkan score terlalu tinggi (kecurigaan **data leakage**)
    - Kesulitan interpretasi pada confusion matrix
    
    **Contoh encoding target:**
    - `G3 >= 10` → Lulus (1) / Gagal (0)
    - `stress_level` → High (1) / Low (0)
    - `math score >= 60` → Lulus (1) / Gagal (0)
    """)

# ---------------------------------------------------------
# TAB 1: EDA & OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.subheader("🔍 Exploratory Data Analysis (EDA)")

    # --- 1. Data Mentah ---
    st.markdown("#### 1. Data Mentah (Raw Data)")
    st.dataframe(df.head(20), use_container_width=True)

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    col_info1.metric("Total Baris", f"{df.shape[0]:,}")
    col_info2.metric("Total Kolom", f"{df.shape[1]}")
    col_info3.metric("Kolom Numerik", f"{len(df.select_dtypes(include=np.number).columns)}")
    col_info4.metric("Kolom Kategorikal", f"{len(df.select_dtypes(include=['object', 'category']).columns)}")

    st.divider()

    # --- 2. Missing Value ---
    st.markdown("#### 2. Penanganan Missing Value")
    missing_data = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Kolom': missing_data.index, 'Jumlah Missing': missing_data.values, 'Persentase (%)': missing_pct.values})
    missing_df = missing_df[missing_df['Jumlah Missing'] > 0]

    if len(missing_df) > 0:
        st.warning(f"⚠️ Ditemukan {len(missing_df)} kolom dengan missing value!")
        st.dataframe(missing_df, use_container_width=True)
        handle_missing = st.radio("Bagaimana menangani missing value?",
                                  ["Drop baris dengan missing", "Isi dengan mean/modus", "Biarkan saja"])
        if handle_missing == "Drop baris dengan missing":
            df = df.dropna()
            st.success(f"✅ Missing value di-drop. Sisa data: {len(df)} baris.")
        elif handle_missing == "Isi dengan mean/modus":
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            st.success("✅ Missing value diisi dengan mean (numerik) / modus (kategorikal).")
    else:
        st.markdown('<div class="alert-ok">✅ Tidak ada missing value pada dataset ini.</div>', unsafe_allow_html=True)

    st.divider()

    # --- 3. Deskriptif Statistik ---
    st.markdown("#### 3. Deskriptif Statistik (Univariate)")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if num_cols:
        desc = df[num_cols].describe().T
        desc['skewness'] = df[num_cols].skew()
        desc['kurtosis'] = df[num_cols].kurtosis()
        st.dataframe(desc.round(3).style.background_gradient(cmap='Blues', subset=['mean', 'std']),
                     use_container_width=True)

    if cat_cols:
        st.markdown("**Statistik Kolom Kategorikal:**")
        cat_stats = []
        for col in cat_cols:
            cat_stats.append({
                'Kolom': col,
                'Unique Values': df[col].nunique(),
                'Modus': df[col].mode()[0],
                'Frekuensi Modus': df[col].value_counts().iloc[0],
                'Top 3 Values': ', '.join(df[col].value_counts().head(3).index.tolist())
            })
        st.dataframe(pd.DataFrame(cat_stats), use_container_width=True)

    st.divider()

    # --- 4. Distribusi Target & Fitur (Univariate) ---
    st.markdown("#### 4. Distribusi Fitur (Univariate)")

    all_cols = num_cols + cat_cols
    dist_col = st.selectbox("Pilih kolom untuk dilihat distribusinya:", all_cols,
                            index=0 if all_cols else None, key="dist_col_eda")

    if dist_col:
        if dist_col in num_cols:
            fig_dist = px.histogram(df, x=dist_col, marginal="box",
                                    color_discrete_sequence=['#4C72B0'],
                                    title=f"Distribusi {dist_col}")
            fig_dist.update_layout(template="plotly_white")
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            fig_bar = px.bar(df[dist_col].value_counts().reset_index(),
                            x='index', y=dist_col,
                            color='index',
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            title=f"Distribusi {dist_col}",
                            labels={'index': dist_col, dist_col: 'Jumlah'})
            fig_bar.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # --- 5. Bivariate Analysis ---
    st.markdown("#### 5. Hubungan 2 Atribut (Bivariate)")
    if len(num_cols) >= 2:
        col_bv1, col_bv2 = st.columns(2)
        with col_bv1:
            bv_x = st.selectbox("Sumbu X:", num_cols, index=0, key="bv_x")
        with col_bv2:
            bv_y = st.selectbox("Sumbu Y:", num_cols, index=min(1, len(num_cols)-1), key="bv_y")

        color_by = None
        if cat_cols:
            color_by = st.selectbox("Warnai berdasarkan (opsional):", ["Tanpa warna"] + cat_cols, key="bv_color")
            if color_by == "Tanpa warna":
                color_by = None

        fig_scatter = px.scatter(df, x=bv_x, y=bv_y, color=color_by,
                                 opacity=0.7, title=f"{bv_x} vs {bv_y}",
                                 color_discrete_sequence=px.colors.qualitative.Set2)
        fig_scatter.update_layout(template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)

    elif cat_cols and num_cols:
        st.write("Grouped Bar Chart:")
        cat_sel = st.selectbox("Kategori:", cat_cols, key="bv_cat")
        num_sel = st.selectbox("Nilai:", num_cols, key="bv_num")
        fig_group = px.box(df, x=cat_sel, y=num_sel, color=cat_sel,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_group.update_layout(template="plotly_white")
        st.plotly_chart(fig_group, use_container_width=True)

    st.divider()

    # --- 6. Multivariate — Heatmap Korelasi ---
    st.markdown("#### 6. Korelasi Antar Fitur (Multivariate — Heatmap)")
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr()
        fig_heatmap = px.imshow(corr_matrix.round(2),
                                text_auto=True,
                                color_continuous_scale='RdBu_r',
                                title="Heatmap Korelasi Antar Fitur Numerik",
                                aspect="auto")
        fig_heatmap.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Heatmap korelasi membutuhkan minimal 2 kolom numerik.")


# ---------------------------------------------------------
# TAB 2: PREPROCESSING & ENCODING
# ---------------------------------------------------------
with tab2:
    st.subheader("⚙️ Preprocessing Data")

    df_prep = df.copy()
    num_cols_prep = df_prep.select_dtypes(include=np.number).columns.tolist()
    cat_cols_prep = df_prep.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- 1. Pilih Kolom Target ---
    st.markdown("#### 1. Penentuan Kolom Target")
    all_cols_prep = num_cols_prep + cat_cols_prep
    target_col = st.selectbox("Pilih kolom yang akan dijadikan TARGET klasifikasi:",
                              all_cols_prep, index=0 if all_cols_prep else None, key="target_prep")

    st.divider()

    # --- 2. Binary Encoding Target ---
    st.markdown("#### 2. Encoding Target → Binary (2 Kelas)")
    st.markdown('<div class="info-box">💡 <strong>Binary Classification</strong> dipilih agar hasil evaluasi lebih interpretable dan menghindari kecurigaan data leakage pada multi-class.</div>', unsafe_allow_html=True)

    if target_col in num_cols_prep:
        st.write(f"Kolom `{target_col}` bersifat **numerik**. Akan di-encode menjadi binary berdasarkan ambang batas.")
        threshold_val = st.number_input(f"Ambang batas untuk {target_col}:", value=int(pass_threshold), key="thresh_prep")
        df_prep['Target_Binary'] = (df_prep[target_col] >= threshold_val).astype(int)
        label_map = {1: f"≥ {threshold_val} (Positif)", 0: f"< {threshold_val} (Negatif)"}
    else:
        st.write(f"Kolom `{target_col}` bersifat **kategorikal**.")
        unique_vals = df_prep[target_col].unique().tolist()
        if len(unique_vals) == 2:
            positive_class = st.selectbox("Pilih kelas POSITIF (1):", unique_vals, key="pos_class")
            df_prep['Target_Binary'] = (df_prep[target_col] == positive_class).astype(int)
            label_map = {1: positive_class, 0: [v for v in unique_vals if v != positive_class][0]}
        else:
            st.write(f"Ada {len(unique_vals)} kelas unik. Pilih satu kelas sebagai kelas POSITIF (sisanya = NEGATIF).")
            positive_class = st.selectbox("Pilih kelas POSITIF (1):", unique_vals, key="pos_class_multi")
            df_prep['Target_Binary'] = (df_prep[target_col] == positive_class).astype(int)
            label_map = {1: positive_class, 0: "Lainnya"}

    # Tampilkan distribusi target binary
    target_dist = df_prep['Target_Binary'].value_counts()
    col_td1, col_td2 = st.columns(2)
    with col_td1:
        fig_target = px.pie(values=target_dist.values, names=[str(label_map.get(k, k)) for k in target_dist.index],
                           title="Distribusi Target (Binary)",
                           color_discrete_sequence=['#4C72B0', '#DD8452'])
        fig_target.update_layout(template="plotly_white")
        st.plotly_chart(fig_target, use_container_width=True)
    with col_td2:
        st.metric("Kelas 0 (Negatif)", f"{target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(df_prep)*100:.1f}%)")
        st.metric("Kelas 1 (Positif)", f"{target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(df_prep)*100:.1f}%)")
        ratio = target_dist.get(1, 0) / max(target_dist.get(0, 1), 1)
        if ratio < 0.3 or ratio > 3:
            st.warning("⚠️ Dataset imbalanced! Pertimbangkan resampling atau class_weight.")

    st.divider()

    # --- 3. Data Leakage Prevention ---
    st.markdown("#### 3. Pencegahan Data Leakage")
    st.markdown('<div class="alert-leakage">⚠️ <strong>Data Leakage</strong> terjadi ketika informasi dari target "bocor" ke fitur training. Kolom yang langsung menentukan target HARUS di-drop!</div>', unsafe_allow_html=True)

    cols_available = [c for c in df_prep.columns if c != 'Target_Binary']
    default_drop = [target_col] if target_col in cols_available else []

    # Deteksi otomatis kolom yang sangat berkorelasi dengan target
    if num_cols_prep:
        for nc in num_cols_prep:
            if nc in df_prep.columns and nc != 'Target_Binary':
                try:
                    corr_val = abs(df_prep[nc].corr(df_prep['Target_Binary']))
                    if corr_val > 0.85 and nc not in default_drop:
                        default_drop.append(nc)
                except:
                    pass

    cols_to_drop = st.multiselect(
        "Pilih kolom yang harus di-DROP untuk mencegah Data Leakage:",
        cols_available, default=default_drop, key="drop_cols_prep"
    )
    df_prep = df_prep.drop(columns=cols_to_drop, errors='ignore')

    st.divider()

    # --- 4. Label Encoding Kategorikal ---
    st.markdown("#### 4. Encoding Kolom Kategorikal")
    cat_cols_remaining = df_prep.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols_remaining:
        st.write(f"Kolom kategorikal yang akan di-encode: `{', '.join(cat_cols_remaining)}`")
        le = LabelEncoder()
        encoding_map = {}
        for col in cat_cols_remaining:
            le.fit(df_prep[col].astype(str))
            encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            df_prep[col] = le.transform(df_prep[col].astype(str))
        with st.expander("📋 Lihat Mapping Encoding"):
            for col, mapping in encoding_map.items():
                st.write(f"**{col}**: {mapping}")
    else:
        st.success("✅ Tidak ada kolom kategorikal yang perlu di-encode.")

    st.divider()

    # --- 5. Merge Dataset (Opsional) ---
    st.markdown("#### 5. Merge Dataset Tambahan (Opsional)")
    do_merge = st.checkbox("Ingin menambahkan / merge dataset lain?", value=False)
    if do_merge:
        merge_file = st.file_uploader("Upload dataset tambahan:", type=['csv', 'xls', 'xlsx'], key="merge_upload")
        if merge_file:
            if merge_file.name.endswith('.csv'):
                df_extra = pd.read_csv(merge_file)
            else:
                df_extra = pd.read_excel(merge_file)
            st.write(f"Dataset tambahan: {df_extra.shape[0]} baris × {df_extra.shape[1]} kolom")
            merge_type = st.selectbox("Tipe Merge:", ["Concat (Tambah baris)", "Join (Tambah kolom)"])
            if merge_type == "Concat (Tambah baris)":
                df_prep = pd.concat([df_prep, df_extra], ignore_index=True)
                st.success(f"✅ Dataset di-concat. Total baris sekarang: {len(df_prep)}")
            else:
                merge_key = st.selectbox("Pilih kolom kunci untuk join:", df_prep.columns.tolist())
                if merge_key in df_extra.columns:
                    df_prep = df_prep.merge(df_extra, on=merge_key, how='left')
                    st.success(f"✅ Dataset di-join. Total kolom sekarang: {df_prep.shape[1]}")
                else:
                    st.error(f"Kolom `{merge_key}` tidak ditemukan di dataset tambahan!")

    st.divider()

    # --- 6. Cuplikan Data Setelah Preprocessing ---
    st.markdown("#### 6. Cuplikan Data Setelah Preprocessing")
    st.dataframe(df_prep.head(20), use_container_width=True)
    st.write(f"Shape: {df_prep.shape[0]} baris × {df_prep.shape[1]} kolom")

    # Simpan ke session state untuk tab berikutnya
    st.session_state['df_prep'] = df_prep


# ---------------------------------------------------------
# TAB 3: CLUSTERING (K-MEANS)
# ---------------------------------------------------------
with tab3:
    st.subheader("🧩 Clustering (K-Means)")

    if 'df_prep' not in st.session_state:
        st.warning("⚠️ Silakan selesaikan Preprocessing di tab sebelumnya terlebih dahulu.")
        st.stop()

    df_cluster = st.session_state['df_prep'].copy()

    st.markdown("#### 📝 Apa itu Clustering?")
    st.markdown("""
    **Clustering** adalah teknik *unsupervised learning* yang mengelompokkan data ke dalam kelompok-kelompok (cluster) 
    berdasarkan kemiripan fitur — **tanpa menggunakan label/target**. K-Means adalah salah satu algoritma clustering 
    yang paling populer, bekerja dengan mencari `K` centroid yang meminimalkan jarak intra-cluster (WCSS).
    
    Kegunaan dalam konteks pendidikan:
    - Mengelompokkan siswa berdasarkan profil akademik
    - Menemukan pola tersembunyi dalam data
    - Membantu personalisasi intervensi pendidikan
    """)

    cluster_features = df_cluster.drop(columns=['Target_Binary'], errors='ignore')
    # Pastikan semua numerik
    cluster_features = cluster_features.select_dtypes(include=np.number)

    if cluster_features.shape[1] < 2:
        st.error("Minimal 2 fitur numerik dibutuhkan untuk clustering.")
    else:
        # Scale data untuk clustering
        scaler_cluster = StandardScaler()
        cluster_scaled = scaler_cluster.fit_transform(cluster_features)

        # --- 1. Elbow Method ---
        st.markdown("#### 1. Penentuan Jumlah Cluster — Elbow Method")
        st.markdown("Mencari nilai `K` optimal dimana penambahan cluster tidak lagi memberikan penurunan WCSS yang signifikan.")

        max_k = min(10, len(cluster_features) - 1)
        wcss = []
        K_range = range(1, max_k + 1)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(cluster_scaled)
            wcss.append(kmeans.inertia_)

        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(K_range), y=wcss, mode='lines+markers',
                                        marker=dict(size=10, color='#4C72B0'),
                                        line=dict(width=2, color='#4C72B0')))
        fig_elbow.update_layout(title='Elbow Method — Menentukan K Optimal',
                                xaxis_title='Jumlah Cluster (K)',
                                yaxis_title='WCSS (Within-Cluster Sum of Squares)',
                                template='plotly_white')
        st.plotly_chart(fig_elbow, use_container_width=True)

        k_selected = st.slider("Pilih K optimal berdasarkan grafik di atas:", 2, max_k, 3, key="k_cluster")

        st.divider()

        # --- 2. Hasil Clustering ---
        st.markdown("#### 2. Hasil Clustering")
        final_kmeans = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(cluster_scaled)
        df_cluster['Cluster'] = cluster_labels

        # Rata-rata tiap cluster
        st.markdown("##### 📊 Rata-rata Fitur Tiap Cluster")
        cluster_avg = df_cluster.groupby('Cluster').mean(numeric_only=True).round(3)
        st.dataframe(cluster_avg.style.background_gradient(cmap='YlGnBu'), use_container_width=True)

        # Jumlah anggota cluster
        cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
        st.markdown("##### 👥 Jumlah Anggota Tiap Cluster")
        col_cl = st.columns(k_selected)
        for i, (cluster_id, count) in enumerate(cluster_counts.items()):
            col_cl[i % len(col_cl)].metric(f"Cluster {cluster_id}", f"{count} siswa")

        st.divider()

        # --- 3. Visualisasi Cluster (PCA 2D) ---
        st.markdown("#### 3. Visualisasi Cluster (PCA 2D)")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(cluster_scaled)
        df_pca = pd.DataFrame({'PC1': pca_result[:, 0], 'PC2': pca_result[:, 1],
                               'Cluster': cluster_labels.astype(str)})

        fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                             title=f'Visualisasi Cluster (K={k_selected}) — PCA 2 Komponen',
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             opacity=0.7)
        fig_pca.update_layout(template='plotly_white', height=500)
        st.plotly_chart(fig_pca, use_container_width=True)

        st.write(f"📈 **Explained Variance**: PC1 = {pca.explained_variance_ratio_[0]:.2%}, PC2 = {pca.explained_variance_ratio_[1]:.2%}")
        st.write(f"📈 **Total Variance Explained**: {sum(pca.explained_variance_ratio_):.2%}")


# ---------------------------------------------------------
# TAB 4: CLASSIFICATION & EVALUASI
# ---------------------------------------------------------
with tab4:
    st.subheader("🎯 Classification & Evaluasi Model")

    if 'df_prep' not in st.session_state:
        st.warning("⚠️ Silakan selesaikan Preprocessing di tab sebelumnya terlebih dahulu.")
        st.stop()

    df_model = st.session_state['df_prep'].copy()

    if 'Target_Binary' not in df_model.columns:
        st.error("❌ Kolom `Target_Binary` tidak ditemukan. Pastikan encoding target di Tab Preprocessing sudah benar.")
        st.stop()

    # --- 1. Setup ---
    st.markdown("#### 1. Setup Pemodelan")

    X = df_model.drop(columns=['Target_Binary', 'Cluster'], errors='ignore')
    X = X.select_dtypes(include=np.number)  # pastikan semua numerik
    y = df_model['Target_Binary']

    st.write(f"**Fitur (X)**: {X.shape[1]} kolom — `{', '.join(X.columns.tolist())}`")
    st.write(f"**Target (y)**: Target_Binary — {y.value_counts().to_dict()}")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=42, stratify=y)

    st.write(f"✅ Data di-split: **Train** = {len(X_train)} | **Test** = {len(X_test)} | Rasio = {1-test_size:.0%} / {test_size:.0%}")

    # Feature Scaling (fit HANYA di train → mencegah leakage)
    if scaler_choice == "StandardScaler":
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        st.write(f"✅ Feature Scaler: **{scaler_choice}** (fit pada train, transform pada test)")
    elif scaler_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        st.write(f"✅ Feature Scaler: **{scaler_choice}** (fit pada train, transform pada test)")
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        st.write("✅ Feature Scaler: **None** (tanpa scaling)")

    st.divider()

    # --- 2. Model Training & Evaluation ---
    st.markdown("#### 2. Evaluasi Model (Classification Metrics)")

    # Inisialisasi model berdasarkan pilihan user
    model_dict = {}
    param_grids = {}

    if "Logistic Regression" in algo_choices:
        model_dict["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=42)
        param_grids["Logistic Regression"] = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs']}

    if "Random Forest" in algo_choices:
        model_dict["Random Forest"] = RandomForestClassifier(random_state=42)
        param_grids["Random Forest"] = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None]}

    if "SVM" in algo_choices:
        model_dict["SVM"] = SVC(probability=True, random_state=42)
        param_grids["SVM"] = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}

    if "Gradient Boosting" in algo_choices:
        model_dict["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)
        param_grids["Gradient Boosting"] = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5]}

    if not model_dict:
        st.warning("⚠️ Pilih minimal 1 algoritma di sidebar!")
        st.stop()

    results_summary = []

    for name, model in model_dict.items():
        st.markdown(f"---")
        st.markdown(f"### 🤖 {name}")

        # Hyperparameter Tuning
        if do_hyperparameter and name in param_grids:
            st.write(f"🔍 Menjalankan GridSearchCV untuk {name}...")
            grid = GridSearchCV(model, param_grids[name], cv=cv_folds, scoring='f1',
                               n_jobs=-1, refit=True)
            grid.fit(X_train_scaled, y_train)
            best_model = grid.best_estimator_
            st.write(f"✅ **Best Parameters**: `{grid.best_params_}`")
            st.write(f"✅ **Best CV F1-Score**: `{grid.best_score_:.4f}`")
        else:
            best_model = model

        # Cross Validation
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv_folds, scoring='accuracy')
        cv_f1 = cross_val_score(best_model, X_train_scaled, y_train, cv=cv_folds, scoring='f1')

        col_cv1, col_cv2 = st.columns(2)
        col_cv1.metric(f"CV Accuracy ({cv_folds}-Fold)", f"{cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        col_cv2.metric(f"CV F1-Score ({cv_folds}-Fold)", f"{cv_f1.mean():.4f} (±{cv_f1.std()*2:.4f})")

        # Fitting & Prediction
        best_model.fit(X_train_scaled, y_train)
        y_pred = best_model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        # Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Test Accuracy", f"{acc:.4f}")
        col_m2.metric("Test F1-Score (Macro)", f"{f1:.4f}")

        roc_auc = None
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
            col_m3.metric("ROC-AUC", f"{roc_auc:.4f}")

        # ⚠️ Data Leakage Alert
        if acc > 0.85:
            st.markdown(f'<div class="alert-leakage">⚠️ <strong>PERINGATAN DATA LEAKAGE!</strong><br>'
                        f'Accuracy = {acc:.2%} — Score terlalu tinggi! Periksa apakah ada fitur yang bocor ke target. '
                        f'Multi-class atau fitur yang langsung menentukan target bisa menyebabkan hal ini. '
                        f'Pertimbangkan untuk drop kolom yang berkorelasi tinggi dengan target di Tab Preprocessing.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-ok">✅ Score wajar ({acc:.2%}). Tidak ada indikasi data leakage.</div>',
                        unsafe_allow_html=True)

        # Classification Report
        col_report, col_cm = st.columns(2)

        with col_report:
            st.markdown("##### 📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(4)
            st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']),
                        use_container_width=True)

        # Confusion Matrix
        with col_cm:
            st.markdown("##### 🔲 Confusion Matrix")
            cm_labels_all = sorted(set(y_test) | set(y_pred))
            cm = confusion_matrix(y_test, y_pred, labels=cm_labels_all)
            cm_display = [f"Negatif (0)" if l == 0 else f"Positif (1)" for l in cm_labels_all]
            fig_cm = px.imshow(cm, text_auto=True,
                              labels=dict(x="Prediksi", y="Aktual"),
                              x=cm_display,
                              y=cm_display,
                              color_continuous_scale='Blues',
                              title=f"Confusion Matrix — {name}")
            fig_cm.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

        # ROC Curve
        if roc_auc is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                          name=f'{name} (AUC={roc_auc:.4f})',
                                          line=dict(color='#4C72B0', width=2)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                          name='Random Baseline', line=dict(dash='dash', color='gray')))
            fig_roc.update_layout(title=f'ROC Curve — {name}',
                                  xaxis_title='False Positive Rate',
                                  yaxis_title='True Positive Rate',
                                  template='plotly_white', height=400)
            st.plotly_chart(fig_roc, use_container_width=True)

        # Simpan hasil
        row = {'Model': name, 'Accuracy': acc, 'F1-Score': f1,
               'CV Accuracy': cv_scores.mean(), 'CV F1': cv_f1.mean()}
        if roc_auc is not None:
            row['ROC-AUC'] = roc_auc
        results_summary.append(row)

    st.divider()

    # --- 3. Leaderboard ---
    st.markdown("#### 3. 🏆 Model Leaderboard")
    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    st.dataframe(results_df.style.background_gradient(cmap='Greens', subset=['Accuracy', 'F1-Score']),
                 use_container_width=True)

    best_model_name = results_df.iloc[0]['Model']
    st.success(f"🏆 Model terbaik berdasarkan F1-Score: **{best_model_name}** ({results_df.iloc[0]['F1-Score']:.4f})")

    st.divider()

    # --- 4. Feature Importance ---
    st.markdown("#### 4. 📊 Fitur Paling Berpengaruh (Feature Importance)")

    # Gunakan model Random Forest untuk feature importance (paling informatif)
    if "Random Forest" in model_dict:
        rf_model = model_dict["Random Forest"]
        if do_hyperparameter and "Random Forest" in param_grids:
            rf_grid = GridSearchCV(rf_model, param_grids["Random Forest"], cv=cv_folds, scoring='f1', n_jobs=-1)
            rf_grid.fit(X_train_scaled, y_train)
            rf_fitted = rf_grid.best_estimator_
        else:
            rf_fitted = rf_model.fit(X_train_scaled, y_train)

        importances = rf_fitted.feature_importances_
        feat_imp_df = pd.DataFrame({'Fitur': X.columns, 'Importance': importances})
        feat_imp_df = feat_imp_df.sort_values('Importance', ascending=True)

        fig_imp = px.bar(feat_imp_df, x='Importance', y='Fitur', orientation='h',
                         color='Importance', color_continuous_scale='Blues',
                         title='Feature Importance (Random Forest)')
        fig_imp.update_layout(template='plotly_white', height=max(400, len(X.columns) * 25))
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("##### 🏅 Top 5 Fitur Paling Berpengaruh:")
        top5 = feat_imp_df.sort_values('Importance', ascending=False).head(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            st.write(f"**{i+1}. {row['Fitur']}** — Importance: `{row['Importance']:.4f}`")

    else:
        # Fallback: koefisien Logistic Regression
        if "Logistic Regression" in model_dict:
            lr_model = model_dict["Logistic Regression"].fit(X_train_scaled, y_train)
            coefs = np.abs(lr_model.coef_[0])
            feat_imp_df = pd.DataFrame({'Fitur': X.columns, 'Coefficient (abs)': coefs})
            feat_imp_df = feat_imp_df.sort_values('Coefficient (abs)', ascending=True)

            fig_imp = px.bar(feat_imp_df, x='Coefficient (abs)', y='Fitur', orientation='h',
                             color='Coefficient (abs)', color_continuous_scale='Blues',
                             title='Feature Importance (Logistic Regression Coefficients)')
            fig_imp.update_layout(template='plotly_white', height=max(400, len(X.columns) * 25))
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Tambahkan Random Forest atau Logistic Regression untuk melihat Feature Importance.")

    st.divider()

    # --- 5. Ringkasan Akhir ---
    st.markdown("#### 5. 📝 Ringkasan Pipeline")
    st.markdown(f"""
    | Aspek | Detail |
    |-------|--------|
    | **Dataset** | {dataset_name} ({df.shape[0]} baris × {df.shape[1]} kolom) |
    | **Target** | {target_col} → Binary (2 kelas) |
    | **Preprocessing** | Label Encoding, {scaler_choice}, Data Leakage Prevention |
    | **Clustering** | K-Means (Elbow Method) |
    | **Klasifikasi** | {', '.join(algo_choices)} |
    | **Evaluasi** | Accuracy, F1-Score, ROC-AUC, Confusion Matrix, Classification Report |
    | **Cross-Validation** | {cv_folds}-Fold |
    | **Test Size** | {test_size:.0%} |
    | **Hyperparameter Tuning** | {'GridSearchCV' if do_hyperparameter else 'Default'} |
    | **Model Terbaik** | {best_model_name} |
    """)
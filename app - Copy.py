import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Import Scikit-Learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans

# ==========================================
# 1. KONFIGURASI HALAMAN (LIGHT THEME DEFAULT)
# ==========================================
st.set_page_config(page_title="Edu Data Mining Pipeline", page_icon="📊", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #ffffff; color: #000000; }
    h1, h2, h3 { color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR: PENGATURAN PIPELINE
# ==========================================
with st.sidebar:
    st.title("⚙️ Konfigurasi Pipeline")
    uploaded_file = st.file_uploader("1. Upload Dataset", type=['csv', 'xls', 'xlsx'])
    
    st.divider()
    st.subheader("Pengaturan Modeling")
    test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
    cv_folds = st.slider("Cross-Validation (K-Folds)", 2, 10, 5)
    scaler_choice = st.selectbox("Feature Scaler", ["StandardScaler", "MinMaxScaler", "None"])
    pass_threshold = st.number_input("Ambang Batas Lulus (Untuk Klasifikasi Biner)", value=60)

# ==========================================
# 3. KONTEN UTAMA
# ==========================================
st.title("📊 Educational Data Mining Pipeline")
st.markdown("Implementasi End-to-End: EDA ➔ Preprocessing ➔ Clustering ➔ Classification")

if uploaded_file is None:
    st.info("Silakan unggah dataset di panel kiri untuk memulai pipeline.")
    st.stop()

# Membaca data
if uploaded_file.name.endswith('.csv'):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# Buat Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 EDA & Overview", "⚙️ Preprocessing", "🧩 Clustering", "🎯 Classification (Modeling)"])

# ---------------------------------------------------------
# TAB 1: EDA & OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.subheader("1. Data Mentah (Raw Data)")
    st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("2. Penanganan Missing Value")
    missing_data = df.isnull().sum()
    st.write(missing_data[missing_data > 0] if missing_data.any() else "✅ Tidak ada missing value pada dataset ini.")
    
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    st.subheader("3. Distribusi Target & Fitur (Univariate)")
    target_col = st.selectbox("Pilih kolom untuk dilihat distribusinya:", num_cols, index=0 if num_cols else None)
    if target_col:
        fig_dist = px.histogram(df, x=target_col, marginal="box", color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig_dist, use_container_width=True)
        
    st.subheader("4. Korelasi Antar Fitur (Heatmap)")
    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------------
# TAB 2: PREPROCESSING & ENCODING
# ---------------------------------------------------------
with tab2:
    st.subheader("1. Encoding Kategorikal & Target Biner")
    st.markdown("Mengubah data teks menjadi angka agar bisa dibaca oleh model *Machine Learning*.")
    
    # Copy dataframe untuk preprocessing
    df_prep = df.copy()
    
    # Mengubah Target menjadi Biner (Mencegah Multi-class yang membingungkan)
    if target_col:
        st.info(f"Target klasifikasi dibuat biner (Lulus/Gagal) berdasarkan {target_col} >= {pass_threshold}")
        df_prep['Target_Lulus'] = (df_prep[target_col] >= pass_threshold).astype(int)
        
        # Mencegah Data Leakage: Menghapus kolom nilai asli agar model tidak "mencontek"
        cols_to_drop = st.multiselect("Pilih kolom yang harus didrop untuk mencegah Data Leakage:", 
                                      num_cols, default=num_cols)
        df_prep = df_prep.drop(columns=cols_to_drop)

    # Label Encoding untuk kolom kategorikal
    cat_cols = df_prep.select_dtypes(include=['object', 'category']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df_prep[col] = le.fit_transform(df_prep[col].astype(str))
        
    st.subheader("2. Cuplikan Data Setelah Preprocessing")
    st.dataframe(df_prep.head(), use_container_width=True)

# ---------------------------------------------------------
# TAB 3: CLUSTERING (K-MEANS)
# ---------------------------------------------------------
with tab3:
    st.subheader("1. Penentuan Jumlah Cluster (Elbow Method)")
    st.markdown("Mencari nilai K (kelompok) yang paling optimal.")
    
    cluster_data = df_prep.drop(columns=['Target_Lulus'], errors='ignore')
    
    # Menghitung WCSS untuk Elbow Method
    wcss = []
    K_range = range(1, 8)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(cluster_data)
        wcss.append(kmeans.inertia_)
        
    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
    ax_elbow.plot(K_range, wcss, marker='o', linestyle='--', color='b')
    ax_elbow.set_title('Elbow Method')
    ax_elbow.set_xlabel('Jumlah Cluster (K)')
    ax_elbow.set_ylabel('WCSS')
    st.pyplot(fig_elbow)
    
    k_selected = st.slider("Pilih K optimal berdasarkan grafik di atas:", 2, 7, 3)
    
    st.subheader("2. Rata-rata Tiap Cluster")
    final_kmeans = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
    df_prep['Cluster'] = final_kmeans.fit_predict(cluster_data)
    
    cluster_avg = df_prep.groupby('Cluster').mean().round(2)
    st.dataframe(cluster_avg.style.background_gradient(cmap='YlGnBu'), use_container_width=True)

# ---------------------------------------------------------
# TAB 4: MODELING (CLASSIFICATION)
# ---------------------------------------------------------
with tab4:
    st.subheader("1. Setup Pemodelan & Cross Validation")
    if 'Target_Lulus' not in df_prep.columns:
        st.warning("Silakan tentukan kolom target di Tab Preprocessing terlebih dahulu.")
    else:
        # Menyiapkan X dan y
        X = df_prep.drop(columns=['Target_Lulus', 'Cluster'], errors='ignore')
        y = df_prep['Target_Lulus']
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Fitur Scaler (Mencegah Leakage: Fit hanya di Train, Transform di Test)
        if scaler_choice == "StandardScaler":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif scaler_choice == "MinMaxScaler":
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
        st.write(f"✅ Data di-split dengan rasio {1.0 - test_size:.2f} (Train) : {test_size:.2f} (Test)")
        st.write(f"✅ Scaler yang digunakan: {scaler_choice}")

        st.subheader("2. Evaluasi Model (Classification Metrics)")
        
        # Inisialisasi Model
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(random_state=42)
        }
        
        for name, model in models.items():
            st.markdown(f"#### 🤖 {name}")
            
            # Cross Validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            st.write(f"**Cross-Validation Accuracy ({cv_folds} Folds):** {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Fitting & Prediction
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Classification Report
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            col_m2.metric("F1-Score (Macro)", f"{f1_score(y_test, y_pred, average='macro'):.4f}")
            
            # ROC AUC (Jika model mendukung predict_proba)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                col_m3.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
            
            with st.expander(f"Lihat Detail Classification Report - {name}"):
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
            st.divider()
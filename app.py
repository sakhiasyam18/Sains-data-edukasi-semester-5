import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURASI HALAMAN & UI/UX (UI/UX & Framework)
# ==========================================
st.set_page_config(
    page_title="EduAnalytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik UI (Bagus dan Modern)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1e3d59; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .st-tabs { font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR: PENGATURAN & UPLOAD FILE (Dinamis)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3074/3074058.png", width=100)
    st.title("⚙️ Konfigurasi")
    st.markdown("Unggah dataset untuk memulai analisis.")
    
    # Mendukung upload CSV maupun Excel
    uploaded_file = st.file_uploader("Upload File Data (.csv / .xls / .xlsx)", type=['csv', 'xls', 'xlsx'])
    
    st.divider()
    st.subheader("Pengaturan Lanjutan")
    # Interactive Slider
    pass_threshold = st.slider("Ambang Batas Lulus (Nilai)", min_value=0, max_value=100, value=60, step=1)
    
    st.markdown("---")
    st.caption("Dibuat menggunakan Streamlit & Plotly")

# ==========================================
# 3. KONTEN UTAMA & LOGIKA PROGRAM
# ==========================================
st.title("🎓 Student Performance Data Mining Testbed")
st.markdown("Platform analisis data siswa interaktif untuk *Exploratory Data Analysis* (EDA) dan Pemodelan.")

# Logika penanganan file (Bisa berubah-ubah filenya)
if uploaded_file is None:
    # Tampilan kosong/landing page jika file belum diunggah
    st.info("👋 Halo! Silakan unggah file dataset (`StudentsPerformance.csv` atau Excel) di panel sebelah kiri untuk melihat visualisasi.")
    
    # Placeholder gambar untuk UI/UX
    st.image("https://illustrations.popsy.co/amber/student-going-to-school.svg", width=400)
    st.stop() # Menghentikan eksekusi kode di bawahnya hingga file diunggah

else:
    # Membaca data secara dinamis berdasarkan ekstensi
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
        st.stop()

    # ==========================================
    # 4. MEMBUAT TABS (Seperti Dashboard Asli)
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📋 Overview & Insight", "🔍 Interactive EDA", "🤖 Simulasi Modeling"])

    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.markdown("### Ringkasan Dataset")
        st.markdown("Bagian ini memberikan gambaran umum mengenai struktur data yang kamu unggah beserta metrik utamanya.")
        
        # Metrik dinamis (Mudah dibaca)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Baris", f"{df.shape[0]:,}")
        col2.metric("Total Kolom", f"{df.shape[1]}")
        col3.metric("Data Kosong (Missing)", df.isnull().sum().sum())
        
        # Mencari kolom nilai secara dinamis (memprioritaskan math score jika ada)
        score_col = 'math score' if 'math score' in df.columns else df.select_dtypes(include='number').columns[0]
        
        if score_col:
            pass_rate = (df[score_col] >= pass_threshold).mean() * 100
            col4.metric(f"Tingkat Kelulusan ({score_col})", f"{pass_rate:.1f}%", 
                        delta="Batas: " + str(pass_threshold), delta_color="off")
        else:
            col4.metric("Tingkat Kelulusan", "N/A")

        st.divider()
        
        # Tabel Interaktif
        st.subheader("Cuplikan Data Mentah (Raw Data)")
        st.dataframe(df.head(50), use_container_width=True)
        
        # Penjelasan Data (Bisa dibaca dan informatif)
        with st.expander("💡 Klik di sini untuk melihat Penjelasan Kolom (Data Dictionary)"):
            st.markdown("""
            * **gender**: Jenis kelamin siswa.
            * **race/ethnicity**: Kelompok ras/etnis siswa.
            * **parental level of education**: Tingkat pendidikan terakhir orang tua.
            * **lunch**: Jenis makan siang (berhubungan dengan status ekonomi).
            * **test preparation course**: Apakah siswa mengikuti kursus persiapan.
            * **math/reading/writing score**: Nilai mata pelajaran (0-100).
            """)

    # --- TAB 2: INTERACTIVE EDA ---
    with tab2:
        st.markdown("### Exploratory Data Analysis (EDA)")
        st.markdown("Jelajahi pola data menggunakan visualisasi interaktif di bawah ini. **Kamu bisa mengarahkan kursor (hover), melakukan zoom, dan menyimpan grafik.**")
        
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if num_cols and cat_cols:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Interaktif Histogram (UI/UX)
                st.write(f"**Distribusi {score_col.title()}**")
                fig1 = px.histogram(df, x=score_col, nbins=20, marginal="box", 
                                    color_discrete_sequence=['#4C72B0'],
                                    hover_data=df.columns)
                st.plotly_chart(fig1, use_container_width=True)

            with col_chart2:
                # Interaktif Pie Chart
                cat_choice = st.selectbox("Pilih Kategori untuk dianalisis:", cat_cols)
                fig2 = px.pie(df, names=cat_choice, hole=0.4, 
                              color_discrete_sequence=px.colors.qualitative.Pastel)
                fig2.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig2, use_container_width=True)
            
            st.divider()
            st.write("**Hubungan Antar Nilai (Scatter Plot)**")
            if len(num_cols) >= 2:
                fig3 = px.scatter(df, x=num_cols[0], y=num_cols[1], color=cat_choice,
                                  hover_name=cat_choice, size_max=10, opacity=0.7,
                                  color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig3, use_container_width=True)

    # --- TAB 3: MODELING (Simulasi Leaderboard) ---
    with tab3:
        st.markdown("### 🚀 Quick Leaderboard (Simulasi)")
        st.markdown("Ini adalah simulasi hasil pemodelan *Machine Learning* seperti yang ada pada referensi gambarmu.")
        
        st.success("Training complete — see the metrics below.")
        
        # Membuat dummy dataframe untuk meniru gambar leaderboard
        leaderboard_data = pd.DataFrame({
            "Model": ["Logistic Regression", "Naive Bayes", "Random Forest", "Gradient Boosting"],
            "Accuracy": [0.658228, 0.658228, 0.670886, 0.620253],
            "Recall (Fail)": [0.307692, 0.346154, 0.269231, 0.346154],
            "Precision (Fail)": [0.470588, 0.473684, 0.500000, 0.409091],
            "Macro F1": [0.568655, 0.580531, 0.564631, 0.551136],
            "ROC-AUC": [0.590711, 0.636430, 0.617562, 0.513788]
        })
        
        # Menampilkan tabel dengan highlight/styling bawaan pandas
        st.dataframe(
            leaderboard_data.style.background_gradient(cmap="Greens", subset=["Accuracy", "Macro F1", "ROC-AUC"]),
            use_container_width=True,
            hide_index=True
        )
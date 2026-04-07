import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Palm Oil Management Dashboard", 
    page_icon="🌴", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS UNTUK TAMPILAN PREMIUM ---
st.markdown("""
    <style>
    /* Styling untuk warna utama teks metrik */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        color: #A5D6A7; /* Berubah jadi hijau muda terang agar cocok di Dark Mode */
        font-weight: 800;
    }
    /* Mengubah warna font delta (peningkatan/penurunan) */
    div[data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    
    /* Pembaruan Judul */
    .premium-title {
        font-family: 'Helvetica Neue', sans-serif;
        color: #FFFFFF; /* Putih bersih agar sangat terang */
        font-size: 2.8rem !important; /* Ukuran diperbesar drastis */
        font-weight: 800;
        margin-bottom: -10px;
    }
    .premium-subtitle {
        color: #81C784; /* Hijau muda terang (Soft Mint) */
        font-size: 1.3rem !important; /* Subtitle juga sedikit diperbesar */
        font-weight: 500;
        margin-bottom: 20px;
    }

    /* Styling card/container */
    div[data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid #424242; /* Border disesuaikan untuk dark mode */
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)


# --- FUNGSI PROSES DATA ---
@st.cache_data
def process_data(file):
    if file is not None:
        try:
            file.seek(0) 
            df = pd.read_csv(file, sep=None, engine='python')
            df.columns = df.columns.str.strip()
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            return None
    else:
        # Data Dummy Default
        np.random.seed(42)
        data = {
            'bulan_mupuk': pd.date_range(start='2025-01-01', periods=20, freq='ME'),
            'ID_Blok': [f"Blok {chr(65+i%3)}{i%5}" for i in range(20)],
            'umur_tanaman': np.random.randint(3, 25, 20),
            'populasi_ha': np.random.randint(136, 143, 20),
            'pupuk_N': np.random.randint(100, 250, 20),
            'pupuk_P': np.random.randint(50, 150, 20),
            'pupuk_K': np.random.randint(100, 250, 20),
            'curah_hujan': np.random.randint(50, 400, 20),
            'hari_hujan': np.random.randint(5, 25, 20),
            'yield_actual': np.random.randint(1500, 3500, 20),
            'biaya_pupuk': np.random.randint(2000000, 4000000, 20)
        }
        df = pd.DataFrame(data)
    
    return df

# --- LOAD MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "FO3rfmodel.pkl")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        # Hapus error st.error agar UI bersih jika model tidak ada (bisa disesuaikan)
        return None

rf_model = load_model()

# --- SIDEBAR (KHUSUS UNTUK FILTER GLOBAL & UPLOAD) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3105/3105307.png", width=60) # Ikon opsional
    st.header("Konfigurasi Data")
    uploaded_file = st.file_uploader("Upload Dataset Kebun (CSV)", type=["csv"])

    # Ambil data
    try:
        df = process_data(uploaded_file)
    except Exception as e:
        st.error(f"Gagal memproses file. Error: {e}")
        st.stop()

    st.divider()
    with st.expander("📝 Panduan Format CSV"):
        st.caption("""
        1. **bulan_mupuk**: (YYYY-MM-DD).
        2. **ID_Blok**: Kode unik.
        3. **umur_tanaman**: Tahun.
        4. **populasi_ha**: Pokok/ha.
        5. **pupuk_N, pupuk_P, pupuk_K**: kg/ha.
        6. **curah_hujan**: mm.
        7. **hari_hujan**: Hari.
        8. **yield_actual**: kg/ha.
        9. **biaya_pupuk**: Rupiah.
        """)
        # Tombol Download Template
        template_data = pd.DataFrame({
            'bulan_mupuk': ['2025-01-01'], 'ID_Blok': ['A01'], 'umur_tanaman': [5], 'populasi_ha':[140],
            'pupuk_N': [150], 'pupuk_P': [100], 'pupuk_K': [120],
            'curah_Hujan': [250], 'hari_hujan': [15], 'yield_actual': [2200], 'biaya_pupuk': [2850000]
        })
        csv_template = template_data.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Template", data=csv_template, file_name='template_kebun.csv', mime='text/csv')
        
    st.divider()
    st.subheader("Filter Area")
    # Multiselect dengan desain yang lebih rapi
    selected_blok = st.multiselect("Pilih Blok Lahan:", df['ID_Blok'].unique(), default=df['ID_Blok'].unique())
    filtered_df = df[df['ID_Blok'].isin(selected_blok)]
    
# --- HEADER UTAMA ---
st.markdown('<p class="premium-title">🌴 Palm Oil Executive Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="premium-subtitle">Analisis Produktivitas dan Optimasi Nutrisi Berbasis Machine Learning</p>', unsafe_allow_html=True)


# --- GLOBAL CONTROL PANEL (Di Main Page, dekat output) ---
with st.expander("⚙️ Parameter Bisnis & Finansial", expanded=True):
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        harga_tbs = st.number_input("Harga TBS Aktual (Rp/kg)", min_value=0, value=2800, step=100)
    with col_p2:
        target_yield_rkap = st.number_input("Target Yield RKAP (kg/ha)", min_value=0, value=3000, step=100)
    with col_p3:
        total_budget_planned = st.number_input("Budget Pupuk Keseluruhan (Rp)", min_value=0, value=500000000, step=10000000)

# Proses penambahan kolom Revenue & Profit secara dinamis berdasarkan input Harga TBS
col_yield = next((c for c in filtered_df.columns if 'yield_actual' in c.lower()), None)
col_biaya = next((c for c in filtered_df.columns if 'biaya_pupuk' in c.lower()), None)

if col_yield and col_biaya:
    filtered_df['Revenue'] = filtered_df[col_yield] * harga_tbs
    filtered_df['Profit'] = filtered_df['Revenue'] - filtered_df[col_biaya]

# Kalkulasi Metrik Utama
total_actual_cost = filtered_df[col_biaya].sum() if col_biaya else 0
absorption_rate = (total_actual_cost / total_budget_planned) * 100 if total_budget_planned > 0 else 0


# --- KARTU KPI (RINGKASAN EKSEKUTIF) ---
st.write("") # Spacer
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_yield = filtered_df[col_yield].sum()/1000 if col_yield else 0
    st.metric("Total Produksi (Ton)", f"{total_yield:,.1f}")
with col2:
    avg_yield = filtered_df[col_yield].mean() if col_yield else 0
    st.metric("Avg Yield (kg/ha)", f"{avg_yield:,.0f}", delta=f"Target: {target_yield_rkap}", delta_color="off")
with col3:
    avg_cost = (total_actual_cost/filtered_df[col_yield].sum()) if col_yield and filtered_df[col_yield].sum() > 0 else 0
    st.metric("Avg Cost (Rp/kg)", f"Rp {avg_cost:,.0f}")
with col4:
    sisa_budget = total_budget_planned - total_actual_cost
    st.metric(
        label="Budget Serapan", 
        value=f"{absorption_rate:.1f}%", 
        delta=f"Sisa: Rp {sisa_budget:,.0f}",
        delta_color="normal" if sisa_budget >= 0 else "inverse"
    )
    st.progress(min(absorption_rate / 100, 1.0))

st.write("---")

# --- VISUALISASI UTAMA (TABS) ---
tab1, tab2, tab3 = st.tabs(["📈 Performance Analysis", "🌐 AI Nutrient Optimization", "💰 Financial Insight"])

with tab1:
    st.markdown("#### 📊 Analisis Tren Produksi")
    
    col_date = next((c for c in filtered_df.columns if 'bulan_mupuk' in c.lower() or 'date' in c.lower()), None)
    col_id = next((c for c in filtered_df.columns if 'id_blok' in c.lower()), 'ID_Blok')
    col_umur = next((c for c in filtered_df.columns if 'umur' in c.lower()), 'umur_tanaman')

    if col_yield and col_date:
        df_trend = filtered_df.groupby(col_date).agg({col_yield: 'mean'}).reset_index()
        df_trend['Target_Sidebar'] = target_yield_rkap 
        df_trend = df_trend.sort_values(col_date)
        
        fig_yield = px.line(
            df_trend, x=col_date, y=[col_yield, 'Target_Sidebar'], markers=True,
            color_discrete_map={col_yield: "#A5D6A7", 'Target_Sidebar': "#d62828"}
        )
        fig_yield.update_traces(line=dict(dash='dash'), selector=dict(name='Target_Sidebar'), line_width=3)
        fig_yield.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', # Latar transparan (elegan)
            yaxis_title="Yield (kg/ha)", xaxis_title="", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_yield, use_container_width=True)

    st.markdown("#### 🏆 Kinerja Blok Lahan")
    col_left, col_right = st.columns(2)
    with col_left:
        st.success("**Top 5 Blok Terproduktif**")
        top_5 = filtered_df.nlargest(5, col_yield)[[col_id, col_yield, col_umur]]
        st.dataframe(top_5.reset_index(drop=True).style.format({col_yield: "{:,.0f} kg"}), use_container_width=True)
    with col_right:
        st.error("**Bottom 5 Blok Perlu Perhatian**")
        bottom_5 = filtered_df.nsmallest(5, col_yield)[[col_id, col_yield, col_umur]]
        st.dataframe(bottom_5.reset_index(drop=True).style.format({col_yield: "{:,.0f} kg"}), use_container_width=True)


with tab2:
    st.markdown("#### 🌐 AI Precision Fertilizing")
    if filtered_df.empty or rf_model is None:
        st.info("👋 Unggah data dan pastikan file model (.pkl) tersedia untuk memulai simulasi AI.")
    else:
        # Layout simulasi di dalam Tab AI (Memindahkan input sedekat mungkin dengan output)
        st.write("Sesuaikan parameter penambahan dosis pupuk Nitrogen untuk melihat estimasi kenaikan produksi:")
        simulasi_n = st.slider("Simulasi Penambahan Nitrogen (%)", min_value=0, max_value=50, value=10, step=5)
        
        df_pred = filtered_df.copy()
        df_pred.columns = df_pred.columns.str.strip().str.lower()
        if 'populasi_ha' not in df_pred.columns: df_pred['populasi_ha'] = 143

        feature_cols = ['umur_tanaman', 'curah_hujan', 'hari_hujan', 'populasi_ha', 'pupuk_n', 'pupuk_p', 'pupuk_k']
        
        try:
            X_input = df_pred[feature_cols]
            yield_saat_ini = rf_model.predict(X_input.values)
            
            # Terapkan persentase dinamis dari slider
            X_input_plus = X_input.copy()
            X_input_plus['pupuk_n'] = X_input_plus['pupuk_n'] * (1 + simulasi_n/100)
            yield_if_plus_n = rf_model.predict(X_input_plus.values)
            
            rekom_df = filtered_df.copy()
            rekom_df['Estimasi Yield (kg/ha)'] = yield_saat_ini
            rekom_df[f'Potensi Yield (N +{simulasi_n}%)'] = yield_if_plus_n
            rekom_df['Kenaikan (kg)'] = yield_if_plus_n - yield_saat_ini
            
            cols_to_show = ['ID_Blok', 'Estimasi Yield (kg/ha)', f'Potensi Yield (N +{simulasi_n}%)', 'Kenaikan (kg)']
            display_df = rekom_df[cols_to_show]
            
            st.dataframe(
                display_df.sort_values('Kenaikan (kg)', ascending=False).style.format({
                    'Estimasi Yield (kg/ha)': '{:.1f}',
                    f'Potensi Yield (N +{simulasi_n}%)': '{:.1f}',
                    'Kenaikan (kg)': '{:.2f}'
                })
                .highlight_max(subset=['Kenaikan (kg)'], color="off") # Highlight hijau lembut
                .background_gradient(subset=[f'Potensi Yield (N +{simulasi_n}%)'], cmap='Greens')
            )
        except Exception as e:
             st.warning(f"Pastikan format kolom sesuai untuk model AI. {e}")


with tab3:
    st.markdown("#### 💰 Financial Insight & Profitability")
    if not filtered_df.empty:
        df_profit_summary = filtered_df.groupby('ID_Blok')['Profit'].sum().reset_index()
        df_profit_summary = df_profit_summary.sort_values('Profit', ascending=False)
        
        fig_profit = px.bar(
            df_profit_summary, x='ID_Blok', y='Profit', color='Profit',
            color_continuous_scale='Greens', text_auto='.3s' 
        )
        fig_profit.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Kode Blok", yaxis_title="Total Profit (Rp)",
            title=f"Peringkat Profitabilitas per Blok (Harga TBS: Rp {harga_tbs:,}/kg)"
        )
        st.plotly_chart(fig_profit, use_container_width=True)

        # Kontrol simulasi biaya didekatkan dengan grafiknya
        st.markdown("##### 📉 Simulasi Kenaikan Harga Pupuk")
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
             persen_naik = st.number_input("Simulasi Kenaikan Biaya (%)", min_value=0, max_value=200, value=15, step=5)
             
             df_fin = filtered_df.copy()
             df_fin['Biaya_Simulasi'] = df_fin[col_biaya] * (1 + persen_naik/100)
             df_fin['Profit_Simulasi'] = df_fin['Revenue'] - df_fin['Biaya_Simulasi']
             
             impact = df_fin['Profit_Simulasi'].sum() - df_fin['Profit'].sum()
             st.metric("Total Dampak ke Profit", f"Rp {impact:,.0f}", f"-{persen_naik}% kenaikan harga", delta_color="inverse")

        with col_s2:
            df_sens = df_fin.groupby('ID_Blok')[['Profit', 'Profit_Simulasi']].sum().reset_index()
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Bar(x=df_sens['ID_Blok'], y=df_sens['Profit'], name='Profit Saat Ini', marker_color='#2d6a4f'))
            fig_sens.add_trace(go.Bar(x=df_sens['ID_Blok'], y=df_sens['Profit_Simulasi'], name=f'Setelah Kenaikan', marker_color='#e07a5f'))
            
            fig_sens.update_layout(
                barmode='group', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_sens, use_container_width=True)


# ---FOOTER---
st.caption('Dashboard dikembangkan oleh Ham D Roger v1.1 - 2026 | Powered by Machine Learning')
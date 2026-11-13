"""
ChurnGuard - Aplikasi Web Prediksi Churn Pelanggan Telco
Logika aplikasi ada di file ini, tampilan ada di style.css
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="ChurnGuard | Prediksi Churn Pelanggan",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_churn.pkl"
CSS_PATH = BASE_DIR / "style.css"
AUTHOR = "Alya Andina"


def h(html: str) -> str:
    """Ratakan HTML jadi satu blok agar tidak terbaca sebagai code block oleh Markdown."""
    return " ".join(baris.strip() for baris in html.splitlines() if baris.strip())


if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ==========================================
# STATUS ANALISIS
# ==========================================

if "hasil_analisis" not in st.session_state:
    st.session_state.hasil_analisis = None
if "data_pelanggan" not in st.session_state:
    st.session_state.data_pelanggan = None


def ulangi_analisis():
    st.session_state.hasil_analisis = None
    st.session_state.data_pelanggan = None


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_error = None
except Exception as e:  # noqa: BLE001
    model = None
    model_error = str(e)

NAMA_MODEL = {
    "LogisticRegression": "Logistic Regression",
    "DecisionTreeClassifier": "Decision Tree",
    "RandomForestClassifier": "Random Forest",
}
nama_model = "Machine Learning"
if model is not None:
    try:
        jenis = type(model.named_steps["model"]).__name__
    except Exception:  # noqa: BLE001
        jenis = type(model).__name__
    nama_model = NAMA_MODEL.get(jenis, jenis)

# ==========================================
# PILIHAN INPUT (nilai dataset -> label Indonesia)
# ==========================================

GENDER = {"Male": "Laki-laki", "Female": "Perempuan"}
YA_TIDAK = {"Yes": "Ya", "No": "Tidak"}
LANSIA = {0: "Bukan lansia", 1: "Lansia (65+)"}
TELEPON_GANDA = {"No": "Tidak", "Yes": "Ya", "No phone service": "Tidak ada layanan telepon"}
INTERNET = {"DSL": "DSL", "Fiber optic": "Fiber optic", "No": "Tidak berlangganan"}
TAMBAHAN = {"No": "Tidak", "Yes": "Ya", "No internet service": "Tidak ada layanan internet"}
KONTRAK = {"Month-to-month": "Bulanan", "One year": "1 Tahun", "Two year": "2 Tahun"}
PEMBAYARAN = {
    "Electronic check": "Cek elektronik",
    "Mailed check": "Cek via pos",
    "Bank transfer (automatic)": "Transfer bank otomatis",
    "Credit card (automatic)": "Kartu kredit otomatis",
}


def pilih(label, opsi, index=0, key=None):
    """Selectbox berlabel Indonesia yang mengembalikan nilai asli dataset."""
    return st.selectbox(label, list(opsi.keys()), index=index,
                        format_func=lambda v: opsi[v], key=key)


# ==========================================
# PERSIAPAN DATA (sama persis dengan notebook)
# ==========================================

def kelompok_tenure(bulan):
    if bulan <= 12:
        return "0-12"
    if bulan <= 24:
        return "13-24"
    if bulan <= 48:
        return "25-48"
    return "49-72"


def hitung_layanan(d):
    daftar = [d["PhoneService"], d["MultipleLines"], d["OnlineSecurity"], d["OnlineBackup"],
              d["DeviceProtection"], d["TechSupport"], d["StreamingTV"], d["StreamingMovies"]]
    return sum(1 for x in daftar if x == "Yes")


def buat_fitur(d):
    total = d["MonthlyCharges"] * d["tenure"]  # perkiraan TotalCharges
    data = {
        **d,
        "TotalCharges": total,
        "tenure_group": kelompok_tenure(d["tenure"]),
        "num_services": hitung_layanan(d),
        "avg_charge": total / (d["tenure"] + 1),
    }
    df = pd.DataFrame([data])
    # Kompatibilitas pandas/scikit-learn: kolom teks dijadikan tipe object
    return df.astype({c: object for c in df.select_dtypes(exclude="number").columns})


def siapkan_input(model, df):
    harapan = getattr(model, "feature_names_in_", None)
    if harapan is None:
        return df
    harapan = list(harapan)
    kurang = [c for c in harapan if c not in df.columns]
    if kurang:
        raise ValueError("Kolom berikut tidak ditemukan pada input: " + ", ".join(kurang))
    return df[harapan]


def ambil_prob_churn(model, df):
    """Mengembalikan (prob_churn, {label_kelas: probabilitas})."""
    proba = model.predict_proba(df)[0]
    peta, prob_churn = {}, 0.0
    for kelas, p in zip(model.classes_, proba):
        akan_churn = kelas in (1, "1", "Yes", True)
        peta["Akan Churn" if akan_churn else "Tetap Berlangganan"] = float(p)
        if akan_churn:
            prob_churn = float(p)
    return prob_churn, peta


def tingkat_risiko(prob):
    if prob >= 0.5:
        return ("churn", "⚠️", "Berisiko Churn",
                "Pelanggan ini diprediksi akan berhenti berlangganan. Disarankan segera "
                "melakukan tindakan retensi.")
    if prob >= 0.35:
        return ("waspada", "👀", "Perlu Diwaspadai",
                "Pelanggan diprediksi masih bertahan, tetapi potensi churn cukup besar. "
                "Pantau dan pertimbangkan penawaran khusus.")
    return ("aman", "✅", "Cenderung Bertahan",
            "Pelanggan ini diprediksi tetap berlangganan dan tidak menunjukkan risiko yang kuat.")


def indikator(d):
    """Indikator umum pada data pelanggan (berdasarkan pola dataset Telco)."""
    risiko, pelindung = [], []
    if d["Contract"] == "Month-to-month":
        risiko.append(("📝", "Kontrak bulanan",
                       "Tanpa ikatan jangka panjang, pelanggan mudah berhenti.",
                       "Tawarkan diskon atau bonus untuk kontrak 1 atau 2 tahun."))
    if d["tenure"] <= 12:
        risiko.append(("🌱", f"Pelanggan baru ({d['tenure']} bulan)",
                       "Pelanggan dengan masa langganan singkat lebih rentan churn.",
                       "Berikan promo sambutan dan pendampingan di tahun pertama."))
    if d["InternetService"] == "Fiber optic":
        risiko.append(("📶", "Internet fiber optic",
                       "Pada dataset Telco, pengguna fiber optic cenderung lebih sering churn.",
                       "Pastikan kualitas jaringan dan harga sebanding dengan kebutuhan."))
    if d["InternetService"] != "No" and d["TechSupport"] == "No":
        risiko.append(("🛠️", "Tanpa dukungan teknis",
                       "Pelanggan internet tanpa dukungan teknis lebih mudah berpindah.",
                       "Tawarkan paket dukungan teknis gratis atau berdiskon."))
    if d["PaymentMethod"] == "Electronic check":
        risiko.append(("💳", "Pembayaran cek elektronik",
                       "Metode ini sering dikaitkan dengan tingkat churn yang lebih tinggi.",
                       "Ajak beralih ke pembayaran otomatis (transfer bank atau kartu kredit)."))
    if d["MonthlyCharges"] >= 80:
        risiko.append(("💰", "Biaya bulanan tinggi",
                       "Tagihan besar dapat memicu pelanggan mencari alternatif.",
                       "Tinjau paket dan tawarkan opsi bundling yang lebih hemat."))
    if d["Contract"] in ("One year", "Two year"):
        pelindung.append(("🔒", f"Kontrak {KONTRAK[d['Contract']]}", "Ikatan jangka panjang menurunkan risiko churn."))
    if d["tenure"] >= 49:
        pelindung.append(("🏅", f"Pelanggan lama ({d['tenure']} bulan)", "Pelanggan lama umumnya lebih loyal."))
    if d["TechSupport"] == "Yes" or d["OnlineSecurity"] == "Yes":
        pelindung.append(("🛡️", "Memakai layanan dukungan", "Layanan keamanan atau dukungan teknis meningkatkan loyalitas."))
    if d["PaymentMethod"] in ("Bank transfer (automatic)", "Credit card (automatic)"):
        pelindung.append(("✔️", "Pembayaran otomatis", "Pembayaran otomatis membuat pelanggan lebih konsisten."))
    return risiko[:5], pelindung[:4]


def baris_probabilitas(peta, label_atas):
    baris = []
    for label, p in sorted(peta.items(), key=lambda x: x[1], reverse=True):
        pct = max(0.0, min(100.0, p * 100))
        atas = " top" if label == label_atas else ""
        lencana = '<em class="prob-badge">Prediksi</em>' if label == label_atas else ""
        baris.append(
            f'<div class="prob-row{atas}"><div class="prob-head">'
            f'<span class="prob-name">{label}{lencana}</span>'
            f'<span class="prob-pct">{pct:.1f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{pct:.2f}%"></div></div></div>'
        )
    return '<div class="prob-card">' + "".join(baris) + "</div>"


# ==========================================
# HEADER DAN HERO
# ==========================================

st.markdown(h("""
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">📡</div>
    <div>
      <div class="brand-name">CHURN<span>GUARD</span></div>
      <div class="brand-caption">SMART CUSTOMER RETENTION</div>
    </div>
  </div>
  <div class="status-pill"><span class="status-dot"></span>MACHINE LEARNING</div>
</div>
"""), unsafe_allow_html=True)

st.markdown(h("""
<div class="hero">
  <div class="hero-copy">
    <div class="eyebrow">✦ CUSTOMER CHURN ANALYSIS</div>
    <h1>Kenali Pelanggan<br>yang Akan Pergi.</h1>
    <p>Analisis profil, kontrak, tagihan, dan layanan pelanggan untuk memprediksi
    kemungkinan mereka berhenti berlangganan, lengkap dengan indikator risiko
    dan saran retensi.</p>
    <div class="hero-tags">
      <span>◆ Prediksi Churn</span>
      <span>◉ Indikator Risiko</span>
      <span>✦ Saran Retensi</span>
    </div>
  </div>
  <div class="hero-visual">
    <div class="ring ring-1"></div>
    <div class="ring ring-2"></div>
    <div class="ring ring-3"></div>
    <div class="ring ring-4"></div>
    <div class="hero-emoji">📡</div>
    <div class="floating-card card-top">
      <div class="fc-icon">✦</div>
      <div><div class="fc-title">AI Powered</div><div class="fc-sub">Churn Prediction</div></div>
    </div>
    <div class="floating-card card-bottom">
      <span class="mini-dot"></span>
      <div><div class="fc-title">18 Parameter</div><div class="fc-sub">Customer Profile</div></div>
    </div>
  </div>
</div>
"""), unsafe_allow_html=True)

if model_error:
    st.error(f"Model gagal dimuat: {model_error}\n\n"
             "Pastikan file model_churn.pkl berada satu folder dengan app.py.")
    st.stop()

# ==========================================
# FORM INPUT
# ==========================================

st.markdown(h("""
<div class="section-heading">
  <div>
    <div class="eyebrow">INPUT PARAMETERS</div>
    <h2>Data Pelanggan</h2>
    <p>Masukkan data pelanggan yang ingin dianalisis.</p>
  </div>
  <div class="step-label">STEP 01 / 02</div>
</div>
"""), unsafe_allow_html=True)

with st.form("form_churn"):

    st.markdown('<div class="input-section-title">01 &nbsp; Kontrak dan Tagihan</div>',
                unsafe_allow_html=True)
    k1, k2 = st.columns(2, gap="large")
    with k1:
        tenure = st.slider("Lama berlangganan (bulan)", 0, 72, 12,
                           help="Sudah berapa bulan pelanggan berlangganan.")
        contract = pilih("Jenis kontrak", KONTRAK)
        paperless = pilih("Tagihan tanpa kertas (paperless)", YA_TIDAK)
    with k2:
        monthly = st.slider("Biaya bulanan (USD)", 18.0, 120.0, 70.0, 0.5,
                            help="Tagihan per bulan dalam dolar.")
        payment = pilih("Metode pembayaran", PEMBAYARAN)

    st.markdown('<div class="input-section-title second-title">02 &nbsp; Profil Pelanggan</div>',
                unsafe_allow_html=True)
    p1, p2 = st.columns(2, gap="medium")
    with p1:
        gender = pilih("Jenis kelamin", GENDER)
        partner = pilih("Memiliki pasangan", YA_TIDAK, index=1)
    with p2:
        senior = pilih("Kelompok usia", LANSIA)
        dependents = pilih("Memiliki tanggungan", YA_TIDAK, index=1)

    st.markdown('<div class="input-section-title second-title">03 &nbsp; Layanan yang Digunakan</div>',
                unsafe_allow_html=True)
    l1, l2, l3 = st.columns(3, gap="medium")
    with l1:
        phone = pilih("Layanan telepon", YA_TIDAK)
        security = pilih("Keamanan online", TAMBAHAN)
        tv = pilih("Streaming TV", TAMBAHAN)
    with l2:
        multi = pilih("Jalur telepon ganda", TELEPON_GANDA)
        backup = pilih("Cadangan online", TAMBAHAN)
        movies = pilih("Streaming film", TAMBAHAN)
    with l3:
        internet = pilih("Layanan internet", INTERNET, index=1)
        protection = pilih("Perlindungan perangkat", TAMBAHAN)
        tech = pilih("Dukungan teknis", TAMBAHAN)

    st.caption("Jika pelanggan tidak memakai telepon atau internet, layanan terkait otomatis "
               "dianggap tidak tersedia.")

    kirim = st.form_submit_button("📡  ANALISIS RISIKO CHURN", use_container_width=True)

# ==========================================
# PROSES PREDIKSI
# ==========================================

if kirim:
    # Samakan dengan pola dataset: tanpa telepon/internet, layanan terkait = "tidak tersedia"
    if phone == "No":
        multi = "No phone service"
    if internet == "No":
        security = backup = protection = tech = tv = movies = "No internet service"

    data_input = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multi, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": tech,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly,
    }

    try:
        df_fitur = buat_fitur(data_input)
        df_model = siapkan_input(model, df_fitur)
        prob_churn, peta = ambil_prob_churn(model, df_model)
        kelas, ikon, judul, deskripsi = tingkat_risiko(prob_churn)
        label_prediksi = "Akan Churn" if prob_churn >= 0.5 else "Tetap Berlangganan"

        st.session_state.hasil_analisis = {
            "prob_churn": prob_churn,
            "peta": peta,
            "kelas": kelas, "ikon": ikon, "judul": judul, "deskripsi": deskripsi,
            "label_prediksi": label_prediksi,
            "keyakinan": peta.get(label_prediksi, prob_churn) * 100,
            "jumlah_layanan": int(df_fitur["num_services"].iloc[0]),
            "total_tagihan": float(df_fitur["TotalCharges"].iloc[0]),
        }
        st.session_state.data_pelanggan = data_input

    except Exception as e:  # noqa: BLE001
        st.error("Prediksi belum dapat dilakukan karena format input tidak sesuai dengan "
                 "model yang tersimpan.")
        st.code(str(e))
        st.warning("Pastikan model_churn.pkl dibuat dari notebook proyek ini, dengan fitur "
                   "dan versi pandas/scikit-learn yang sama.")

# ==========================================
# HASIL
# ==========================================

if st.session_state.hasil_analisis is not None:
    hasil = st.session_state.hasil_analisis
    d = st.session_state.data_pelanggan
    pct_churn = hasil["prob_churn"] * 100

    st.markdown('<div id="hasil-prediksi"></div>', unsafe_allow_html=True)
    components.html(
        """
        <script>
        setTimeout(function () {
            const t = window.parent.document.getElementById("hasil-prediksi");
            if (t) { t.scrollIntoView({behavior: "smooth", block: "start"}); }
        }, 350);
        </script>
        """,
        height=0,
    )

    st.markdown(h("""
    <div class="section-heading result-heading">
      <div>
        <div class="eyebrow">ANALYSIS COMPLETE</div>
        <h2>Hasil Prediksi</h2>
        <p>Berikut hasil analisis berdasarkan data pelanggan.</p>
      </div>
      <div class="step-label">STEP 02 / 02</div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown(h(f"""
    <div class="result-card {hasil['kelas']}">
      <div class="result-left">
        <div class="result-icon">{hasil['ikon']}</div>
        <div class="result-label">HASIL KLASIFIKASI</div>
        <div class="result-name">{hasil['judul']}</div>
        <p>{hasil['deskripsi']}</p>
      </div>
      <div class="result-right">
        <div class="result-ring">
          <div class="result-ring-inner">
            <span class="ring-number">{pct_churn:.0f}%</span>
            <span class="ring-caption">RISIKO CHURN</span>
          </div>
        </div>
        <div class="result-right-caption">MODEL {nama_model.upper()}</div>
      </div>
    </div>
    """), unsafe_allow_html=True)

    st.button("↻ Analisis Pelanggan Lain", key="ulang_atas",
              use_container_width=True, on_click=ulangi_analisis)

    # ---------- Keyakinan model ----------
    st.markdown('<div class="subsection-title">Tingkat Risiko dan Keyakinan Model</div>',
                unsafe_allow_html=True)
    st.markdown(h(f"""
    <div class="conf-grid">
      <div class="confidence-card">
        <div class="confidence-top">
          <span>Probabilitas pelanggan churn</span>
          <strong>{pct_churn:.1f}%</strong>
        </div>
        <div class="confidence-track">
          <div class="confidence-fill" style="width:{pct_churn:.2f}%"></div>
          <div class="threshold" style="left:50%"><span>batas 50%</span></div>
        </div>
        <div class="confidence-foot">Pelanggan dikategorikan churn jika probabilitas mencapai 50% atau lebih.</div>
      </div>
      <div class="metric-card">
        <div class="metric-icon">✦</div>
        <div class="metric-label">KEYAKINAN PREDIKSI</div>
        <div class="metric-value">{hasil['keyakinan']:.1f}%</div>
        <div class="metric-desc">Probabilitas kelas terpilih</div>
      </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="subsection-title small-subsection">Probabilitas Setiap Kelas</div>',
                unsafe_allow_html=True)
    st.markdown(baris_probabilitas(hasil["peta"], hasil["label_prediksi"]), unsafe_allow_html=True)
    st.caption("Catatan: probabilitas model bukan jaminan mutlak. Hasil dipengaruhi kualitas "
               "data pelatihan.")

    # ---------- Indikator dan saran ----------
    risiko, pelindung = indikator(d)

    st.markdown('<div class="subsection-title">Indikator pada Data Pelanggan</div>',
                unsafe_allow_html=True)
    kartu = []
    for ikon, judul, isi, _ in risiko:
        kartu.append(f'<div class="indikator naik"><div class="ind-ikon">{ikon}</div>'
                     f'<div><span class="ind-tag">Faktor risiko</span><b>{judul}</b><p>{isi}</p></div></div>')
    for ikon, judul, isi in pelindung:
        kartu.append(f'<div class="indikator turun"><div class="ind-ikon">{ikon}</div>'
                     f'<div><span class="ind-tag">Faktor pelindung</span><b>{judul}</b><p>{isi}</p></div></div>')
    if not kartu:
        kartu.append('<div class="indikator netral"><div class="ind-ikon">➖</div>'
                     '<div><b>Tidak ada indikator menonjol</b>'
                     '<p>Data pelanggan tidak memicu indikator risiko maupun pelindung yang kuat.</p></div></div>')
    st.markdown('<div class="indikator-grid">' + "".join(kartu) + "</div>", unsafe_allow_html=True)
    st.caption("Indikator berdasarkan pola umum pada dataset Telco, bukan penjelasan langsung "
               "dari cara kerja model.")

    if hasil["prob_churn"] >= 0.35 and risiko:
        st.markdown('<div class="subsection-title">💡 Saran Retensi</div>', unsafe_allow_html=True)
        saran = "".join(
            f'<div class="saran"><div class="saran-no">{i}</div>'
            f'<div><b>{judul}</b><p>{tindakan}</p></div></div>'
            for i, (_, judul, _, tindakan) in enumerate(risiko[:4], start=1)
        )
        st.markdown('<div class="saran-list">' + saran + "</div>", unsafe_allow_html=True)

    # ---------- Ringkasan ----------
    st.markdown('<div class="subsection-title">Ringkasan Data yang Dianalisis</div>',
                unsafe_allow_html=True)
    ringkasan = [
        ("LAMA LANGGANAN", f"{d['tenure']} bulan"),
        ("BIAYA BULANAN", f"${d['MonthlyCharges']:.2f}"),
        ("LAYANAN AKTIF", f"{hasil['jumlah_layanan']} layanan"),
        ("PERKIRAAN TOTAL TAGIHAN", f"${hasil['total_tagihan']:,.0f}"),
    ]
    st.markdown('<div class="summary-grid">' + "".join(
        f'<div class="summary-card"><div class="summary-label">{a}</div>'
        f'<div class="summary-value">{b}</div></div>' for a, b in ringkasan) + "</div>",
        unsafe_allow_html=True)

    st.markdown(h(f"""
    <div class="detail-card">
      <div><span>Jenis kontrak</span><strong>{KONTRAK[d['Contract']]}</strong></div>
      <div><span>Metode pembayaran</span><strong>{PEMBAYARAN[d['PaymentMethod']]}</strong></div>
      <div><span>Layanan internet</span><strong>{INTERNET[d['InternetService']]}</strong></div>
      <div><span>Tagihan paperless</span><strong>{YA_TIDAK[d['PaperlessBilling']]}</strong></div>
      <div><span>Jenis kelamin</span><strong>{GENDER[d['gender']]}</strong></div>
      <div><span>Kelompok usia</span><strong>{LANSIA[d['SeniorCitizen']]}</strong></div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown(h("""
    <div class="disclaimer">
      <strong>Informasi:</strong> Hasil ini merupakan prediksi berdasarkan model machine
      learning dan bukan keputusan akhir. Total tagihan diperkirakan dari biaya bulanan
      dikali lama berlangganan.
    </div>
    """), unsafe_allow_html=True)

    st.button("↻ Analisis Pelanggan Lain", key="ulang_bawah",
              use_container_width=True, on_click=ulangi_analisis)

# ==========================================
# FOOTER
# ==========================================

st.markdown(h(f"""
<div class="footer">
  <div class="footer-brand">📡 CHURNGUARD</div>
  <div>SMART RETENTION SYSTEM</div>
  <div class="footer-note">Aplikasi prediksi churn pelanggan berbasis Machine Learning</div>
  <div class="footer-note">Author : {AUTHOR}</div>
</div>
"""), unsafe_allow_html=True)
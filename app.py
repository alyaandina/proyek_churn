"""
ChurnGuard | Aplikasi Web Prediksi Churn Pelanggan Telco
--------------------------------------------------------
File ini berisi logika aplikasi (Python). Tampilan (CSS) ada di style.css.
Alur dan komponen tampilan mengikuti proyek "learning" (Mango AI),
disesuaikan untuk kebutuhan prediksi churn dan nyaman dibuka di HP.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ======================================================================
# KONFIGURASI HALAMAN
# ======================================================================
st.set_page_config(
    page_title="ChurnGuard | Prediksi Churn Pelanggan",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_churn.pkl"
CSS_PATH = BASE_DIR / "style.css"

if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# Penjamin garis tepi: struktur HTML widget Streamlit berbeda antar versi (penanda
# data-baseweb tidak selalu ada), sehingga CSS saja kadang tidak mengenai kotaknya.
# Skrip ini mencari kotak isian dari posisinya di dalam widget (naik dari <input> sampai
# tepat di bawah label), lalu memasang garis langsung (inline) pada semua selectbox dan
# kolom angka. Dipasang ulang tiap halaman dirender. Dijalankan di dokumen induk agar tetap
# hidup walau iframe pembantu diganti.
components.html(
    """
    <script>
    (function () {
        const doc = window.parent.document;
        const lama = doc.getElementById("churnguard-border-script");
        if (lama) { lama.remove(); }
        const kode = `
            (function () {
                if (window.__churnguardObs) { window.__churnguardObs.disconnect(); }
                function kontrol(root) {
                    var start = root.querySelector('input, [role="combobox"]');
                    if (!start) { return null; }
                    var label = root.querySelector('[data-testid="stWidgetLabel"]');
                    var best = start, el = start;
                    while (el.parentElement && el.parentElement !== root) {
                        var p = el.parentElement;
                        if (label && p.contains(label)) { break; }
                        if (p.getBoundingClientRect().height > 64) { break; }
                        best = p; el = p;
                    }
                    return best;
                }
                function pasang() {
                    document.querySelectorAll('[data-testid="stSelectbox"], [data-testid="stNumberInput"]').forEach(function (root) {
                        var k = kontrol(root);
                        if (!k) { return; }
                        k.style.setProperty("outline", "1px solid #b8bfe3", "important");
                        k.style.setProperty("outline-offset", "0px", "important");
                        k.style.setProperty("border-radius", "10px", "important");
                        k.style.setProperty("background-color", "#ffffff", "important");
                    });
                }
                var tunda = null;
                function jadwalkan() { clearTimeout(tunda); tunda = setTimeout(pasang, 60); }
                pasang();
                window.__churnguardObs = new MutationObserver(jadwalkan);
                window.__churnguardObs.observe(document.body, {childList: true, subtree: true});
            })();
        `;
        const s = doc.createElement("script");
        s.id = "churnguard-border-script";
        s.textContent = kode;
        doc.head.appendChild(s);
    })();
    </script>
    """,
    height=0,
)


def html(blok: str) -> None:
    """Tampilkan blok HTML. Baris dirapatkan agar Markdown tidak salah membaca
    indentasi / baris kosong sebagai blok kode."""
    rapat = " ".join(baris.strip() for baris in blok.strip().splitlines())
    st.markdown(rapat, unsafe_allow_html=True)


# ======================================================================
# STATUS HASIL (disimpan di session_state seperti di proyek learning)
# ======================================================================
if "hasil" not in st.session_state:
    st.session_state.hasil = None


def ulangi_analisis() -> None:
    st.session_state.hasil = None


# ======================================================================
# MUAT MODEL
# ======================================================================
@st.cache_resource
def muat_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)


try:
    model = muat_model()
    model_error = None
except Exception as e:  # noqa: BLE001
    model = None
    model_error = str(e)

NAMA_MODEL = {
    "LogisticRegression": "Logistic Regression",
    "DecisionTreeClassifier": "Decision Tree",
    "RandomForestClassifier": "Random Forest",
}


def nama_algoritma(m) -> str:
    if m is None:
        return "Machine Learning"
    try:
        jenis = type(m.named_steps["model"]).__name__
    except Exception:  # noqa: BLE001
        jenis = type(m).__name__
    return NAMA_MODEL.get(jenis, jenis)


nama_model = nama_algoritma(model)

# Metrik hasil evaluasi di notebook (churn.ipynb, data uji).
METRIK_MODEL = {
    "Logistic Regression": {"Accuracy": 0.7374, "Precision": 0.5034, "Recall": 0.7807, "F1": 0.6122, "ROC-AUC": 0.8462},
    "Decision Tree":       {"Accuracy": 0.7459, "Precision": 0.5140, "Recall": 0.7834, "F1": 0.6208, "ROC-AUC": 0.8144},
    "Random Forest":       {"Accuracy": 0.7708, "Precision": 0.5503, "Recall": 0.7460, "F1": 0.6334, "ROC-AUC": 0.8414},
}

# ======================================================================
# PILIHAN INPUT (nilai asli dataset -> label bahasa Indonesia)
# ======================================================================
GENDER = {"Male": "Laki-laki", "Female": "Perempuan"}
YA_TIDAK = {"Yes": "Ya", "No": "Tidak"}
LANSIA = {0: "Bukan lansia", 1: "Lansia (65+)"}
TELEPON_GANDA = {"No": "Tidak", "Yes": "Ya", "No phone service": "Tanpa telepon"}
INTERNET = {"DSL": "DSL", "Fiber optic": "Fiber optic", "No": "Tidak berlangganan"}
TAMBAHAN = {"No": "Tidak", "Yes": "Ya", "No internet service": "Tanpa internet"}
KONTRAK = {"Month-to-month": "Bulanan", "One year": "1 Tahun", "Two year": "2 Tahun"}
PEMBAYARAN = {
    "Electronic check": "Cek elektronik",
    "Mailed check": "Cek via pos",
    "Bank transfer (automatic)": "Transfer bank otomatis",
    "Credit card (automatic)": "Kartu kredit otomatis",
}
KOLOM_TAMBAHAN = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
JUMLAH_PARAMETER = 18


def pilih(label: str, opsi: dict, index: int = 0):
    """Selectbox yang menampilkan label Indonesia tetapi mengembalikan nilai asli."""
    return st.selectbox(label, list(opsi.keys()), index=index, format_func=lambda v: opsi[v])


def penanda_dua_kolom() -> None:
    """Penanda tak terlihat: baris kolom ini tetap 2 kolom di layar HP (lihat style.css)."""
    st.markdown('<span class="dua-kolom"></span>', unsafe_allow_html=True)


# ======================================================================
# FUNGSI BANTU (logika prediksi)
# ======================================================================
def kelompok_tenure(bulan: int) -> str:
    if bulan <= 12:
        return "0-12"
    if bulan <= 24:
        return "13-24"
    if bulan <= 48:
        return "25-48"
    return "49-72"


def buat_dataframe(d: dict) -> pd.DataFrame:
    """Membuat fitur sama persis dengan yang dibuat di notebook."""
    total = d["MonthlyCharges"] * d["tenure"]  # perkiraan TotalCharges
    layanan = [d["PhoneService"], d["MultipleLines"], d["OnlineSecurity"],
               d["OnlineBackup"], d["DeviceProtection"], d["TechSupport"],
               d["StreamingTV"], d["StreamingMovies"]]
    d = {
        **d,
        "TotalCharges": total,
        "tenure_group": kelompok_tenure(d["tenure"]),
        "num_services": sum(1 for x in layanan if x == "Yes"),
        "avg_charge": total / (d["tenure"] + 1),
    }
    df = pd.DataFrame([d])
    # Kompatibilitas pandas/scikit-learn: kolom teks dijadikan tipe object
    return df.astype({c: object for c in df.select_dtypes(exclude="number").columns})


def hitung_probabilitas(data: dict) -> float:
    return float(model.predict_proba(buat_dataframe(data))[0][1])


def rapikan_input(d: dict) -> tuple[dict, list[str]]:
    """Menyelaraskan kombinasi input yang tidak mungkin ada di dataset
    (mis. tidak punya telepon tetapi 'jalur ganda = Ya')."""
    d = dict(d)
    catatan = []

    if d["PhoneService"] == "No":
        if d["MultipleLines"] != "No phone service":
            d["MultipleLines"] = "No phone service"
            catatan.append("Jalur telepon ganda disesuaikan menjadi “Tanpa telepon” "
                           "karena pelanggan tidak memakai layanan telepon.")
    elif d["MultipleLines"] == "No phone service":
        d["MultipleLines"] = "No"
        catatan.append("Jalur telepon ganda disesuaikan menjadi “Tidak” "
                       "karena pelanggan memakai layanan telepon.")

    if d["InternetService"] == "No":
        if any(d[k] != "No internet service" for k in KOLOM_TAMBAHAN):
            catatan.append("Layanan tambahan disesuaikan menjadi “Tanpa internet” "
                           "karena pelanggan tidak berlangganan internet.")
        for k in KOLOM_TAMBAHAN:
            d[k] = "No internet service"
    else:
        diubah = [k for k in KOLOM_TAMBAHAN if d[k] == "No internet service"]
        for k in diubah:
            d[k] = "No"
        if diubah:
            catatan.append("Pilihan “Tanpa internet” pada layanan tambahan diganti “Tidak” "
                           "karena pelanggan berlangganan internet.")
    return d, catatan


def tingkat_risiko(prob: float) -> dict:
    if prob >= 0.5:
        return {"kelas": "tinggi", "label": "Risiko Tinggi", "ikon": "🚨", "warna": "#dc2626",
                "teks": "Pelanggan ini diprediksi akan berhenti berlangganan (churn). "
                        "Disarankan segera dilakukan tindakan retensi."}
    if prob >= 0.35:
        return {"kelas": "sedang", "label": "Perlu Diwaspadai", "ikon": "⚠️", "warna": "#d97706",
                "teks": "Pelanggan ini diprediksi bertahan, namun memiliki potensi churn "
                        "yang cukup besar."}
    return {"kelas": "rendah", "label": "Risiko Rendah", "ikon": "✅", "warna": "#16a34a",
            "teks": "Pelanggan ini diprediksi tetap berlangganan."}


def faktor_profil(d: dict) -> tuple[list[str], list[str]]:
    """Indikator sederhana dari profil pelanggan (berdasarkan aturan, bukan keluaran model)."""
    risiko, aman = [], []
    if d["Contract"] == "Month-to-month":
        risiko.append("Kontrak bulanan")
    else:
        aman.append(f"Kontrak {KONTRAK[d['Contract']].lower()}")
    if d["tenure"] <= 12:
        risiko.append("Pelanggan baru (≤ 12 bulan)")
    elif d["tenure"] > 24:
        aman.append("Berlangganan lebih dari 24 bulan")
    if d["InternetService"] == "Fiber optic":
        risiko.append("Internet fiber optic")
    if d["InternetService"] != "No" and d["TechSupport"] == "No":
        risiko.append("Tanpa dukungan teknis")
    if d["TechSupport"] == "Yes":
        aman.append("Memakai dukungan teknis")
    if d["OnlineSecurity"] == "Yes":
        aman.append("Memakai keamanan online")
    if d["PaymentMethod"] == "Electronic check":
        risiko.append("Bayar dengan cek elektronik")
    elif d["PaymentMethod"] in ("Bank transfer (automatic)", "Credit card (automatic)"):
        aman.append("Pembayaran otomatis")
    if d["MonthlyCharges"] >= 80:
        risiko.append("Tagihan bulanan tinggi (≥ $80)")
    return risiko, aman


def saran_retensi(d: dict) -> list[dict]:
    """Saran tindakan. Jika `ubah` terisi, perubahan itu disimulasikan ke model."""
    saran = []
    if d["Contract"] == "Month-to-month":
        saran.append({"ikon": "📝", "judul": "Tawarkan kontrak jangka panjang",
                      "isi": "Pelanggan kontrak bulanan paling mudah berhenti. Beri diskon atau "
                             "bonus untuk kontrak 1 tahun.",
                      "ubah": {"Contract": "One year"}})
    if d["tenure"] <= 12:
        saran.append({"ikon": "🎁", "judul": "Program untuk pelanggan baru",
                      "isi": "Pelanggan dengan masa berlangganan singkat lebih rentan churn. "
                             "Berikan promo sambutan dan pendampingan awal.",
                      "ubah": None})
    if d["InternetService"] != "No" and d["TechSupport"] == "No":
        saran.append({"ikon": "🛠️", "judul": "Tawarkan dukungan teknis",
                      "isi": "Pelanggan internet tanpa dukungan teknis cenderung lebih mudah "
                             "berpindah penyedia.",
                      "ubah": {"TechSupport": "Yes"}})
    if d["PaymentMethod"] == "Electronic check":
        saran.append({"ikon": "💳", "judul": "Ajak beralih ke pembayaran otomatis",
                      "isi": "Pembayaran otomatis (transfer bank atau kartu kredit) membuat "
                             "pelanggan lebih konsisten.",
                      "ubah": {"PaymentMethod": "Bank transfer (automatic)"}})
    if d["MonthlyCharges"] >= 80:
        saran.append({"ikon": "💰", "judul": "Tinjau paket dan biaya",
                      "isi": "Biaya bulanan yang tinggi dapat memicu churn. Tawarkan paket yang "
                             "lebih hemat atau bundling (simulasi: diskon 10%).",
                      "ubah": {"MonthlyCharges": round(d["MonthlyCharges"] * 0.9, 2)}})
    if not saran:
        saran.append({"ikon": "🤝", "judul": "Pertahankan kualitas layanan",
                      "isi": "Tidak ada faktor risiko yang menonjol. Lanjutkan menjaga "
                             "kepuasan pelanggan.",
                      "ubah": None})
    return saran[:4]


def persen(x: float) -> str:
    return f"{x * 100:.1f}%"


# ======================================================================
# KOMPONEN HTML
# ======================================================================
def html_baris_probabilitas(peta: dict, label_teratas: str) -> str:
    baris = []
    for label, prob in sorted(peta.items(), key=lambda item: item[1], reverse=True):
        pct = max(0.0, min(100.0, prob * 100))
        kelas_top = " top" if label == label_teratas else ""
        lencana = '<em class="prob-badge">Prediksi</em>' if label == label_teratas else ""
        baris.append(
            f'<div class="prob-row{kelas_top}">'
            f'<div class="prob-head"><span class="prob-name">{label}{lencana}</span>'
            f'<span class="prob-pct">{pct:.1f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{pct:.2f}%"></div></div>'
            f'</div>'
        )
    return '<div class="prob-card">' + "".join(baris) + "</div>"


def html_chip(daftar: list[str], kelas: str) -> str:
    return "".join(f'<span class="chip {kelas}">{teks}</span>' for teks in daftar)


# ======================================================================
# HEADER & HERO
# ======================================================================
html(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">📡</div>
    <div>
      <div class="brand-name">CHURN<span>GUARD</span></div>
      <div class="brand-caption">CUSTOMER RETENTION AI</div>
    </div>
  </div>
  <div class="status-pill"><span class="status-dot"></span>{nama_model.upper()}</div>
</div>
""")

html(f"""
<div class="hero">
  <div class="hero-copy">
    <div class="eyebrow">✦ CUSTOMER CHURN PREDICTION</div>
    <h1>Kenali Pelanggan<br>yang Akan Pergi.</h1>
    <p>Perkirakan kemungkinan pelanggan berhenti berlangganan dari profil, kontrak,
    tagihan, dan layanan yang dipakai, lengkap dengan saran retensi untuk
    mempertahankannya.</p>
    <div class="hero-tags">
      <span>◆ {nama_model}</span>
      <span>◉ Probabilitas Churn</span>
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
      <div><div class="fc-title">AI Powered</div><div class="fc-sub">Smart Prediction</div></div>
    </div>
    <div class="floating-card card-bottom">
      <span class="mini-dot"></span>
      <div><div class="fc-title">{JUMLAH_PARAMETER} Parameter</div><div class="fc-sub">Profil &amp; Layanan</div></div>
    </div>
  </div>
</div>
""")

if model_error:
    st.error(
        f"Model gagal dimuat: {model_error}\n\n"
        "Pastikan file model_churn.pkl berada satu folder dengan app.py dan "
        "versi scikit-learn sama dengan saat model dilatih (lihat requirements.txt)."
    )
    st.stop()

# ======================================================================
# FORM INPUT
# ======================================================================
html("""
<div class="section-heading">
  <div>
    <div class="eyebrow">INPUT PARAMETERS</div>
    <h2>Data Pelanggan</h2>
    <p>Isi data pelanggan yang ingin dianalisis.</p>
  </div>
  <div class="step-label">STEP 01 / 02</div>
</div>
""")

with st.form("form_churn"):
    html('<div class="input-section-title">01 &nbsp; Profil Pelanggan</div>')
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        penanda_dua_kolom()
        gender = pilih("Jenis kelamin", GENDER)
    with c2:
        senior = pilih("Usia", LANSIA)
    with c3:
        partner = pilih("Punya pasangan", YA_TIDAK, index=1)
    with c4:
        dependents = pilih("Punya tanggungan", YA_TIDAK, index=1)

    html('<div class="input-section-title second-title">02 &nbsp; Kontrak dan Tagihan</div>')
    tenure = st.slider("Lama berlangganan (bulan)", 0, 72, 12)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        penanda_dua_kolom()
        monthly = st.number_input("Biaya bulanan (USD)", min_value=0.0, max_value=200.0,
                                  value=70.0, step=0.5)
    with c2:
        contract = pilih("Jenis kontrak", KONTRAK)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        paperless = pilih("Tagihan tanpa kertas", YA_TIDAK)
    with c2:
        payment = pilih("Metode pembayaran", PEMBAYARAN)

    html('<div class="input-section-title second-title">03 &nbsp; Layanan Dasar</div>')
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        phone = pilih("Layanan telepon", YA_TIDAK)
    with c2:
        multi = pilih("Jalur telepon ganda", TELEPON_GANDA)
    with c3:
        internet = pilih("Layanan internet", INTERNET, index=1)

    html('<div class="input-section-title second-title">04 &nbsp; Layanan Tambahan</div>')
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        penanda_dua_kolom()
        security = pilih("Keamanan online", TAMBAHAN)
    with c2:
        backup = pilih("Cadangan online", TAMBAHAN)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        penanda_dua_kolom()
        protection = pilih("Perlindungan perangkat", TAMBAHAN)
    with c2:
        tech = pilih("Dukungan teknis", TAMBAHAN)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        penanda_dua_kolom()
        tv = pilih("Streaming TV", TAMBAHAN)
    with c2:
        movies = pilih("Streaming film", TAMBAHAN)

    st.caption("Pilih “Tanpa internet” / “Tanpa telepon” hanya bila pelanggan memang tidak "
               "memakai layanan tersebut. Pilihan yang tidak cocok akan disesuaikan otomatis.")

    kirim = st.form_submit_button("📡  ANALISIS RISIKO CHURN", use_container_width=True)

# ======================================================================
# PROSES PREDIKSI
# ======================================================================
if kirim:
    mentah = {
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
        data_input, catatan = rapikan_input(mentah)
        df_input = buat_dataframe(data_input)
        prob = float(model.predict_proba(df_input)[0][1])
        info = tingkat_risiko(prob)

        saran = []
        if prob >= 0.35:
            for item in saran_retensi(data_input):
                prob_baru = None
                if item["ubah"]:
                    try:
                        hasil_sim = hitung_probabilitas({**data_input, **item["ubah"]})
                        if hasil_sim < prob - 0.005:
                            prob_baru = hasil_sim
                    except Exception:  # noqa: BLE001
                        prob_baru = None
                saran.append({**item, "prob_baru": prob_baru})

        risiko, aman = faktor_profil(data_input)

        st.session_state.hasil = {
            "prob": prob,
            "info": info,
            "data": data_input,
            "catatan": catatan,
            "jumlah_layanan": int(df_input["num_services"].iloc[0]),
            "saran": saran,
            "faktor_risiko": risiko,
            "faktor_aman": aman,
            "scroll": True,
        }
    except Exception as e:  # noqa: BLE001
        st.session_state.hasil = None
        st.error("Prediksi belum dapat dilakukan karena format input tidak sesuai "
                 "dengan model yang tersimpan.")
        st.code(str(e))
        st.warning("Pastikan versi scikit-learn sama dengan saat model dilatih "
                   "dan kolom input sama dengan yang dipakai di notebook.")

# ======================================================================
# HASIL
# ======================================================================
hasil = st.session_state.hasil

if hasil is not None:
    prob = hasil["prob"]
    info = hasil["info"]
    data = hasil["data"]

    html('<div id="hasil-prediksi"></div>')

    if hasil.get("scroll"):
        components.html(
            """
            <script>
            setTimeout(function() {
                const target = window.parent.document.getElementById("hasil-prediksi");
                if (target) { target.scrollIntoView({behavior: "smooth", block: "start"}); }
            }, 350);
            </script>
            """,
            height=0,
        )
        hasil["scroll"] = False  # gulir otomatis hanya sekali setelah tombol analisis ditekan

    html("""
    <div class="section-heading result-heading">
      <div>
        <div class="eyebrow">ANALYSIS COMPLETE</div>
        <h2>Hasil Prediksi</h2>
        <p>Berikut hasil analisis berdasarkan data pelanggan.</p>
      </div>
      <div class="step-label">STEP 02 / 02</div>
    </div>
    """)

    html(f"""
    <div class="result-card {info['kelas']}">
      <div class="result-left">
        <div class="result-icon">{info['ikon']}</div>
        <div class="result-label">HASIL PREDIKSI CHURN</div>
        <div class="result-name">{info['label']}</div>
        <p>{info['teks']}</p>
      </div>
      <div class="result-right">
        <div class="gauge" style="background:conic-gradient({info['warna']} {prob * 100:.1f}%, #e6e8f4 0)">
          <div class="gauge-inner">
            <span class="gauge-angka" style="color:{info['warna']}">{persen(prob)}</span>
            <span class="gauge-cap">PELUANG CHURN</span>
          </div>
        </div>
        <div class="result-right-caption">MODEL {nama_model.upper()}</div>
      </div>
    </div>
    """)

    st.button("↻ Analisis Pelanggan Lain", key="ulang_dekat_hasil",
              use_container_width=True, on_click=ulangi_analisis)

    # ---------- Keyakinan model ----------
    kelas_prediksi = "Churn" if prob >= 0.5 else "Tidak churn"
    keyakinan = max(prob, 1 - prob)

    html('<div class="subsection-title">Tingkat Keyakinan Model</div>')
    html(f"""
    <div class="conf-grid">
      <div class="confidence-card">
        <div class="confidence-top">
          <span>Keyakinan pada hasil prediksi</span>
          <strong>{persen(keyakinan)}</strong>
        </div>
        <div class="confidence-track"><div class="confidence-fill" style="width:{keyakinan * 100:.2f}%"></div></div>
        <div class="confidence-foot">Nilai probabilitas yang dihasilkan model {nama_model}.</div>
      </div>
      <div class="metric-card">
        <div class="metric-icon">✦</div>
        <div class="metric-label">PELUANG BERTAHAN</div>
        <div class="metric-value">{persen(1 - prob)}</div>
        <div class="metric-desc">Probabilitas tidak churn</div>
      </div>
    </div>
    """)

    html('<div class="subsection-title small-subsection">Probabilitas Setiap Kelas</div>')
    html(html_baris_probabilitas(
        {"Churn (berhenti)": prob, "Tidak churn (bertahan)": 1 - prob},
        "Churn (berhenti)" if kelas_prediksi == "Churn" else "Tidak churn (bertahan)",
    ))
    st.caption("Catatan: probabilitas model bukan jaminan mutlak bahwa prediksi benar. "
               "Hasil dipengaruhi kualitas data pelatihan.")

    # ---------- Faktor profil ----------
    if hasil["faktor_risiko"] or hasil["faktor_aman"]:
        html('<div class="subsection-title">Faktor yang Menonjol pada Profil</div>')
        bagian = ""
        if hasil["faktor_risiko"]:
            bagian += ('<div class="faktor-grup"><div class="faktor-judul">PENDORONG RISIKO</div>'
                       f'<div class="chip-wrap">{html_chip(hasil["faktor_risiko"], "risiko")}</div></div>')
        if hasil["faktor_aman"]:
            bagian += ('<div class="faktor-grup"><div class="faktor-judul">PENAHAN CHURN</div>'
                       f'<div class="chip-wrap">{html_chip(hasil["faktor_aman"], "aman")}</div></div>')
        html(f'<div class="faktor-card">{bagian}</div>')
        st.caption("Indikator ini berdasarkan aturan sederhana dari pola umum data Telco, "
                   "bukan penjelasan langsung dari model.")

    # ---------- Saran retensi ----------
    html('<div class="subsection-title">💡 Saran Retensi</div>')
    if hasil["saran"]:
        for item in hasil["saran"]:
            delta = ""
            if item.get("prob_baru") is not None:
                delta = (f'<div class="saran-delta">Simulasi: risiko {persen(prob)} → '
                         f'{persen(item["prob_baru"])}</div>')
            html(f"""
            <div class="saran">
              <div class="saran-ikon">{item['ikon']}</div>
              <div class="saran-isi"><b>{item['judul']}</b><p>{item['isi']}</p>{delta}</div>
            </div>
            """)
        if any(s.get("prob_baru") is not None for s in hasil["saran"]):
            st.caption("Simulasi memakai model yang sama dengan mengubah satu faktor saja. "
                       "Sifatnya perkiraan, bukan jaminan hasil.")
    else:
        html("""
        <div class="saran saran-baik">
          <div class="saran-ikon">✅</div>
          <div class="saran-isi"><b>Pelanggan relatif aman</b>
          <p>Tidak ada indikasi risiko yang kuat. Tetap jaga kualitas layanan.</p></div>
        </div>
        """)

    # ---------- Ringkasan data ----------
    total_bayar = data["MonthlyCharges"] * data["tenure"]
    html('<div class="subsection-title">Ringkasan Data yang Dianalisis</div>')
    html(f"""
    <div class="summary-grid">
      <div class="summary-card"><div class="summary-label">LAMA LANGGANAN</div><div class="summary-value">{data['tenure']} bulan</div></div>
      <div class="summary-card"><div class="summary-label">BIAYA BULANAN</div><div class="summary-value">${data['MonthlyCharges']:,.2f}</div></div>
      <div class="summary-card"><div class="summary-label">PERKIRAAN TOTAL</div><div class="summary-value">${total_bayar:,.0f}</div></div>
      <div class="summary-card"><div class="summary-label">LAYANAN AKTIF</div><div class="summary-value">{hasil['jumlah_layanan']} layanan</div></div>
    </div>
    <div class="detail-card">
      <div><span>Jenis kontrak</span><strong>{KONTRAK[data['Contract']]}</strong></div>
      <div><span>Layanan internet</span><strong>{INTERNET[data['InternetService']]}</strong></div>
      <div><span>Metode pembayaran</span><strong>{PEMBAYARAN[data['PaymentMethod']]}</strong></div>
    </div>
    """)

    if hasil["catatan"]:
        daftar = "<br>".join(f"• {c}" for c in hasil["catatan"])
        html(f'<div class="catatan-input"><strong>Penyesuaian input:</strong><br>{daftar}</div>')

    html("""
    <div class="disclaimer">
      <strong>Informasi:</strong> Hasil ini merupakan prediksi berdasarkan model machine learning.
      Total tagihan diperkirakan dari biaya bulanan × lama berlangganan. Hanya untuk keperluan
      pembelajaran dan bukan jaminan perilaku pelanggan sebenarnya.
    </div>
    """)

    st.button("↻ Analisis Pelanggan Lain", key="ulang_bawah",
              use_container_width=True, on_click=ulangi_analisis)

# ======================================================================
# TENTANG PROYEK
# ======================================================================
with st.expander("ℹ️  Tentang Proyek & Model"):
    html("""
    <div class="kartu">
      <h3>Tentang Proyek</h3>
      <p>Proyek ini memprediksi apakah pelanggan layanan telekomunikasi akan <b>berhenti
      berlangganan (churn)</b>. Pengerjaan mengikuti tahapan <b>CRISP-DM</b>.</p>
    </div>
    <div class="kartu">
      <h3>Tahapan CRISP-DM</h3>
      <div class="langkah"><b>1</b><span>Business Understanding: menentukan masalah dan tujuan bisnis.</span></div>
      <div class="langkah"><b>2</b><span>Data Understanding: eksplorasi data (EDA) dan pengecekan kualitas data.</span></div>
      <div class="langkah"><b>3</b><span>Data Preparation: penanganan missing value, outlier, imbalance, dan feature engineering.</span></div>
      <div class="langkah"><b>4</b><span>Modeling: Logistic Regression, Decision Tree, dan Random Forest dengan GridSearchCV.</span></div>
      <div class="langkah"><b>5</b><span>Evaluation: perbandingan recall, F1-score, dan ROC-AUC untuk memilih model terbaik.</span></div>
      <div class="langkah"><b>6</b><span>Deployment: aplikasi web sederhana menggunakan Streamlit.</span></div>
    </div>
    """)
    html(f"""
    <div class="kartu">
      <h3>Model yang Digunakan: {nama_model}</h3>
      <p>Dipilih dari perbandingan tiga algoritma. Metrik utama adalah <b>recall</b> dan
      <b>F1-score</b> pada kelas churn, karena pelanggan churn yang tidak terdeteksi lebih
      merugikan daripada salah memprediksi pelanggan yang setia.</p>
    </div>
    """)
    metrik = METRIK_MODEL.get(nama_model)
    if metrik:
        kartu = "".join(
            f'<div class="summary-card"><div class="summary-label">{nama.upper()}</div>'
            f'<div class="summary-value">{nilai:.3f}</div></div>'
            for nama, nilai in metrik.items()
        )
        html(f'<div class="summary-grid lima">{kartu}</div>')
        st.caption("Nilai evaluasi pada data uji (dari churn.ipynb).")

# ======================================================================
# FOOTER
# ======================================================================
html("""
<div class="footer">
  <div class="footer-brand">📡 CHURNGUARD</div>
  <div>CUSTOMER RETENTION SYSTEM</div>
  <div class="footer-note">Aplikasi prediksi churn pelanggan berbasis Machine Learning</div>
  <div class="footer-note">Proyek Akhir Machine Learning · Dataset Telco Customer Churn</div>
</div>
""")
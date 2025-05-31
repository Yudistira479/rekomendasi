import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Fungsi untuk set background image
def set_bg_image():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        /* Tambahkan sedikit transparansi ke panel konten agar teks jelas */
        .css-18e3th9 {{
            background-color: rgba(255, 255, 255, 0.85) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image()

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("novels.csv")
    return df

novels = load_data()

# Label Encoding
le_genre = LabelEncoder()
le_title = LabelEncoder()
le_author = LabelEncoder()
le_status = LabelEncoder()

novels['genre_encoded'] = le_genre.fit_transform(novels['genres'])
novels['title_encoded'] = le_title.fit_transform(novels['title'])
novels['author_encoded'] = le_author.fit_transform(novels['authors'])
novels['status_encoded'] = le_status.fit_transform(novels['status'])

# Model untuk prediksi scored (regresi)
X_scored = novels[['genre_encoded', 'author_encoded', 'status_encoded']]
y_scored = novels['scored']
model_scored = RandomForestRegressor(random_state=42)
model_scored.fit(X_scored, y_scored)

# Model untuk prediksi genre (klasifikasi)
X_genre = novels[['scored', 'author_encoded', 'status_encoded']]
y_genre = novels['genre_encoded']
model_genre = RandomForestClassifier(random_state=42)
model_genre.fit(X_genre, y_genre)

# Inisialisasi halaman
st.set_page_config(page_title="Novel Recommendation App", layout="wide")
page = st.sidebar.selectbox("Navigasi", ["Home", "Rekomendasi Berdasarkan Scored", "Rekomendasi Berdasarkan Genre"])

# Riwayat rekomendasi
if "history" not in st.session_state:
    st.session_state.history = []

# HOME PAGE
if page == "Home":
    st.title("📚 Beranda")

    st.subheader("10 Novel Paling Populer")
    top_popular = novels.sort_values(by="popularty", ascending=False).head(10)
    st.dataframe(top_popular[['title', 'authors', 'genres', 'scored', 'popularty']])

    st.subheader("Riwayat Rekomendasi")
    if st.session_state.history:
        for entry in st.session_state.history:
            st.markdown(f"**{entry['title']}** - {entry['type']}:")
            st.dataframe(entry['results'])
    else:
        st.write("Belum ada riwayat rekomendasi.")

# PAGE 2 - SCORED
elif page == "Rekomendasi Berdasarkan Scored":
    st.title("⭐ Rekomendasi Berdasarkan Scored")

    # Input scored manual dengan slider atau number input
    input_scored = st.number_input("Masukkan nilai scored (contoh: 4.5)", min_value=0.0, max_value=10.0, value=5.0, step=0.1, format="%.2f")

    # Buat prediksi menggunakan input scored ini
    # Karena model_scored memerlukan genre_encoded, author_encoded, status_encoded,
    # kita perlu memberikan input default atau tanya user untuk input juga.
    # Sebagai solusi sederhana: ambil median nilai dari dataset
    median_genre = int(novels['genre_encoded'].median())
    median_author = int(novels['author_encoded'].median())
    median_status = int(novels['status_encoded'].median())

    X_input = pd.DataFrame({
        'genre_encoded': [median_genre],
        'author_encoded': [median_author],
        'status_encoded': [median_status]
    })

    # Prediksi scored berdasarkan median fitur (meskipun sebenarnya model prediksi scored dari fitur, bukan dari scored langsung)
    y_pred = model_scored.predict(X_input)[0]

    # Untuk rekomendasi kita cari novel yang scored-nya paling dekat dengan input user
    novels['score_diff'] = (novels['scored'] - input_scored).abs()
    result = novels.sort_values(by='score_diff').head(10)

    st.write(f"Rekomendasi novel dengan scored mendekati {input_scored}:")
    st.dataframe(result[['title', 'authors', 'genres', 'scored', 'popularty']])

    st.session_state.history.append({
        "title": f"Scored {input_scored}",
        "type": "Scored",
        "results": result[['title', 'authors', 'genres', 'scored', 'popularty']]
    })

    novels.drop(columns=['score_diff'], inplace=True)

# PAGE 3 - GENRE
elif page == "Rekomendasi Berdasarkan Genre":
    st.title("🎯 Rekomendasi Berdasarkan Genre")

    # Input judul novel dengan input text dan scrollable selectbox sekaligus
    # Karena user mau input sendiri, kita pakai text_input tapi kita kasih autocomplete (pakai st.text_input + list)
    # Streamlit belum punya autocomplete native, jadi kita buat text_input saja
    input_title = st.text_input("Masukkan judul novel (case sensitive, contoh: The Great Adventure)")

    if input_title:
        if input_title in novels['title'].values:
            selected_row = novels[novels['title'] == input_title].iloc[0]
            X_input = pd.DataFrame({
                'scored': [selected_row['scored']],
                'author_encoded': [selected_row['author_encoded']],
                'status_encoded': [selected_row['status_encoded']]
            })

            y_pred = model_genre.predict(X_input)[0]
            genre_name = le_genre.inverse_transform([y_pred])[0]
            result = novels[novels['genres'] == genre_name].sort_values(by='scored', ascending=False).head(10)

            st.write(f"Rekomendasi novel berdasarkan genre dari \"{input_title}\" (Genre: {genre_name}):")
            st.dataframe(result[['title', 'authors', 'genres', 'scored', 'popularty']])

            st.session_state.history.append({
                "title": input_title,
                "type": "Genre",
                "results": result[['title', 'authors', 'genres', 'scored', 'popularty']]
            })
        else:
            st.warning("Judul novel tidak ditemukan. Mohon cek kembali penulisan.")


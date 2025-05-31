import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# WAJIB: set_page_config harus di paling atas sebelum komponen lainnya
st.set_page_config(page_title="📖 Novel Recommendation App", layout="wide")

# 🚚 Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("novels.csv")
    return df

novels = load_data()

# 🔢 Label Encoding
le_genre = LabelEncoder()
le_author = LabelEncoder()
le_status = LabelEncoder()

novels['genre_encoded'] = le_genre.fit_transform(novels['genres'])
novels['author_encoded'] = le_author.fit_transform(novels['authors'])
novels['status_encoded'] = le_status.fit_transform(novels['status'])

# 🤖 Model untuk prediksi scored (regresi)
X_scored = novels[['genre_encoded', 'author_encoded', 'status_encoded']]
y_scored = novels['scored']
model_scored = RandomForestRegressor(random_state=42)
model_scored.fit(X_scored, y_scored)

# 🤖 Model untuk prediksi genre (klasifikasi)
X_genre = novels[['scored', 'author_encoded', 'status_encoded']]
y_genre = novels['genre_encoded']
model_genre = RandomForestClassifier(random_state=42)
model_genre.fit(X_genre, y_genre)

# 🧭 Sidebar Navigasi
page = st.sidebar.selectbox("📌 Navigasi", ["📚 Beranda", "🎯 Rekomendasi Berdasarkan Genre", "⭐ Rekomendasi Berdasarkan Scored"])

# 🗃️ Inisialisasi riwayat rekomendasi
if "history" not in st.session_state:
    st.session_state.history = []

# 🏠 Halaman Beranda
if page == "📚 Beranda":
    st.title("📚 Novel Recommendation App")

    st.subheader("🔥 10 Novel Paling Populer")
    top_popular = novels.sort_values(by="popularty", ascending=False).head(10)
    st.dataframe(top_popular[['title', 'authors', 'genres', 'scored', 'popularty']])

    st.subheader("📜 Riwayat Rekomendasi")
    if st.session_state.history:
        for entry in st.session_state.history:
            st.markdown(f"**{entry['title']}** - {entry['type']}:")
            st.dataframe(entry['results'])
    else:
        st.write("Belum ada riwayat rekomendasi.")

# 🎯 Halaman Genre (input judul manual)
elif page == "🎯 Rekomendasi Berdasarkan Genre":
    st.title("🎯 Rekomendasi Berdasarkan Genre")

    st.markdown("Masukkan judul novel yang kamu sukai:")
    input_title = st.text_input("Judul Novel")

    if input_title:
        match = novels[novels['title'].str.lower() == input_title.lower()]
        if not match.empty:
            selected_row = match.iloc[0]
            X_input = pd.DataFrame({
                'scored': [selected_row['scored']],
                'author_encoded': [selected_row['author_encoded']],
                'status_encoded': [selected_row['status_encoded']]
            })

            y_pred = model_genre.predict(X_input)[0]
            genre_name = le_genre.inverse_transform([y_pred])[0]
            result = novels[novels['genres'] == genre_name].sort_values(by='scored', ascending=False).head(10)

            st.success(f"Genre dari \"{input_title}\" diprediksi: **{genre_name}**")
            st.dataframe(result[['title', 'authors', 'genres', 'scored', 'popularty']])

            st.session_state.history.append({
                "title": input_title,
                "type": "Genre",
                "results": result[['title', 'authors', 'genres', 'scored', 'popularty']]
            })
        else:
            st.error("Judul tidak ditemukan dalam database.")

# ⭐ Halaman Scored (input skor manual)
elif page == "⭐ Rekomendasi Berdasarkan Scored":
    st.title("⭐ Rekomendasi Berdasarkan Skor (Scored)")

    input_score = st.number_input("Masukkan skor (contoh: 7.5)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    input_genre = st.selectbox("Pilih genre", novels['genres'].unique())
    input_author = st.selectbox("Pilih author", novels['authors'].unique())
    input_status = st.selectbox("Pilih status", novels['status'].unique())

    if st.button("🔍 Cari Rekomendasi"):
        genre_encoded = le_genre.transform([input_genre])[0]
        author_encoded = le_author.transform([input_author])[0]
        status_encoded = le_status.transform([input_status])[0]

        input_df = pd.DataFrame({
            'genre_encoded': [genre_encoded],
            'author_encoded': [author_encoded],
            'status_encoded': [status_encoded]
        })

        y_pred = model_scored.predict(input_df)[0]

        novels['score_diff'] = (novels['scored'] - y_pred).abs()
        result = novels.sort_values(by='score_diff').head(10)

        st.success(f"Menampilkan rekomendasi dengan skor mendekati: **{y_pred:.2f}**")
        st.dataframe(result[['title', 'authors', 'genres', 'scored', 'popularty']])

        st.session_state.history.append({
            "title": f"Input Score: {input_score}",
            "type": "Scored",
            "results": result[['title', 'authors', 'genres', 'scored', 'popularty']]
        })

        novels.drop(columns=['score_diff'], inplace=True)

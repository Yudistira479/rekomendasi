import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

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

# PAGE 2 - SCORED (input manual)
elif page == "Rekomendasi Berdasarkan Scored":
    st.title("⭐ Rekomendasi Berdasarkan Scored")
    scored_input = st.number_input("Masukkan nilai scored (rating) antara 0 sampai 10:", min_value=0.0, max_value=10.0, step=0.1, format="%.1f")

    # Cari 10 novel dengan scored terdekat ke nilai input
    novels['score_diff'] = (novels['scored'] - scored_input).abs()
    result = novels.sort_values(by='score_diff').head(10)

    st.write(f"Rekomendasi novel dengan scored paling dekat dengan {scored_input}:")
    st.dataframe(result[['title', 'authors', 'genres', 'scored', 'popularty']])

    st.session_state.history.append({
        "title": f"Input Scored {scored_input}",
        "type": "Scored",
        "results": result[['title', 'authors', 'genres', 'scored', 'popularty']]
    })

    novels.drop(columns=['score_diff'], inplace=True)

# PAGE 3 - GENRE (input judul scroll)
elif page == "Rekomendasi Berdasarkan Genre":
    st.title("🎯 Rekomendasi Berdasarkan Genre")
    # Input judul novel pakai text_input dengan auto scroll (list suggestion)
    selected_title = st.text_input("Masukkan judul novel:", key="genre_input")

    if selected_title and selected_title in novels['title'].values:
        selected_row = novels[novels['title'] == selected_title].iloc[0]
        X_input = pd.DataFrame({
            'scored': [selected_row['scored']],
            'author_encoded': [selected_row['author_encoded']],
            'status_encoded': [selected_row['status_encoded']]
        })

        y_pred = model_genre.predict(X_input)[0]
        genre_name = le_genre.inverse_transform([y_pred])[0]
        result = novels[novels['genres'] == genre_name].sort_values(by='scored', ascending=False).head(10)

        st.write(f"Rekomendasi novel berdasarkan genre dari \"{selected_title}\" (Genre: {genre_name}):")
        st.dataframe(result[['title', 'authors', 'genres', 'scored', 'popularty']])

        st.session_state.history.append({
            "title": selected_title,
            "type": "Genre",
            "results": result[['title', 'authors', 'genres', 'scored', 'popularty']]
        })
    elif selected_title:
        st.warning("Judul novel tidak ditemukan, coba masukkan judul yang benar.")


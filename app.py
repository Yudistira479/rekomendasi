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

# PAGE 2 - SCORED
elif page == "Rekomendasi Berdasarkan Scored":
    st.title("⭐ Rekomendasi Berdasarkan Scored")
    selected_title = st.selectbox("Pilih judul novel", novels['title'].unique())

    selected_row = novels[novels['title'] == selected_title].iloc[0]
    X_input = pd.DataFrame({
        'genre_encoded': [selected_row['genre_encoded']],
        'author_encoded': [selected_row['author_encoded']],
        'status_encoded': [selected_row['status_encoded']]
    })

    y_pred = model_scored.predict(X_input)[0]

    # Cari 10 novel dengan scored terdekat ke hasil prediksi
    novels['score_diff'] = (novels['scored'] - y_pred).abs()
    result = novels.sort_values(by='score_diff').head(10)

    st.write(f"Rekomendasi novel berdasarkan scored dari \"{selected_title}\" (Prediksi: {y_pred:.2f}):")
    st.dataframe(result[['title', 'authors', 'genres', 'scored', 'popularty']])

    st.session_state.history.append({
        "title": selected_title,
        "type": "Scored",
        "results": result[['title', 'authors', 'genres', 'scored', 'popularty']]
    })

    # Hapus kolom sementara agar tidak mengganggu operasi selanjutnya
    novels.drop(columns=['score_diff'], inplace=True)

# PAGE 3 - GENRE
elif page == "Rekomendasi Berdasarkan Genre":
    st.title("🎯 Rekomendasi Berdasarkan Genre")
    selected_title = st.selectbox("Pilih judul novel", novels['title'].unique(), key='genre')

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

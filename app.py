import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Konfigurasi halaman - HARUS DI PALING ATAS
st.set_page_config(page_title="Novel Recommendation App", layout="wide")

# Load dataset
@st.cache_data(ttl=600)
def load_data():
    df = pd.read_csv("novels1.csv")
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

# Inisialisasi halaman dan navigasi
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
            st.dataframe(entry['results'].style.format({'scored': '{:.2f}'}))
    else:
        st.write("Belum ada riwayat rekomendasi.")

    if st.button("🔄 Reset Riwayat"):
        st.session_state.history = []
        st.success("Riwayat rekomendasi telah direset.")

# PAGE 2 - SCORED (input manual)
elif page == "Rekomendasi Berdasarkan Scored":
    st.title("⭐ Rekomendasi Berdasarkan Scored")
    scored_input = st.number_input("Masukkan nilai scored (rating) antara 0 sampai 10:", min_value=0.0, max_value=10.0, step=0.1, format="%.1f")

    if scored_input > 0.0:
        temp_novels = novels.copy()
        temp_novels['score_diff'] = (temp_novels['scored'] - scored_input).abs()
        result = temp_novels.sort_values(by='score_diff').head(10)

        st.write(f"Rekomendasi novel dengan scored paling dekat dengan {scored_input}:")
        st.dataframe(result[['title', 'authors', 'genres', 'scored', 'popularty']].style.format({'scored': '{:.2f}'}))

        st.session_state.history.append({
            "title": f"Input Scored {scored_input}",
            "type": "Scored",
            "results": result[['title', 'authors', 'genres', 'scored', 'popularty']]
        })

# PAGE 3 - GENRE (input judul scroll)
elif page == "Rekomendasi Berdasarkan Genre":
    st.title("🎯 Rekomendasi Berdasarkan Genre")
    selected_title = st.selectbox("Pilih judul novel:", sorted(novels['title'].unique()))

    if selected_title:
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
        st.dataframe(result[['title', 'authors', 'genres', 'scored', 'popularty']].style.format({'scored': '{:.2f}'}))

        st.session_state.history.append({
            "title": selected_title,
            "type": "Genre",
            "results": result[['title', 'authors', 'genres', 'scored', 'popularty']]
        })

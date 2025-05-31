import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

# Model untuk prediksi berdasarkan rating
X_rating = novels[['genre_encoded', 'author_encoded', 'status_encoded']]
y_rating = novels['scored']
model_rating = RandomForestClassifier()
model_rating.fit(X_rating, y_rating)

# Model untuk prediksi berdasarkan genre
X_genre = novels[['scored', 'author_encoded', 'status_encoded']]
y_genre = novels['genre_encoded']
model_genre = RandomForestClassifier()
model_genre.fit(X_genre, y_genre)

# Inisialisasi halaman
st.set_page_config(page_title="Novel Recommendation App", layout="wide")
page = st.sidebar.selectbox("Navigasi", ["Home", "Rekomendasi Berdasarkan Rating", "Rekomendasi Berdasarkan Genre"])

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

# PAGE 2 - RATING
elif page == "Rekomendasi Berdasarkan Rating":
    st.title("⭐ Rekomendasi Berdasarkan Rating")
    selected_title = st.selectbox("Pilih judul novel", novels['title'].unique())

    selected_row = novels[novels['title'] == selected_title].iloc[0]
    X_input = pd.DataFrame({
        'genre_encoded': [selected_row['genre_encoded']],
        'author_encoded': [selected_row['author_encoded']],
        'status_encoded': [selected_row['status_encoded']]
    })

    y_pred = model_rating.predict(X_input)[0]
    result = novels[novels['scored'] == y_pred].sort_values(by='popularty', ascending=False).head(10)

    st.write(f"Rekomendasi novel berdasarkan rating dari \"{selected_title}\":")
    st.dataframe(result[['title', 'authors', 'genres', 'scored', 'popularty']])

    st.session_state.history.append({"title": selected_title, "type": "Rating", "results": result[['title', 'authors', 'genres', 'scored', 'popularty']]})

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

    st.session_state.history.append({"title": selected_title, "type": "Genre", "results": result[['title', 'authors', 'genres', 'scored', 'popularty']]})

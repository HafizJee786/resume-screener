import re
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from app.utils import preprocess, extract_skills


def train_model():
    print("Loading cleaned dataset...")
    df = pd.read_csv('data/cleaned_resumes.csv')

  # Drop rows where cleaned_resume is empty or NaN
    df = df.dropna(subset=['cleaned_resume'])
    df = df[df['cleaned_resume'].str.strip() != '']
    df = df.reset_index(drop=True)

    X = df['cleaned_resume']
    y = df['Category']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X_vectorized = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y_encoded, test_size=0.2, random_state=42
    )

    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    with open('models/resume_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("\n✅ Model saved to models/")
    return model, tfidf, le


def get_match_score(resume_text, job_description):
    # Use raw text for skill matching (no preprocessing)
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    # Extract all words from JD
    jd_words = set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#]*\b', jd_lower))
    
    # Remove common stop words
    stop = {'we', 'are', 'the', 'a', 'an', 'and', 'or', 'for', 
            'with', 'of', 'in', 'is', 'to', 'at', 'be', 'as',
            'have', 'has', 'its', 'from', 'that', 'this', 'must',
            'will', 'can', 'should', 'their', 'our', 'your'}
    jd_words = jd_words - stop

    # Check how many JD words exist in resume
    matched = [w for w in jd_words if w in resume_lower]
    
    keyword_score = len(matched) / len(jd_words) if jd_words else 0

    # TF-IDF cosine similarity
    tfidf_temp = TfidfVectorizer()
    vectors = tfidf_temp.fit_transform([resume_lower, jd_lower])
    cosine_score = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Combined score
    final_score = (keyword_score * 0.65) + (cosine_score * 0.35)
    return round(float(final_score) * 100, 2)


def predict_category(resume_text):
    with open('models/resume_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    processed = preprocess(resume_text)
    vectorized = tfidf.transform([processed])
    prediction = model.predict(vectorized)
    category = le.inverse_transform(prediction)[0]
    return category


if __name__ == "__main__":
    train_model()

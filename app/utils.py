import re
import nltk
import spacy

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean and preprocess resume text
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove phone numbers
    text = re.sub(r'\+?\d[\d\-\s]{7,}\d', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def lemmatize_text(text):
    """
    Lemmatize text using spaCy
    """
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.text not in STOPWORDS
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 2
    ]
    return " ".join(tokens)


def preprocess(text):
    """
    Full preprocessing pipeline:
    clean → lemmatize
    """
    cleaned = clean_text(text)
    lemmatized = lemmatize_text(cleaned)
    return lemmatized


def extract_skills(text):
    """
    Extract common tech skills from resume text
    """
    SKILLS_LIST = [
        'python', 'java', 'sql', 'machine learning', 'deep learning',
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
        'javascript', 'react', 'nodejs', 'flask', 'fastapi', 'django',
        'html', 'css', 'git', 'docker', 'kubernetes', 'aws', 'azure',
        'nlp', 'computer vision', 'data analysis', 'tableau', 'power bi',
        'excel', 'r', 'scala', 'spark', 'hadoop', 'mongodb', 'postgresql',
        'c++', 'c#', 'php', 'swift', 'kotlin', 'linux', 'bash'
    ]

    text_lower = text.lower()
    found_skills = [skill for skill in SKILLS_LIST if skill in text_lower]
    return found_skills
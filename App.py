from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ─── 1. DECLARE YOUR ARTICLES ───────────────────────────────────────────────────
articles = [
    {"article": "Article 1",   "description": "Name and territory of the Union."},
    {"article": "Article 2",   "description": "Admission or establishment of new States."},
    {"article": "Article 12",  "description": "Definition of 'State' for Fundamental Rights."},
    {"article": "Article 13",  "description": "Laws inconsistent with Fundamental Rights are void."},
    {"article": "Article 14",  "description": "Equality before the law and equal protection of laws."},
    {"article": "Article 15",  "description": "Prohibits discrimination on grounds of religion, race, caste, sex or birth."},
    {"article": "Article 17",  "description": "Abolition of untouchability and prohibition of its practice."},
    {"article": "Article 19",  "description": "Freedom of speech, expression, assembly, movement, residence, profession."},
    {"article": "Article 21",  "description": "Protection of life and personal liberty; includes dignity, privacy, bodily integrity."},
    {"article": "Article 22",  "description": "Protection against arrest and detention; right to know grounds and to legal counsel."},
    {"article": "Article 23",  "description": "Prohibition of trafficking in human beings and forced labour."},
    {"article": "Article 24",  "description": "Prohibition of employment of children below age 14 in hazardous work."},
    {"article": "Article 25",  "description": "Freedom of conscience and free profession, practice and propagation of religion."},
    {"article": "Article 368","description": "Procedure for amendment of the Constitution."},
    {"article": "Article 370","description": "Special provisions for the State of Jammu & Kashmir (now abrogated)."},
    # Add more articles here...
]

# ─── 2. VECTORIZE DESCRIPTIONS ───────────────────────────────────────────────────
descs = [a["description"] for a in articles]
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(descs)

# ─── 3. MATCH FUNCTION ──────────────────────────────────────────────────────────
def find_top_articles(text, top_k=5):
    vec = vectorizer.transform([text])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    top_idxs = sims.argsort()[::-1][:top_k]
    return [(articles[i]["article"], articles[i]["description"], sims[i]) for i in top_idxs]

# ─── 4. FLASK ROUTES ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    query = ""
    if request.method == "POST":
        query = request.form.get("problem", "")
        if query:
            results = find_top_articles(query, top_k=5)
    return render_template("index.html", results=results, query=query)

# ─── 5. RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)

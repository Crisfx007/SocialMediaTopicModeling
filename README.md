# SocialMediaTopicModeling

Topic modeling on social media data using **BERTopic**, with three approaches — basic, intermediate, and advanced (NER-enhanced) — to extract meaningful topics such as events, trends, and user feedback from tweets/posts.

---

## Aim

This repository provides three Jupyter notebooks for topic modeling on social media data (tweets/posts) using BERTopic. The notebooks vary in complexity, from basic to advanced, incorporating Named Entity Recognition (NER) for improved topic extraction.

---

## Notebooks

### 1. `TopicModelingBasic.ipynb` (Basic)

- Quick topic modeling on a small dataset (~163 posts).
- Basic preprocessing: lowercase conversion, removing special characters and stopwords.
- Uses `paraphrase-MiniLM-L6-v2` embedding model, default UMAP/vectorizer settings.
- Parameters: 5 topics, `min_cluster_size=3`.
- Outputs: `tweet_topics_output.csv`, `topics_output.txt`.
- **Best for:** Small-scale, quick analysis.

---

### 2. `TopicModelingIntermediate.ipynb` (Intermediate)

- Robust topic modeling on a larger dataset (~3244 posts).
- Comprehensive preprocessing: removes URLs, emojis, mentions, special characters, stopwords.
- Uses `paraphrase-MiniLM-L6-v2` embedding model, UMAP with `n_neighbors=5`.
- Incorporates bigrams, 10 topics, `min_cluster_size=10`.
- Outputs: `topic_output.csv`, `topics_output.txt`.
- **Best for:** General social media analysis with moderate complexity.

---

### 3. `TopicModelingusingNERAdvanced.ipynb` (Advanced, Recommended)

- Advanced topic modeling with Named Entity Recognition (NER) on a large dataset (~3244 posts).
- Robust preprocessing plus spaCy NER to emphasize entities like names, locations.
- Uses `paraphrase-xlm-r-multilingual-v1` embedding model, UMAP with `n_neighbors=10`.
- Incorporates bigrams, 10 topics, and term filtering (`score_threshold=0.05`).
- Outputs: `Jsonentities_output.csv`, `olaJSonwith_entities.txt`.
- **Best for:** Precise, entity-focused topic extraction, especially for multilingual data.

---

## Recommendation

- Use **`TopicModelingusingNER.ipynb`** for the most coherent, entity-focused topics enhanced by NER and multilingual support.
- Use **`refined.ipynb`** for simpler English-only analysis.
- Use **`topic.ipynb`** for quick tests on small datasets.

---

## Dependencies

```bash
pip install pandas nltk spacy bertopic sentence-transformers umap-learn scikit-learn
python -m spacy download en_core_web_sm

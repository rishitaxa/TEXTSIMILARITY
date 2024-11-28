import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
with open('file1.txt', 'r', encoding='utf-8') as file1:
    sample1 = file1.read()
with open('file2.txt', 'r', encoding='utf-8') as file2:
    sample2 = file2.read()
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([sample1, sample2])
similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
similarity_score = similarity_matrix[0][0]
print(f"Cosine similarity between the two documents: {similarity_score*100} %")
plt.bar(['Cosine Similarity'], [similarity_score], color='pink')
plt.ylim(0, 1)
plt.title('Cosine Similarity Between Documents')
plt.ylabel('Similarity Score')
plt.show()


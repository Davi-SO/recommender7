import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Dados fictícios para ilustração
data = {
    'post_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'texto': [
        "Amo passear ao ar livre!",
        "Dicas de programação em Python",
        "Como cuidar de plantas de interior",
        "Tutorial de Django para iniciantes",
        "Fotografia de paisagens naturais",
        "Banco de dados",
        "Projetos para iniciantes em JavaScript",
        "Algoritmos em Java"
    ],
    'tags': [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    ]
}

# Criando o DataFrame
posts_df = pd.DataFrame(data)

# Inicializa o vetorizador TF-IDF
vectorizer = TfidfVectorizer()

# Vetoriza o texto dos posts
tfidf_matrix = vectorizer.fit_transform(posts_df['texto'])

# Adiciona a matriz TF-IDF ao DataFrame para fácil acesso
posts_df['tfidf'] = list(tfidf_matrix.toarray())
def recomendar_posts(usuario_likes):
    # Concatena os vetores TF-IDF dos posts curtidos pelo usuário para criar o perfil do usuário
    user_vector = np.sum(tfidf_matrix[np.array(usuario_likes)], axis=0)

    # Calcula a similaridade do vetor do usuário com todos os posts
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)

    # Recupera os índices dos posts com maior pontuação de similaridade
    recommended_post_indices = np.argsort(similarity_scores[0])[:][:]
    
    # Retorna os posts recomendados
    return posts_df.iloc[recommended_post_indices]['texto']

# Exemplo de uso: o usuário gostou dos posts com ID 2 e 4
usuario_likes = [1, 3]
recomendacoes = recomendar_posts(usuario_likes)
print("Posts recomendados para o usuário:")
print(recomendacoes)

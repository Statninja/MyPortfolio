import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain

# Knowledge Base, Memory, and Feedback Structures
knowledge_base = []
short_term_memory = []
feedback_scores = {}

# Initialize TF-IDF Vectorizer for better search
vectorizer = TfidfVectorizer()

# Adding Knowledge for Initial Training with the "AI Book"
def add_knowledge(text):
    chunks = split_text(text)
    knowledge_base.extend(chunks)
    update_vectors()

# Splitting large text for efficient searching
def split_text(text, chunk_size=200):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Vectorizing the Knowledge Base
def update_vectors():
    global knowledge_vectors
    knowledge_vectors = vectorizer.fit_transform(knowledge_base)

# Retrieval with Advanced Cosine Similarity for Relevance
def retrieve_relevant_texts(question, top_k=3):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(knowledge_vectors, question_vector).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [knowledge_base[i] for i in top_indices if similarities[i] > 0]

# Dynamic Answer Generation Combining Retrieved Knowledge
def respond_to_question(question):
    relevant_texts = retrieve_relevant_texts(question)
    potential_answers = generate_potential_answers(relevant_texts, question)
    ranked_answers = rank_answers(potential_answers, question)
    short_term_memory.append(question)
    return ranked_answers[0] if ranked_answers else "I'm still learning and don't have an answer yet."

# Generate Answers from Multiple Relevant Sources
def generate_potential_answers(relevant_texts, question):
    answers = []
    for text in relevant_texts:
        # Dynamically generated responses from relevant knowledge
        answers.append(f"Based on what I know: {text[:150]}...")
        answers.append(f"Here's an insight: {text[:200]}...")
        answers.append(f"From my AI knowledge: {text[:100]}...")
    return answers

# Rank Answers Using Weighted Scoring for Relevance, Context, and Feedback
def rank_answers(answers, question):
    ranked = sorted(answers, key=lambda ans: calculate_score(ans, question), reverse=True)
    return ranked

# Calculate Score with Enhanced Weighting
def calculate_score(answer, question):
    relevance_score = relevance(answer, question)
    context_score = context_match(answer)
    feedback_score = feedback_scores.get(answer, 0)
    return relevance_score * 0.5 + context_score * 0.3 + feedback_score * 0.2

# Relevance Scoring Using Keywords and Cosine Similarity
def relevance(answer, question):
    question_vector = vectorizer.transform([question])
    answer_vector = vectorizer.transform([answer])
    similarity = cosine_similarity(answer_vector, question_vector).flatten()[0]
    return similarity

# Context Matching with Short-Term Memory for Conversation Flow
def context_match(answer):
    return sum(1 for q in short_term_memory if any(word in answer for word in extract_keywords(q)))

# Feedback for Adjusting Response Quality
def give_feedback(answer, is_positive):
    if answer in feedback_scores:
        feedback_scores[answer] += 1 if is_positive else -1
    else:
        feedback_scores[answer] = 1 if is_positive else -1

# Extracting Keywords Including Synonyms
def extract_keywords(question):
    words = question.split()
    synonyms = {"learn": ["study", "understand"], "course": ["class", "lesson", "module"]}
    return list(set(chain.from_iterable([synonyms.get(word, [word]) for word in words])))



import streamlit as st
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("Please install the spaCy English model: 'python -m spacy download en_core_web_sm'")
    st.stop()

# Load datasets
@st.cache_data
def load_data():
    try:
        career_data = pd.read_csv("final_extended_career_recommender.csv")
        suggestion_data = pd.read_csv("career_role_suggestions_large_full.csv")
        
        # Combine text fields for vectorization
        career_data['Text'] = (
            career_data['Resume'].fillna('') + ' ' +
            career_data['Interests'].fillna('') + ' ' +
            career_data['Skills'].fillna('')
        )
        
        return career_data, suggestion_data
    except Exception as e:
        st.error(f"Failed to load data files: {e}")
        st.stop()

career_data, suggestion_data = load_data()

# TF-IDF vectorization of dataset
@st.cache_resource
def initialize_vectorizer():
    try:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.8)
        career_vectors = vectorizer.fit_transform(career_data['Text'])
        return vectorizer, career_vectors
    except Exception as e:
        st.error(f"Failed to initialize vectorizer: {e}")
        st.stop()

vectorizer, career_vectors = initialize_vectorizer()

# Extract enhanced keywords from text using spaCy
def extract_keywords(text):
    try:
        doc = nlp(text.lower())
        keywords = set()
        
        # Add noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Avoid long phrases
                keywords.add(chunk.text.strip())
        
        # Add named entities
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "GPE", "NORP", "WORK_OF_ART", "PRODUCT", "EVENT", "SKILL"]:
                keywords.add(ent.text.strip())
        
        # Add important verbs and adjectives
        for token in doc:
            if token.pos_ in ["VERB", "ADJ"] and token.lemma_ not in nlp.Defaults.stop_words:
                keywords.add(token.lemma_)
        
        # Add compound nouns
        for token in doc:
            if token.dep_ == "compound":
                keywords.add(f"{token.text} {token.head.text}")
        
        return " ".join(keywords)
    except Exception as e:
        st.error(f"Error in keyword extraction: {e}")
        return ""

# Improved role recommendation with better similarity handling
def recommend_roles(user_input, threshold=0.25, min_threshold=0.15):
    try:
        cleaned_input = extract_keywords(user_input)
        if not cleaned_input:
            return [], []
        
        user_vector = vectorizer.transform([cleaned_input])
        similarities = cosine_similarity(user_vector, career_vectors).flatten()
        
        # Group by career path and get best scores
        role_scores = defaultdict(float)
        for idx, score in enumerate(similarities):
            role = career_data.iloc[idx]["Career_Path"]
            if score > role_scores[role]:
                role_scores[role] = score
        
        # Sort roles by score and apply threshold
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Dynamic threshold adjustment
        if not sorted_roles or sorted_roles[0][1] < 0.5:
            threshold = min_threshold
        
        recommended_roles = []
        recommended_scores = []
        
        for role, score in sorted_roles:
            if score >= threshold and len(recommended_roles) < 10:
                recommended_roles.append(role)
                recommended_scores.append(score)
        
        return recommended_roles, recommended_scores
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return [], []

# Enhanced alternative suggestions
def suggest_alternatives(main_roles, max_suggestions=10):
    try:
        suggestions = []
        seen = set(main_roles)
        
        for role in main_roles:
            matches = suggestion_data[suggestion_data['Main_Career_Path'] == role]
            for alt in matches['Suggested_Role']:
                if alt not in seen:
                    suggestions.append(alt)
                    seen.add(alt)
                    if len(suggestions) >= max_suggestions:
                        return suggestions
        return suggestions
    except Exception as e:
        st.error(f"Error in suggestion generation: {e}")
        return []

# (Keep your existing role_tips dictionary here...)

# Streamlit App
def main():
    st.title("🎓 CareerCompass: Enhanced Career Path Recommender")
    
    # User input section
    col1, col2 = st.columns([3, 1])
    with col1:
        name = st.text_input("Enter your name (optional):")
    with col2:
        threshold = st.slider("Recommendation Sensitivity", 0.1, 0.5, 0.25, 0.05)
    
    resume_input = st.text_area("Paste your resume or describe your interests, skills, or background:", 
                              height=200,
                              placeholder="E.g.: 'I have a degree in computer science with skills in Python, machine learning, and data analysis. I enjoy solving complex problems and working with large datasets.'")
    
    uploaded_file = st.file_uploader("Or upload your resume as a .txt file", type=["txt"])
    
    user_text = ""
    
    if uploaded_file is not None:
        try:
            user_text = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            try:
                user_text = uploaded_file.read().decode("latin1")
            except Exception as e:
                st.error(f"Failed to decode the file. Error: {e}")
                user_text = ""
    elif resume_input:
        user_text = resume_input
    
    if user_text and st.button("Get Career Recommendations"):
        with st.spinner("Analyzing your profile and finding the best career matches..."):
            st.markdown("---")
            st.subheader("🔍 Career Recommendations")
            
            recs, sims = recommend_roles(user_text, threshold=threshold)
            greeting = f"Hi {name}," if name else "Hi there,"
            
            if not recs:
                st.warning(f"{greeting} we couldn't find strong matches. Try adding more details about your skills, education, and interests.")
                st.info("💡 Tip: Include specific skills, tools you've used, projects you've worked on, and your educational background for better results.")
            else:
                st.write(f"{greeting} here are careers that match your profile:")
                
                # Display recommendations in a nice format
                cols = st.columns(len(recs) if len(recs) <= 5 else 5)
                for i, (role, score) in enumerate(zip(recs, sims)):
                    with cols[i % len(cols)]:
                        st.metric(label=role, value=f"{score:.0%} match")
                
                # Show alternative suggestions
                alt_suggestions = suggest_alternatives(recs)
                if alt_suggestions:
                    st.markdown("---")
                    st.subheader("💡 Related Career Paths to Consider")
                    st.write("You might also explore these related fields:")
                    st.write(", ".join(alt_suggestions))
                
                # Show detailed tips
                st.markdown("---")
                st.subheader("📌 Career Development Tips")
                
                tab1, tab2, tab3 = st.tabs(["Top Recommendations", "All Recommendations", "General Advice"])
                
                with tab1:
                    for role in recs[:3]:
                        tip = role_tips.get(role, "Explore this career path further by connecting with professionals and doing internships.")
                        st.markdown(f"### {role}")
                        st.write(tip)
                
                with tab2:
                    for role in recs:
                        tip = role_tips.get(role, "Explore this career path further by connecting with professionals and doing internships.")
                        with st.expander(role):
                            st.write(tip)
                
                with tab3:
                    st.write("""
                    **General Career Advice:**
                    - Network with professionals in your target field
                    - Build a portfolio of projects demonstrating your skills
                    - Consider certifications relevant to your desired career
                    - Gain practical experience through internships or freelance work
                    - Stay updated with industry trends and technologies
                    """)
    
    # Add some sample inputs for quick testing
    st.markdown("---")
    with st.expander("💡 Not sure what to input? Try these examples"):
        st.write("""
        **Computer Science Graduate:**
        "I have a BS in Computer Science with skills in Python, Java, and SQL. I've worked on machine learning projects using TensorFlow and enjoy data analysis. Looking for roles in tech."
        
        **Business Student:**
        "MBA graduate with finance specialization. Strong analytical skills, Excel modeling experience, and internship at a financial services company. Interested in investment analysis or consulting."
        
        **Career Changer:**
        "5 years experience in retail management looking to transition to HR. Strong people skills, conflict resolution experience, and training in organizational psychology."
        """)

if __name__ == "__main__":
    main()
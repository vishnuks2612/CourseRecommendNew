# course_recommendation_app/views.py

from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from django.http import HttpResponse

import pandas as pd
import joblib

# Load the dataset and pre-trained model

course_similarity_matrix_train = joblib.load('C:\\Users\\Galaxy\\PycharmProjects\\pythonProject\\CourseRecommend\\NoteBook\\course_recommendation')
## Update the column names in text_attributes
text_attributes = ['course_organization', 'course_Certificate_type', 'course_difficulty']

# Load the dataset and pre-trained model
data = pd.read_csv("C:\\Users\Galaxy\\PycharmProjects\\pythonProject\\CourseRecommend\\NoteBook\\coursea_data.csv")

# Combine relevant text-based attributes into a single column for TF-IDF
data['course_text'] = data[text_attributes].apply(lambda x: ' '.join(x), axis=1)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['course_text'])


def course_recommendation(request):
    recommended_courses = []

    if request.method == 'POST':
        organization = request.POST['organization']
        certificate_type = request.POST['certificate_type']
        difficulty = request.POST['difficulty']

        course_attributes = {
            'course_organization': organization,
            'course_Certificate_type': certificate_type,
            'course_difficulty': difficulty,
        }

        course_text = ' '.join(course_attributes.values())
        course_tfidf = tfidf_vectorizer.transform([course_text])
        cosine_similarities = linear_kernel(course_tfidf, tfidf_matrix).flatten()
        similar_courses_indices = cosine_similarities.argsort()[::-1][1:11]  # Get top 5 recommendations
        recommended_courses = [data.iloc[index]['course_title'] for index in similar_courses_indices]

    return render(request, 'index.html', {'recommended_courses': recommended_courses})


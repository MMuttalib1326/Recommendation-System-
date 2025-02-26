# Recommendation-System-
Develop a REST API endpoint using FastAPI to deliver personalized insurance policy recommendations based on user profiles. This task involves integrating a SentenceTransformer model with TensorFlow to compute embeddings and rank policies using similarity scores.

The following are the key objectives:

Model Integration:
Utilize the SentenceTransformer model ("all-MiniLM-L6-v2") to generate embeddings for both user profiles and policy descriptions.
Convert the PyTorch-based embeddings to TensorFlow tensors to perform operations like matrix multiplication.

API Development:
Create a FastAPI application with a POST endpoint (/recommendations/) that accepts a JSON payload representing the user's profile.
Define the user input schema using Pydantic to validate fields such as Age, Location, Health Needs, Budget, Customer Satisfaction Preference, and the number of top recommendations (k).

Recommendation Logic:
Encode the user profile into an embedding vector.
Precompute and store embeddings for a set of insurance policies.
Compute similarity scores between the user embedding and each policy embedding using TensorFlow’s matrix multiplication (tf.matmul).
Rank policies based on similarity scores and return the top-k policies in the API response.

Output Formatting:
Parse the policy information from JSON strings and include additional details (e.g., ranking, similarity score) in the final API response.

Testing & Validation:
Ensure that the API correctly processes various user profiles.
Validate the recommendations by checking the similarity scores and their corresponding policy details.

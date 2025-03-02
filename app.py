from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Movie reviews and labels for multiple movies (from Telugu, Hindi, Tamil, Kannada, Malayalam, and English)
movies_data = {
    "RRR": [
        {"review": "An extraordinary visual spectacle with top-notch performances from NTR and Ram Charan, but the plot could have been more concise.", "label": "Positive"},
        {"review": "A great blend of action, drama, and emotion that kept me at the edge of my seat throughout.", "label": "Positive"},
        {"review": "The story feels a bit stretched at times, but the action sequences are stunning.", "label": "Positive"},
        {"review": "While the film looks great, the pacing slows down at times.", "label": "Negative"},
        {"review": "A modern-day epic with outstanding performances by the leads, but the film is not perfect.", "label": "Positive"},
        {"review": "It has some amazing moments, but the story lacks a bit of depth.", "label": "Negative"},
        {"review": "A visually stunning film with a story full of heroism and emotion.", "label": "Positive"},
        {"review": "The historical aspect was interesting but felt a bit over the top.", "label": "Negative"},
        {"review": "The scale of the film is grand, but the emotional beats could have been stronger.", "label": "Negative"},
        {"review": "A captivating movie with great action but predictable storylines.", "label": "Positive"}
    ],
    "Kabir Singh": [
        {"review": "A powerful, emotionally charged performance by Shahid Kapoor, but the film's toxic masculinity is a big issue.", "label": "Negative"},
        {"review": "An intense portrayal of love and obsession, though the storyline feels troubling at times.", "label": "Negative"},
        {"review": "The chemistry between Shahid Kapoor and Kiara Advani was great, but the film could have had a stronger message.", "label": "Positive"},
        {"review": "The film is gripping but shows a toxic relationship dynamic that might be harmful for viewers.", "label": "Negative"},
        {"review": "A gripping love story with intense performances, though the message was unclear.", "label": "Positive"},
        {"review": "A rollercoaster of emotions, but the film’s portrayal of love is problematic.", "label": "Negative"},
        {"review": "Shahid Kapoor shines in his role, but the movie loses its emotional connection halfway through.", "label": "Negative"},
        {"review": "A heartbreakingly realistic portrayal of a flawed man’s emotional journey.", "label": "Positive"},
        {"review": "The music was soulful, but the film’s tone and direction leave much to be desired.", "label": "Negative"},
        {"review": "A beautifully shot film with a lot of depth, though it leaves you conflicted about its message.", "label": "Positive"}
    ],
    "Master": [
        {"review": "Vijay delivers a mass entertainer with lots of action and a gripping story, though it’s predictable.", "label": "Positive"},
        {"review": "A stylish action film, though it could have used more depth in the storyline.", "label": "Positive"},
        {"review": "Master has some excellent moments, but the screenplay feels stretched.", "label": "Negative"},
        {"review": "A complete mass entertainer with a perfect performance by Vijay, but a weak villain.", "label": "Positive"},
        {"review": "The movie is high on action but lacks any real emotional depth.", "label": "Negative"},
        {"review": "Vijay’s charisma and the movie’s energy keep you hooked, but the story fails to impress.", "label": "Negative"},
        {"review": "Master offers great action sequences, but it’s a bit too predictable.", "label": "Positive"},
        {"review": "A gripping tale of good versus evil, though the plot could have been sharper.", "label": "Negative"},
        {"review": "Vijay’s performance is the highlight, but the film’s pace drags in the second half.", "label": "Negative"},
        {"review": "A thrilling movie with high-octane action, but lacking emotional resonance.", "label": "Positive"}
    ],
    "Kumbalangi Nights": [
        {"review": "A beautiful film with fantastic performances and a refreshing take on relationships.", "label": "Positive"},
        {"review": "A slow-burn movie with rich emotional depth, though not for everyone.", "label": "Positive"},
        {"review": "The film's depiction of family dynamics is heartwarming, but it’s a bit too slow in parts.", "label": "Negative"},
        {"review": "Kumbalangi Nights offers great performances, but the pace drags.", "label": "Negative"},
        {"review": "A subtle, emotional film with amazing character arcs, though the plot could have been tighter.", "label": "Positive"},
        {"review": "A beautiful portrayal of family and bonding, but could have had more engaging moments.", "label": "Positive"},
        {"review": "Kumbalangi Nights is deeply emotional, but it could have been edited better.", "label": "Negative"},
        {"review": "The natural performances and chemistry between the cast are the movie's strength.", "label": "Positive"},
        {"review": "The movie feels authentic but at times, the pace makes it feel longer than it should be.", "label": "Negative"},
        {"review": "A calm, refreshing movie with a unique story, perfect for those who enjoy slow, introspective films.", "label": "Positive"}
    ],
    # Example of other movies
    "Pushpa": [
        {"review": "Allu Arjun delivers a stellar performance, though the film's pacing could have been tighter.", "label": "Positive"},
        {"review": "A gripping story but a bit too long, some scenes drag.", "label": "Negative"},
        {"review": "The visuals are stunning, and the story has a lot of raw energy.", "label": "Positive"},
        {"review": "The film is high on style but low on substance.", "label": "Negative"},
        {"review": "Fantastic action sequences and a powerhouse performance by Allu Arjun.", "label": "Positive"},
        {"review": "A lot of potential but fails to capture the emotional depth it aimed for.", "label": "Negative"},
        {"review": "A film full of intensity and drama, it keeps you hooked from start to finish.", "label": "Positive"},
        {"review": "Though visually striking, the plot gets repetitive and predictable.", "label": "Negative"},
        {"review": "Allu Arjun's performance makes the movie worth watching.", "label": "Positive"},
        {"review": "The film's commercial appeal is strong, but the script lacks depth.", "label": "Negative"}
    ],
    "Sooryavanshi": [
        {"review": "Akshay Kumar brings his A-game, but the story feels recycled.", "label": "Negative"},
        {"review": "The action is top-notch, but the storyline lacks freshness.", "label": "Negative"},
        {"review": "The film offers solid action sequences but fails to leave a lasting impact.", "label": "Negative"},
        {"review": "A high-octane cop thriller with excellent performances, but the plot could use more originality.", "label": "Positive"},
        {"review": "A great combination of drama and action, keeping the audience engaged.", "label": "Positive"},
        {"review": "The film is packed with action but lacks emotional depth.", "label": "Negative"},
        {"review": "A typical Bollywood action film with a predictable plot but worth watching for the action.", "label": "Positive"},
        {"review": "The film delivers what it promises but doesn’t break any new ground.", "label": "Negative"},
        {"review": "Sooryavanshi brings solid action, but fails to offer a fresh take on the cop genre.", "label": "Negative"},
        {"review": "A typical masala film, full of action but lacking in plot and character development.", "label": "Negative"}
    ],
    "Dune": [
        {"review": "An epic adaptation with breathtaking visuals, but the pacing is slow.", "label": "Positive"},
        {"review": "Visually stunning but the plot is a bit too complex.", "label": "Positive"},
        {"review": "A cinematic masterpiece, though it may be hard to follow for casual viewers.", "label": "Positive"},
        {"review": "A very slow burn that could have been edited for a tighter experience.", "label": "Negative"},
        {"review": "The film delivers on visuals but lacks emotional connection.", "label": "Negative"},
        {"review": "Great world-building, but the story feels incomplete.", "label": "Negative"},
        {"review": "An ambitious project that doesn't quite deliver on the emotional front.", "label": "Negative"},
        {"review": "Breathtaking visuals and an intriguing story make it worth watching.", "label": "Positive"},
        {"review": "A bit too long and slow, but the grandeur is impressive.", "label": "Negative"},
        {"review": "A cinematic treat for science fiction lovers, though it might alienate casual viewers.", "label": "Positive"}
    ],
    # Add other 40 movies here
}



# Prepare all the reviews and labels for model training
reviews = []
labels = []

for movie, movie_reviews in movies_data.items():
    for review in movie_reviews:
        reviews.append(review['review'])
        labels.append(1 if review['label'] == "Positive" else 0)

# Train the Naive Bayes model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    movie_name = request.form['movie_name']
    
    if movie_name not in movies_data:
        return render_template('result.html', movie_name=movie_name, prediction="Movie not found currently")
    
    # Get a simulated review for the searched movie (or a random review from the dataset)
    movie_reviews = movies_data[movie_name]
    simulated_review = movie_reviews[0]['review']  # Using the first review as a simulated one
    
    # Transform the simulated review and predict the label
    review_vec = vectorizer.transform([simulated_review])
    prediction = model.predict(review_vec)[0]
    
    result = "Worth Watching" if prediction == 1 else "Not Worth Watching"
    
    return render_template('result.html', movie_name=movie_name, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

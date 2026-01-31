import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    "title": [
        "Avengers", "Iron Man", "Thor", "Captain America", "Black Panther",
        "Doctor Strange", "Guardians Galaxy", "Ant Man", "Hulk",
        "Batman", "Dark Knight", "Joker", "Man of Steel", "Wonder Woman",
        "Aquaman", "Flash",
        "Inception", "Interstellar", "Tenet", "Matrix", "Avatar",
        "Titanic", "La La Land", "Notebook", "Pride and Prejudice",
        "Gladiator", "300", "Braveheart",
        "Jurassic Park", "Jurassic World", "King Kong",
        "Fast Furious", "Fast Five", "Tokyo Drift",
        "Mission Impossible", "MI Fallout",
        "John Wick", "Mad Max"
    ],
    "tags": [
        "action superhero marvel",
        "action superhero marvel",
        "action fantasy superhero marvel",
        "action war superhero marvel",
        "action superhero marvel africa",
        "action fantasy magic marvel",
        "action sci-fi space marvel",
        "action superhero comedy marvel",
        "action superhero marvel",
        "action superhero dc",
        "action crime thriller dc",
        "crime psychological drama dc",
        "action superhero sci-fi dc",
        "action superhero fantasy dc",
        "action superhero fantasy dc",
        "action superhero dc speed",
        "sci-fi thriller mind",
        "sci-fi space emotion",
        "sci-fi time paradox",
        "sci-fi virtual reality",
        "sci-fi fantasy adventure",
        "romance tragedy drama",
        "romance musical drama",
        "romance drama",
        "romance classic drama",
        "action historical war",
        "action war warriors",
        "action historical war",
        "adventure sci-fi dinosaur",
        "adventure sci-fi dinosaur",
        "adventure fantasy monster",
        "action racing crime",
        "action racing heist",
        "action racing drift",
        "action spy thriller",
        "action spy mission",
        "action revenge assassin",
        "action survival apocalypse"
    ]
}

df = pd.DataFrame(data)
df["title_lower"] = df["title"].str.lower()

vector = TfidfVectorizer()
matrix = vector.fit_transform(df["tags"])
score = cosine_similarity(matrix)

def recommend(name):
    name = name.lower()
    if name not in df["title_lower"].values:
        print("Movie not available")
        return

    idx = df[df["title_lower"] == name].index[0]
    values = sorted(list(enumerate(score[idx])), key=lambda x: x[1], reverse=True)

    for i in values[1:6]:
        print(df["title"][i[0]])

print("Movies List:")
for m in df["title"]:
    print(m)

choice = input("\nEnter movie name: ")
recommend(choice)


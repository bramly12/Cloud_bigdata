from flask import Flask, render_template, request

app = Flask(__name__, template_folder='./templates')

@app.route('/')
def home():
    return "Bienvenue dans mon application Flask ! 🚀"

@app.route('/hello')
def hello():
    return 'Bonjour depuis Flask ! 👋'

@app.route('/bonjour/<prenom>')
def greet(prenom):
    return f"Bonjour, {prenom.capitalize()} ! 😄"

@app.route('/articles', methods=['GET'])
def list_articles():
    return {'articles': ['Article 1', 'Article 2', 'Article 3']}

@app.route('/articles', methods=['POST'])
def add_article():
    data = request.json
    article = data.get('article')
    # Ajouter l'article à une base de données
    return {'message': f"Article '{article}' ajouté avec succès! ✅"}, 201

@app.route('/page')
def page_html():
    try:
        return render_template('page.html')
    except Exception as e:
        return f"Erreur: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5005)

from flask import Flask, request, render_template, flash, redirect, url_for
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import joblib
import os
import re

# Loading models
model_path = os.path.join('model', 'random_forest_model.pkl')
vectorizer_path = os.path.join('model', 'vectorizer.pkl')
matplotlib.use('Agg')  # Use non-interactive backend

app = Flask(__name__)
app.secret_key = 'thesensxo'  # Flash messages

# Load the model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Initialize variables for storing results
    reviews, predictions, plot_paths = [], [], {}
    
    # Check if the request contains a URL or file
    if 'url' in request.form and request.form['url']:
        url = request.form['url']
        try:
            # Scraping reviews
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise ValueError('Failed to retrieve the page. Please check the URL.')

            soup = BeautifulSoup(response.text, 'html.parser')
            reviews = [review.text for review in soup.find_all('span', class_='review-text')]
            if not reviews:
                flash('No reviews found. Please check the URL.', 'error')
                return redirect(url_for('index'))

            # Preprocessing
            cleaned_reviews = [re.sub(r'\W+', ' ', review).lower().strip() for review in reviews]

        except Exception as e:
            flash(f'An error occurred while analyzing the URL: {str(e)}', 'error')
            return redirect(url_for('index'))

    elif 'file' in request.files and request.files['file'].filename != '':
        # Handling file uploads
        file = request.files['file']
        if file and allowed_file(file.filename):
            try:
                # Read the uploaded file
                df = pd.read_csv(file)
                reviews = df['review_text'].tolist()  # Assuming the column name in the CSV is 'review_text'
                if not reviews:
                    flash('No reviews found in the uploaded file.', 'error')
                    return redirect(url_for('index'))

                # Preprocessing
                cleaned_reviews = [re.sub(r'\W+', ' ', review).lower().strip() for review in reviews]

            except Exception as e:
                flash(f'An error occurred while reading the file: {str(e)}', 'error')
                return redirect(url_for('index'))

    else:
        flash('No valid input provided. Please enter a URL or upload a file.', 'error')
        return redirect(url_for('index'))

    try:
        # Vectorize and predict
        X = vectorizer.transform(cleaned_reviews)
        predictions = model.predict(X)

        # Plot storage in static directory
        sentiment_counts = pd.Series(predictions).value_counts()
        static_dir = 'static'
        os.makedirs(static_dir, exist_ok=True)

        # Create and save each plot type
        plot_paths = create_plots(sentiment_counts, static_dir)

        return render_template('index.html', reviews=reviews, predictions=predictions, plot_paths=plot_paths)

    except Exception as e:
        flash(f'An error occurred during prediction: {str(e)}', 'error')
        return redirect(url_for('index'))

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    ALLOWED_EXTENSIONS = {'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_plots(sentiment_counts, static_dir):
    """Create and save different plots, returning their paths."""
    plot_paths = {}

    # Bar Plot
    plt.figure(figsize=(5, 4), dpi=100)
    sentiment_counts.plot(kind='bar', color=['green', 'red'])
    plt.title('Sentiment Analysis Results - Bar Plot')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    bar_plot_path = os.path.join(static_dir, 'sentiment_bar_plot.png')
    plt.savefig(bar_plot_path, bbox_inches='tight')
    plt.close()
    plot_paths['bar'] = bar_plot_path

    # Line Plot
    plt.figure(figsize=(5, 4), dpi=100)
    sentiment_counts.plot(kind='line', color='blue', marker='o')
    plt.title('Sentiment Analysis Results - Line Plot')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    line_plot_path = os.path.join(static_dir, 'sentiment_line_plot.png')
    plt.savefig(line_plot_path, bbox_inches='tight')
    plt.close()
    plot_paths['line'] = line_plot_path

    # Pie Chart
    plt.figure(figsize=(5, 5), dpi=100)
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['green', 'red'])
    plt.title('Sentiment Analysis Results - Pie Chart')
    pie_chart_path = os.path.join(static_dir, 'sentiment_pie_chart.png')
    plt.savefig(pie_chart_path, bbox_inches='tight')
    plt.close()
    plot_paths['pie'] = pie_chart_path

    # Scatter Plot
    plt.figure(figsize=(5, 4), dpi=100)
    plt.scatter(sentiment_counts.index, sentiment_counts.values, color='purple')
    plt.title('Sentiment Analysis Results - Scatter Plot')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    scatter_plot_path = os.path.join(static_dir, 'sentiment_scatter_plot.png')
    plt.savefig(scatter_plot_path, bbox_inches='tight')
    plt.close()
    plot_paths['scatter'] = scatter_plot_path

    return plot_paths

if __name__ == '__main__':
    app.run(debug=True)

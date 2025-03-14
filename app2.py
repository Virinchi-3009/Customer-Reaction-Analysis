from flask import Flask, request, render_template, flash
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import joblib
import os
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configure Matplotlib to use non-interactive backend
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'thesensxo'


model_path = os.path.join('model', 'random_forest_model.pkl')
vectorizer_path = os.path.join('model', 'vectorizer.pkl')


model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'url' in request.form:
        url = request.form['url']
        try:
            
            options = Options()
            options.add_argument('--headless')  
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            driver.get(url)

            
            wait = WebDriverWait(driver, 10)
            review_section = wait.until(EC.presence_of_element_located((By.ID, 'customerReviews')))  
            driver.execute_script("arguments[0].scrollIntoView();", review_section)

            # Extracting reviews
            review_elements = driver.find_elements(By.XPATH, "//span[@data-hook='review-body']")
            if not review_elements:
                review_elements = driver.find_elements(By.XPATH, "//span[contains(@class, 'review-text')]")

            reviews = [element.text.strip() for element in review_elements]

            # Quit WebDriver
            driver.quit()

            
            if not reviews:
                flash('No reviews found on the page. Please check the URL or try a different one.', 'error')
                return render_template('index2.html')

            # Preprocess reviews
            cleaned_reviews = [re.sub(r'\W+', ' ', review).lower().strip() for review in reviews]

        except Exception as e:
            flash(f'An error occurred while analyzing the URL: {str(e)}', 'error')
            return render_template('index2.html')

    elif 'file' in request.files:
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading', 'error')
            return render_template('index2.html')
        if file and allowed_file(file.filename):
            try:
                df = pd.read_csv(file)
                reviews = df['review_text'].tolist()
                if not reviews:
                    flash('No reviews found in the uploaded file.', 'error')
                    return render_template('index2.html')

               
                cleaned_reviews = [re.sub(r'\W+', ' ', review).lower().strip() for review in reviews]

            except Exception as e:
                flash(f'An error occurred while reading the file: {str(e)}', 'error')
                return render_template('index2.html')
    else:
        flash('No valid input provided. Please enter a URL or upload a file.', 'error')
        return render_template('index2.html')

    try:
        
        X = vectorizer.transform(cleaned_reviews)
        predictions = model.predict(X)

        
        sentiment_counts = pd.Series(predictions).value_counts()
        static_dir = 'static'
        os.makedirs(static_dir, exist_ok=True)

        plot_paths = {}
        for plot_type, color in zip(['bar', 'line', 'pie', 'scatter'], ['green', 'blue', 'red', 'purple']):
            plt.figure(figsize=(5, 4), dpi=100)
            if plot_type == 'bar':
                sentiment_counts.plot(kind='bar', color=['green', 'red'])
            elif plot_type == 'line':
                sentiment_counts.plot(kind='line', color=color, marker='o')
            elif plot_type == 'pie':
                sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['green', 'red'])
            elif plot_type == 'scatter':
                plt.scatter(sentiment_counts.index, sentiment_counts.values, color=color)

            plt.title(f'Sentiment Analysis Results - {plot_type.capitalize()} Plot')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plot_path = os.path.join(static_dir, f'sentiment_{plot_type}_plot.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            plot_paths[plot_type] = plot_path

        return render_template('index2.html', reviews=cleaned_reviews, predictions=predictions, plot_paths=plot_paths)

    except Exception as e:
        flash(f'An error occurred during prediction: {str(e)}', 'error')
        return render_template('index2.html')

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    ALLOWED_EXTENSIONS = {'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)

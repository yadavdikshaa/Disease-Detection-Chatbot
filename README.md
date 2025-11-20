#  Meddy - Medical Chatbot

An intelligent medical chatbot that uses Natural Language Processing (NLP) and Machine Learning to help users identify potential diseases based on their symptoms. Built with PyTorch, Flask, and modern web technologies.

##  Features

- **NLP-Powered Symptom Recognition**: Uses a neural network trained on medical intents to understand and classify user symptoms
- **Disease Prediction**: Ensemble ML model (kNN, Decision Tree, Logistic Regression, SVM) for accurate disease diagnosis
- **Modern Web Interface**: Beautiful, responsive chat interface with real-time symptom suggestions
- **Severity Assessment**: Automatically recommends consulting a doctor for severe symptoms
- **Disease Information**: Provides detailed descriptions and precautionary measures for predicted diseases

##  Technologies

- **Backend**: Flask, PyTorch, scikit-learn
- **NLP**: NLTK (Natural Language Toolkit)
- **Frontend**: HTML5, CSS3, JavaScript (jQuery)
- **ML Models**: Neural Networks, kNN, Decision Tree, Logistic Regression, SVM, Stacking Classifier

##  Requirements

- Python 3.7+
- Flask
- PyTorch
- NLTK
- NumPy
- Pandas
- scikit-learn
- matplotlib


##  Usage

### Training the Models (First Time)

If models don't exist, train them first:

```bash
python Meddy.py
```

This will:
- Train the NLP model for symptom recognition
- Train disease prediction models
- Save models to `models/` directory

### Running the Application

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

##  Project Structure

```
Medical Chatbot [END 2 END] [NLP]/
├── app.py                 # Flask web application
├── Meddy.py              # Model training script
├── nnet.py               # Neural network architecture
├── nltk_utils.py         # NLP utility functions
├── intents_short.json    # Training data for NLP model
├── data/                 # CSV datasets and pickled data
│   ├── dataset.csv
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── Symptom-severity.csv
├── models/               # Trained models
│   ├── data.pth         # NLP model
│   └── fitted_model.pickle2  # Disease prediction model
├── templates/            # HTML templates
│   └── index.html
└── static/               # CSS, JS, and assets
    ├── css/style.css
    └── js/main.js
```

##  Features in Detail

1. **Symptom Input**: Users can describe symptoms in natural language
2. **Autocomplete**: Real-time symptom suggestions as you type
3. **Symptom Recognition**: NLP model identifies and extracts symptoms from user input
4. **Disease Prediction**: Ensemble model predicts the most likely disease
5. **Information Display**: Shows disease description and precautionary measures
6. **Severity Warning**: Alerts users to consult a doctor for severe symptoms

##  Disclaimer

This chatbot is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions you may have regarding a medical condition.

##  License

This project is for educational purposes. Please refer to the LICENSE file for more information.

##  Credits

Original project by: Alagić Aldin, Benkus Maja, Košmerl Igor, Krištofić Miro  
Course: Intelligent Systems at Faculty of Organization and Informatics Varaždin, University of Zagreb

---

**Note**: Make sure to train the models before running the application if the model files don't exist in the `models/` directory.

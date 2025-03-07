import tkinter as tk
from tkinter import filedialog
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_program1():
    def check_cyberbullying():
        text = input_text.get("1.0", tk.END)

        # TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity < -0.5:
            textblob_output.config(bg="red", fg="white", text="Hate speech detected")
        elif polarity < 0:
            textblob_output.config(bg="orange", fg="black", text="Possible hate speech")
        else:
            textblob_output.config(bg="green", fg="white", text="No hate speech detected")

        # vaderSentiment
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound_score = scores["compound"]
        if compound_score < -0.5:
            vader_output.config(bg="red", fg="white", text="Hate speech detected")
        elif compound_score < 0:
            vader_output.config(bg="orange", fg="black", text="Possible hate speech")
        else:
            vader_output.config(bg="green", fg="white", text="No hate speech detected")

        # flair
        classifier = TextClassifier.load('en-sentiment')
        sentence = Sentence(text)
        classifier.predict(sentence)
        sentiment = sentence.labels[0].value
        if sentiment == "NEGATIVE":
            flair_output.config(bg="red", fg="white", text="Hate speech detected")
        elif sentiment == "POSITIVE":
            flair_output.config(bg="green", fg="white", text="No hate speech detected")
        else:
            flair_output.config(bg="orange", fg="black", text="Possible hate speech")

    # Create GUI for hate speech detection
    root = tk.Tk()
    root.title("Hate Speech Detector")

    # Set window dimensions and center the window on the screen
    window_width = 400
    window_height = 400
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_coordinate = int((screen_width / 2) - (window_width / 2))
    y_coordinate = int((screen_height / 2) - (window_height / 2))
    root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    # Create a frame to hold the GUI elements
    frame = tk.Frame(root, bg="#F0F0F0", pady=20)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a label for the title
    title_label = tk.Label(frame, text="Hate Speech Detector", font=("Arial", 18), bg="#F0F0F0")
    title_label.pack()

    # Create a label and input field for the text
    input_label = tk.Label(frame, text="Enter text to check:", font=("Arial", 12), bg="#F0F0F0")
    input_label.pack(pady=10)
    input_text = tk.Text(frame, height=5, width=50, font=("Arial", 12))
    input_text.pack()

    # Create a button to check for hate speech
    check_button = tk.Button(frame, text="Check for hate speech", command=check_cyberbullying, font=("Arial", 12))
    check_button.pack(pady=10)

    # Create output labels for each library
    textblob_label = tk.Label(frame, text="TextBlob Output:", font=("Arial", 14), bg="#F0F0F0")
    textblob_label.pack()
    textblob_output = tk.Label(frame, text="", font=("Arial", 14), bg="#F0F0F0")
    textblob_output.pack()

    vader_label = tk.Label(frame, text="vaderSentiment Output:", font=("Arial", 14), bg="#F0F0F0")
    vader_label.pack()
    vader_output = tk.Label(frame, text="", font=("Arial", 14), bg="#F0F0F0")
    vader_output.pack()

    flair_label = tk.Label(frame, text="Flair Output:", font=("Arial", 14), bg="#F0F0F0")
    flair_label.pack()
    flair_output = tk.Label(frame, text="", font=("Arial", 14), bg="#F0F0F0")
    flair_output.pack()

    root.mainloop()


def run_program2():
    def perform_sentiment_analysis():
        # Open file dialog to select the CSV file
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

        # Read the CSV file
        amz_review = pd.read_csv(file_path, sep=',', names=['tweet', 'label'])

        # Perform TextBlob sentiment analysis
        amz_review['scores_TextBlob'] = amz_review['tweet'].apply(lambda s: TextBlob(s).sentiment.polarity)
        amz_review['pred_TextBlob'] = amz_review['scores_TextBlob'].apply(lambda x: 1 if x >= 0 else 0)
        textblob_accuracy = accuracy_score(amz_review['label'][1:].astype(int), amz_review['pred_TextBlob'][1:])
        precision_textblob = precision_score(amz_review['label'][1:].astype(int), amz_review['pred_TextBlob'][1:])
        recall_textblob = recall_score(amz_review['label'][1:].astype(int), amz_review['pred_TextBlob'][1:])
        f1_textblob = f1_score(amz_review['label'][1:].astype(int), amz_review['pred_TextBlob'][1:])

        # Perform VADER sentiment analysis
        vader_sentiment = SentimentIntensityAnalyzer()
        amz_review['scores_VADER'] = amz_review['tweet'].apply(lambda s: vader_sentiment.polarity_scores(s)['compound'])
        amz_review['pred_VADER'] = amz_review['scores_VADER'].apply(lambda x: 1 if x >= 0 else 0)
        vader_accuracy = accuracy_score(amz_review['label'][1:].astype(int), amz_review['pred_VADER'][1:])
        precision_vader = precision_score(amz_review['label'][1:].astype(int), amz_review['pred_VADER'][1:])
        recall_vader = recall_score(amz_review['label'][1:].astype(int), amz_review['pred_VADER'][1:])
        f1_vader = f1_score(amz_review['label'][1:].astype(int), amz_review['pred_VADER'][1:])

        # Perform Flair sentiment analysis
        classifier = TextClassifier.load('en-sentiment')

        def score_flair(text):
            sentence = Sentence(text)
            classifier.predict(sentence)
            score = sentence.labels[0].score
            value = sentence.labels[0].value
            return score, value

        amz_review['scores_flair'] = amz_review['tweet'].apply(lambda s: score_flair(s)[0])
        amz_review['pred_flair'] = amz_review['tweet'].apply(lambda s: score_flair(s)[1])
        amz_review['pred_flair'] = amz_review['pred_flair'].map({'NEGATIVE': 0, 'POSITIVE': 1})
        flair_accuracy = accuracy_score(amz_review['label'][1:].astype(int), amz_review['pred_flair'][1:])
        precision_flair = precision_score(amz_review['label'][1:].astype(int), amz_review['pred_flair'][1:])
        recall_flair = recall_score(amz_review['label'][1:].astype(int), amz_review['pred_flair'][1:])
        f1_flair = f1_score(amz_review['label'][1:].astype(int), amz_review['pred_flair'][1:])

        # Update the result label in the GUI
        #result_label.config(text=f"TextBlob Accuracy: {textblob_accuracy}\nVADER Accuracy: {vader_accuracy}\nFlair Accuracy: {flair_accuracy}")
        result_text = f"TextBlob\nAccuracy: {textblob_accuracy}\nPrecision: {precision_textblob}\nRecall: {recall_textblob}\nf1 score: {f1_textblob}\n\n" \
                      f"Vader\nAccuracy: {vader_accuracy}\nPrecision: {precision_vader}\nRecall: {recall_vader}\nf1 score: {f1_vader}\n\n" \
                      f"Flair\nAccuracy: {flair_accuracy}\nPrecision: {precision_flair}\nRecall: {recall_flair}\nf1 score: {f1_flair}"
        result_label.configure(text=result_text)

    # Create GUI for sentiment analysis
    window = tk.Tk()
    window.title("Sentiment Analysis GUI")

    # Create a button to open file dialog
    open_button = tk.Button(window, text="Open CSV File", command=perform_sentiment_analysis)
    open_button.pack(pady=20)

    # Create a label to display the result
    result_label = tk.Label(window, text="")
    result_label.pack()

    # Start the GUI main loop
    window.mainloop()


def select_program():
    def run_selected_program():
        selected_program = program_var.get()
        if selected_program == "Hate Speech Detection":
            run_program1()
        elif selected_program == "Sentimental Analysis Accuracy":
            run_program2()

    # Create GUI for program selection
    root = tk.Tk()
    root.title("Program Selection")

    # Set window dimensions and center the window on the screen
    window_width = 400
    window_height = 400
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_coordinate = int((screen_width / 2) - (window_width / 2))
    y_coordinate = int((screen_height / 2) - (window_height / 2))
    root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    # Create a frame to hold the GUI elements
    frame = tk.Frame(root, bg="#F0F0F0", pady=20)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a label for the title
    title_label = tk.Label(frame, text="Program Selection", font=("Arial", 25), bg="#F0F0F0")
    title_label.pack()

    # Create a label and dropdown menu for program selection
    program_label = tk.Label(frame, text="Select a program to run:", font=("Arial", 15), bg="#F0F0F0")
    program_label.pack(pady=15)
    program_var = tk.StringVar(frame)
    program_var.set("Hate Speech Detection")  # Default selection
    program_dropdown = tk.OptionMenu(frame, program_var, "Hate Speech Detection", "Sentimental Analysis Accuracy")
    program_dropdown.pack()

    # Create a button to run the selected program
    run_button = tk.Button(frame, text="Run Program", command=run_selected_program, font=("Arial", 15))
    run_button.pack(pady=15)

    root.mainloop()

# Run the program selection GUI
select_program()
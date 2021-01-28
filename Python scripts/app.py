from flask import Flask, render_template, request
import backend

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"]) 
def home():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        text = request.form.get('textbox')
        return render_template ("index.html",
        output = backend.meters_feet(float(text)),
        user_text = text)


if __name__ == "__main__":
    app.run()
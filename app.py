from flask import Flask, request, redirect, render_template
from main import main
import os
app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World!"

@app.route("/image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]
            image.save(os.path.join('./', image.filename))
            result = main(image.filename)

            return str(result)


    return render_template("public/upload_image.html")

if __name__ == '__main__':
    app.run()
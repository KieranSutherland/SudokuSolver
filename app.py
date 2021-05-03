from flask import Flask, render_template, Response
from main import main
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

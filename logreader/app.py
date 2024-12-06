import os
from datetime import datetime

import jsonlines
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# Path to the jsonlines file
MESSAGES_FILE = "/home/neocortex/repos/LLaMEA/exp-08-09_095316-codellama_7b-ES pop5-8/conversationlog.jsonl"


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    messages = []
    if os.path.exists(MESSAGES_FILE):
        with jsonlines.open(MESSAGES_FILE) as reader:
            for obj in reader:
                messages.append(obj)
    emit("load_messages", messages)


@socketio.on("new_message")
def handle_new_message(data):
    message = {
        "role": data["role"],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "content": data["content"],
    }
    with jsonlines.open(MESSAGES_FILE, mode="a") as writer:
        writer.write(message)
    emit("new_message", message, broadcast=True)


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5001)

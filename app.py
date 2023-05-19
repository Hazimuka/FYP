from flask import Flask, send_file
import subprocess

app = Flask(__name__)

@app.route('/execute')
def execute_script():
    print('testing')
    result = subprocess.run(['python', 'Driver.py'], capture_output=True, text=True)
    print(result)
    return result


if __name__ == '__main__':
    app.run(host="localhost",port=8000,debug=True)

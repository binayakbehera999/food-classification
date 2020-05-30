import tensorflow as tf
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import base64

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

__author__ = 'ibininja'
app = Flask(__name__)
CORS(app)
arr = []
@app.route("/", methods=["GET","POST"])
def upload():
    with open("/home/binayak/Downloads/a", "rb") as image:
        my_string = base64.b64encode(image.read())
    c=my_string
    inputData=str(request.data)
    inputData=inputData[11:-2]
    c=str.encode(inputData)
    image_data = base64.b64decode(c)
    label_lines = [line.rstrip() for line
                       in tf.io.gfile.GFile("tf_files/retrained_labels.txt")]

    with tf.io.gfile.GFile("tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.compat.v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            arr.append(human_string)
    print(arr[0])
    return jsonify({"foodName": arr[0]})

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port=4555)

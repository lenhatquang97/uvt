import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

def process_result(results, index):
    for result in results:
        boxes = result.boxes
        result.save(filename = "result" + str(index) + ".png")
        image = Image.open("result" + str(index) + ".png")
        st.image(image)
        st.text("Number of items are " + str(len(boxes)))


detectCansModel = YOLO('hen3.pt')
def analyzeDetectCanModel(image, index):
    results = detectCansModel([image])
    process_result(results, index)

detectCoverCanModel = YOLO('nap_h.pt')
def analyzeDetectCoverCanModel(image, index):
    results = detectCoverCanModel([image])
    process_result(results, index)

countPersonModel = YOLO('yolov8n.pt')
def analyzeCountPersonModel(image, index):
    results = countPersonModel([image])
    process_result(results, index)

## Paramter
imgLists = []


## UI Parts
st.title('Heineken Demo')

uploaded_file = st.file_uploader("Choose an image...", type="jpg", accept_multiple_files=True)

for item in uploaded_file:
    if item is not None:
        image = Image.open(item)
        imgLists.append(image)
    
aiMode = st.radio("Chọn Mode", ["Đếm số lon Heineken", "Đếm nắp lon", "Đếm số người"])

if st.button('Analyze'):
    index = 0
    if aiMode == "Đếm số lon Heineken": 
        for item in imgLists:
            analyzeDetectCanModel(item, index)
            index += 1
    elif aiMode == "Đếm nắp lon":
        for item in imgLists:
            analyzeDetectCoverCanModel(item, index)
            index += 1
    elif aiMode == "Đếm số người":
        for item in imgLists:
            analyzeCountPersonModel(item, index)
            index += 1 
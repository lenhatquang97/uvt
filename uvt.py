import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import threading

model = YOLO('hen3.pt')
lock = threading.Lock()

if 'boxNumber' not in st.session_state:
    st.session_state.boxNumber = 0

class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        results = model([img])
        for r in results:
            annotator = Annotator(img)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
            img = annotator.result()
        return img

detectCansModel = YOLO('hen3.pt')
detectCoverCanModel = YOLO('nap_h.pt')
countPersonModel = YOLO('yolov8n.pt')

st.title("Heineken Demo with Video Webcam")

aiMode = st.radio("Chọn Mode", ["Đếm số lon Heineken", "Đếm nắp lon", "Đếm số người"])

if aiMode == "Đếm số lon Heineken":
    model = YOLO('hen3.pt')
elif aiMode == "Đếm nắp lon":
    model = YOLO('nap_h.pt')
elif aiMode == "Đếm số người":
    model = YOLO('yolov8n.pt')

webrtc_streamer(
    key="yolo", 
    video_transformer_factory=YOLOTransformer,
        rtc_configuration={  # Add this config
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )


import av
import cv2
import pafy
import keras
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st  # webrtc works on version 1.33
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics.utils.plotting import Annotator, Colors


st.set_page_config(page_title='YOLO', page_icon='🔥')

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("./image.png")

with col3:
    st.write(' ')

st.title("Welcome to my Streamlit App!")

@st.cache_resource()
def load_yolo(file_path):
    return YOLO(file_path)

@st.cache_resource()
def load_keras(file_path):
    return keras.models.load_model(file_path)

def yolo_classifier(model):
    def wrapped(img, **kwargs):
        results = violence(img, **kwargs)[0]
        label = ''
        if results.names:
            label = results.names[results.probs.top1]
        return label
    return wrapped

def keras_classifier(model):
    def wrapped(img):
        probs = model.predict(np.array(img)[None], verbose=False)[0]
        return ['non_violence', 'violence'][np.argmax(probs)]
    return wrapped

def plot_bounding_boxes(yolo_result, colors=Colors()):
    """Draw bounding boxes from yolo.predict() result"""
    image = yolo_result.orig_img[..., ::-1]
    if yolo_result.boxes:
        annotate = Annotator(np.ascontiguousarray(image))
        classes = yolo_result.boxes.cls.tolist()
        scores = yolo_result.boxes.conf.tolist()
        boxes = yolo_result.boxes.xyxy
        for box, class_, score in zip(boxes, classes, scores):
            tag = f"{yolo_result.names[class_].title()}: {score:.0%}"
            annotate.box_label(box, tag, colors(class_))
        image = annotate.result()
    return Image.fromarray(image)

#violence = yolo_classifier(load_yolo("./runsVio/classify/train/weights/best.pt"))
violence = keras_classifier(load_keras("./Violencenotviolence/violence.keras"))
def violence_classification(img ):
    label = violence(img)
    img = cv2.putText(
        np.array(img),
        label,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 0, 0) if label == 'violence' else (0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return Image.fromarray(img)

crowed = load_yolo("./Croweddata/runs/detect/train17/weights/best.pt")
def crowd_counting(img):
    result = crowed(img, conf=confidence)[0]

    # Count the number of people detected
    num_people = len(result.boxes.xyxy)

    # Draw the count on the image
    # img = np.array(plot_bounding_boxes(result))
    img = np.array(img)
    img = cv2.putText(
        img,
        f'Number of People: {num_people}',
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return Image.fromarray(img)

fall = load_yolo("./FFDData/DangerFightFall/runs/detect/Nano/weights/best.pt")
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_image()
    img = crowd_counting(img)
    img = violence_classification(img)
    img = plot_bounding_boxes(fall(img, conf=confidence)[0])
    return av.VideoFrame.from_image(img)



confidence = st.sidebar.slider('Confidence Threshold', 0., 1., 0.5, step=0.05)
iou = st.sidebar.slider('IoU Threshold', 0., 1., 0.5, step=0.05)

# Add navigation links to the sidebar
selection = st.sidebar.radio("Select Source", ["Webcam", "Youtube", "Image","Video"])

# Display content based on the selected option
if selection == "Webcam":
    webrtc_streamer(
        key="camera",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        video_frame_callback=video_frame_callback,
    )

elif selection == "Youtube":
    video_url = st.text_input("Enter the YouTube video URL")
    if video_url:
        frame_holder = st.empty()
        video = pafy.new(video_url)
        best = video.getbest(preftype="mp4")
        container = av.open(best.url)
        stream = container.streams.video[0]

        height= stream.height
        width = stream.width
        output_size = (width, height)
        
        # Create a video writer to save the processed frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('output.mp4', fourcc, 30, output_size)
        
        # Read and process each frame
        try:
            for frame in container.decode(video=0):
                try:
                    img_array = frame.to_ndarray(format="rgb24")
                    img = Image.fromarray(img_array)
                    img = crowd_counting(img)
                    img = violence_classification(img)
                    img = plot_bounding_boxes(fall(img, conf=confidence)[0])
                    frame_holder.image(img)
                    
                    # Write the processed frame to the video file
                    video_writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
        except Exception as e:
            print('Error', e)

        # Release the video writer
        video_writer.release()
        container.close()

elif selection == "Image":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        img = crowd_counting(img)
        img = violence_classification(img)
        img = plot_bounding_boxes(fall(img, conf=confidence)[0])
        st.image(image=img, caption='processed Image', use_column_width=True)

elif selection == "Video":
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        st.video(uploaded_video)

        container = av.open(uploaded_video)
        stream = container.streams.video[0]

        height= stream.height
        width = stream.width
        output_size = (width, height)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('output_processed.mp4', fourcc, 30, output_size)
        
        frame_holder = st.empty()
        try:
            for frame in container.decode(video=0):
                try:
                    img_array = frame.to_ndarray(format="rgb24")
                    img = Image.fromarray(img_array)
                    img = crowd_counting(img)
                    img = violence_classification(img)
                    img = plot_bounding_boxes(fall(img, conf=confidence)[0])
                    frame_holder.image(img)
                    video_writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
        except Exception as e:
            print('Error', e)

        video_writer.release()
        container.close()
        st.success("Video processed and saved as 'output_processed.mp4'")


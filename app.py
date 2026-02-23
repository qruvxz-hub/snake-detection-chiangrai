import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.title("🐍 ระบบคัดแยกงู จ.เชียงราย")
st.write("จัดทำโดย: ภัทชภณ และ ณกัญญา")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False, safe_mode=False)

model = load_my_model()

uploaded_file = st.file_uploader("📸 เลือกรูปงู...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    st.write("กำลังตรวจนะจ๊ะ...")
    
    # ส่วนประมวลผล (แบบย่อเพื่อลดโอกาสผิด)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
    data = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(data)
    with open("labels.txt", "r", encoding="utf-8") as f:
        labels = f.readlines()
    
    st.subheader(f"ผลคือ: {labels[np.argmax(prediction)].strip()}")

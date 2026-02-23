import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ระบบคัดแยกชนิดงู จ.เชียงราย", layout="centered")

st.title("🐍 ระบบคัดแยกชนิดของงูในจังหวัดเชียงราย")
st.write("จัดทำโดย: เด็กชายภัทชภณ และ เด็กหญิงณกัญญา")
st.write("---")

# ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_my_model():
    # แก้ไขจุดนี้เพื่อรองรับโมเดลจาก Teachable Machine รุ่นใหม่
    return tf.keras.models.load_model("keras_model.h5", compile=False, safe_mode=False)

model = load_my_model()

# ฟังก์ชันทำนายผล
def predict(image_data):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    prediction = model.predict(data)
    return prediction

# ส่วนอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("📸 กรุณาอัปโหลดรูปภาพของงู...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพที่อัปโหลด', use_container_width=True)
    st.write("กำลังประมวลผล...")
    
    results = predict(image)
    
    # อ่านชื่อ labels
    with open("labels.txt", "r", encoding="utf-8") as f:
        class_names = f.readlines()
    
    highest_index = np.argmax(results)
    class_name = class_names[highest_index].strip()
    confidence_score = results[0][highest_index]

    st.subheader(f"🐍 ผลการทำนาย: {class_name}")
    st.write(f"📊 ความเชื่อมั่น: {confidence_score * 100:.2f}%")


import os
import cv2
import numpy as np
from numpy.linalg import norm
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import psycopg2

# ==== KONFIGURASI ====
model_path = "face_detection_yunet_2023mar.onnx"  # Ganti jika perlu
output_folder = "faces"

# Database PostgreSQL config
db_config = {
    "dbname": "face_recognition",
    "user": "postgres",
    "password": "admin",  # GANTI sesuai password kamu
    "host": "localhost",
    "port": 5432
}

# ==== GPU Setup ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== Load Model FaceNet ====
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ==== Transformasi Wajah ====
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ==== Buat Folder Output ====
os.makedirs(output_folder, exist_ok=True)

# ==== Detektor YuNet ====
detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config='',
    input_size=(320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

# ==== Fungsi Koneksi Database ====
def get_pg_connection():
    return psycopg2.connect(**db_config)

def init_database():
    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id SERIAL PRIMARY KEY,
            name TEXT,
            face_feature BYTEA,
            filepath TEXT
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()
    
def get_cropped_face(image):
    """Crop face from image using YuNet."""
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    faces = detector.detect(image)[1]

    if faces is not None:
        for i, face in enumerate(faces):
            x, y, width, height = map(int, face[:4])
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x + width)
            y2 = min(h, y + height)

            # Crop wajah
            face_crop = image[y:y2, x:x2]
            return face_crop
    return None

# ==== Registrasi Wajah ====
def register_face(file_path):
    image = cv2.imread(file_path)
    # Gunakan fungsi get_cropped_face untuk mengambil wajah yang telah dicrop
    face_crop = get_cropped_face(image)
    if face_crop is None:
        raise ValueError("Tidak ada wajah terdeteksi pada gambar.")

    # Ambil nama file sebagai nama orang
    name = os.path.splitext(os.path.basename(file_path))[0]

    # Simpan wajah yang telah dicrop ke folder output
    save_path = os.path.join(output_folder, f"{name}.jpg")
    cv2.imwrite(save_path, face_crop)
    print(f"Wajah disimpan ke: {save_path}")

    # Konversi ke format PIL dan transformasi untuk model
    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    # Dapatkan embedding dari model
    with torch.no_grad():
        embedding = facenet_model(face_tensor).cpu().numpy()

    embedding_blob = embedding.tobytes()

    # Simpan embedding ke database PostgreSQL
    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO faces (name, face_feature, filepath)
        VALUES (%s, %s, %s)
    ''', (name, psycopg2.Binary(embedding_blob), save_path))

    conn.commit()
    cursor.close()
    conn.close()
    print("Selesai ekstraksi dan penyimpanan embedding ke database.")

# ==== Ambil Embedding ====
def get_face_embedding(face_image):
    """Convert face image to embedding vector."""
    cropped_face = get_cropped_face(face_image)
    if cropped_face is None:
        print("Tidak ada wajah terdeteksi.")
        return None
    face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = facenet_model(face_tensor).cpu().numpy()
    return embedding[0]

# ==== Cosine Similarity ====
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ==== Pencocokan Wajah ====
def match_face(image_path, threshold=0.6):
    image = cv2.imread(image_path)
    input_embedding = get_face_embedding(image)

    if input_embedding is None:
        print("Wajah tidak terdeteksi.")
        return None

    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, face_feature FROM faces")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    best_match = "Tidak Dikenal"
    best_similarity = 0.0

    for name, feature_blob in rows:
        db_embedding = np.frombuffer(feature_blob, dtype=np.float32)
        similarity = cosine_similarity(input_embedding, db_embedding)

        if similarity > best_similarity:
            best_similarity = similarity
            if similarity >= threshold:
                best_match = name

    print(f"Hasil pencocokan: {best_match} (similarity: {best_similarity:.4f})")
    return best_match, round(float(best_similarity), 4)

# ==== Ambil Semua Wajah ====
def get_all_faces():
    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM faces")
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

# ==== Hapus Wajah Berdasarkan ID ====
def delete_face_by_id(face_id):
    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filepath FROM faces WHERE id = %s", (face_id,))
    result = cursor.fetchone()

    if result:
        filepath = result[0]
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"File {filepath} dihapus.")
        cursor.execute("DELETE FROM faces WHERE id = %s", (face_id,))
        conn.commit()
        print(f"Data ID {face_id} dihapus dari database.")
    else:
        print("Data ID tidak ditemukan.")

    cursor.close()
    conn.close()

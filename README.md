# Face Recognition API

A Deep Learning-based Face Recognition System using FastAPI, FaceNet (`facenet-pytorch`), YuNet (ONNX), and PostgreSQL for face detection, feature extraction, face matching, and database management.

## Features

- Face detection using YuNet (ONNX).
- Facial feature extraction using FaceNet.
- Face matching via cosine similarity.
- REST API for face registration, recognition, deletion, and listing.
- Storage of face features and associated files in PostgreSQL.

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/PutraAlFarizi15/Face-Recognition-API.git
cd Face-Recognition-API
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PostgreSQL
If PostgreSQL is not already installed, download and install it from the official website:

👉 [Download PostgreSQL](https://www.postgresql.org/download/)

Follow the installation guide according to your operating system (Windows, macOS, or Linux).

### 5. Set Up PostgreSQL Database

```sql
psql -U postgres
-- Enter the password, e.g., admin
CREATE DATABASE face_recognition;
\c face_recognition
```

The required table will be created automatically on the first run.

---

## Running the Application

```bash
uvicorn main:app --reload
```

The app will be accessible at: `http://localhost:8000`

---

## API Endpoints

Open a new Command Prompt, Terminal, or Bash window, and run the following command to use the application endpoint.

### 1. **[GET]** `/api/face`

Get a list of all registered faces in the database.

```bash
curl http://localhost:8000/api/face
```

### 2. **[POST]** `/api/face/register`

Register a new face to the database.

```bash
curl -X POST "http://localhost:8000/api/face/register" -F "file=@<path_to_image>"
```

Example:

```bash
curl -X POST "http://localhost:8000/api/face/register" -F "file=@Aaron Eckhart.jpg"
```

Example Output:

```bash
[{"id":1,"name":"Aaron Eckhart"},{"id":2,"name":"Dasha Taran"}]
```

### 3. **[POST]** `/api/face/recognize`

Recognize a face and match it against the database.

```bash
curl -X POST "http://localhost:8000/api/face/recognize" -F "file=@<path_to_image>"
```

Example:

```bash
curl -X POST "http://localhost:8000/api/face/recognize" -F "file=@Aaron Eckhart.jpg"
```

Example Output:

```bash
{"matched_name":"Aaron Eckhart","similarity_score":1.0}
```

### 4. **[DELETE]** `/api/face/{id}`

Delete a face from the database by its ID.

```bash
curl -X DELETE http://localhost:8000/api/face/<id>
```
Example i want delete id=2:

```bash
curl -X DELETE http://localhost:8000/api/face/2
```

Example Output:

```bash
{"message":"Face ID 2 deleted."}
```

---

## Project Structure

```
.
├── main.py                # FastAPI application
├── face_services.py       # Detection, embedding, and database logic
├── uploads/               # Temporary upload folder
├── faces/                 # Cropped face image storage
├── requirements.txt       # Dependencies list
├── README.md              # This documentation
├── .gitignore             # Git ignore rules (e.g., venv, pycache, etc.)
```

---

## Additional Notes

- The face detection model `face_detection_yunet_2023mar.onnx` must be available in the same directory.
- Face embeddings are stored in binary format in PostgreSQL.
- Matching is performed using cosine similarity with a default threshold of 0.6.

---

## Optional: Docker (Not Included Yet)

If you want to run the application using Docker:

```Dockerfile
# Example Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t face-recognition-api .
docker run -p 8000:8000 face-recognition-api
```

---

## Contact

If you have any questions or need further assistance, feel free to contact me directly at:
Email: putra.alfarizi555@gmail.com

---


![License](https://img.shields.io/badge/license-MIT-green)

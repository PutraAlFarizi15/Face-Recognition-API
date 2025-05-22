from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os

from face_services import init_database, register_face, match_face, get_all_faces, delete_face_by_id

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
def startup_event():
    init_database()

@app.get("/api/face")
def list_faces():
    faces = get_all_faces()
    return [{"id": fid, "name": name} for fid, name in faces]

@app.post("/api/face/register")
def register(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        register_face(temp_path)
        return {"message": f"Face from {file.filename} registered."}
    except ValueError as ve:
        # Special for the case of no face
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Other common error cases
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)  # # Delete after use

@app.post("/api/face/recognize")
def recognize(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        result, score = match_face(temp_path)
        return {
            "matched_name": result, 
            "similarity_score": score,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path) 

@app.delete("/api/face/{face_id}")
def delete_face(face_id: int):
    try:
        delete_face_by_id(face_id)
        return {"message": f"Face ID {face_id} deleted."}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
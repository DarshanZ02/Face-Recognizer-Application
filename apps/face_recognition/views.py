import os
import io
import json
import zipfile
import boto3
import hashlib
from datetime import datetime
from django.http import JsonResponse, FileResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile


# ---------------- CONFIG ----------------
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("AWS_S3_BUCKET")
COLLECTION_ID = os.getenv("AWS_REKOGNITION_COLLECTION")
FACE_MATCH_THRESHOLD = int(os.getenv("FACE_MATCH_THRESHOLD", 90))
UPLOAD_FOLDER = os.path.join(settings.BASE_DIR, "uploads")
CACHE_FILE = os.path.join(settings.BASE_DIR, "embeddings_cache.json")
# ----------------------------------------

# Initialize AWS clients (uses keys from .env)
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

rekognition = boto3.client(
    "rekognition",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# -------- Cache handling --------
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
# --------------------------------


def ensure_collection_exists():
    try:
        rekognition.create_collection(CollectionId=COLLECTION_ID)
    except rekognition.exceptions.ResourceAlreadyExistsException:
        pass


def file_hash(path):
    """Generate MD5 hash to identify duplicates"""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def upload_to_s3(file_path, filename):
    s3.upload_file(file_path, BUCKET_NAME, filename)


def index_faces_if_new(filepaths):
    """Index only new images not already cached"""
    cache = load_cache()

    for path in filepaths:
        filename = os.path.basename(path)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_hash = file_hash(path)
        if img_hash in cache:
            print(f"⏩ Skipping {filename} (already indexed)")
            continue

        try:
            rekognition.index_faces(
                CollectionId=COLLECTION_ID,
                Image={"S3Object": {"Bucket": BUCKET_NAME, "Name": filename}},
                ExternalImageId=filename,
                DetectionAttributes=[]
            )
            cache[img_hash] = filename
            print(f"✅ Indexed {filename}")
        except Exception as e:
            print(f"❌ Failed to index {filename}: {e}")

    save_cache(cache)


# ----------- VIEWS -----------

def home(request):
    return render(request, 'face_recognition/index.html')


@csrf_exempt
def upload_images(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)

    ensure_collection_exists()

    gallery_files = request.FILES.getlist("gallery_images")
    reference_file = request.FILES.get("reference_image")

    if not gallery_files or not reference_file:
        return JsonResponse({"error": "Missing files"}, status=400)

    saved_gallery_paths = []
    for file in gallery_files:
        filename = file.name
        path = os.path.join(UPLOAD_FOLDER, filename)
        with open(path, "wb") as f:
            for chunk in file.chunks():
                f.write(chunk)
        upload_to_s3(path, filename)
        saved_gallery_paths.append(path)

    ref_name = reference_file.name
    ref_path = os.path.join(UPLOAD_FOLDER, ref_name)
    with open(ref_path, "wb") as f:
        for chunk in reference_file.chunks():
            f.write(chunk)

    index_faces_if_new(saved_gallery_paths)

    current_upload_filenames = [os.path.basename(p) for p in saved_gallery_paths]

    with open(ref_path, "rb") as ref_img:
        response = rekognition.search_faces_by_image(
            CollectionId=COLLECTION_ID,
            Image={"Bytes": ref_img.read()},
            FaceMatchThreshold=FACE_MATCH_THRESHOLD,
            MaxFaces=1000
        )

    all_matches = [m["Face"]["ExternalImageId"] for m in response.get("FaceMatches", [])]
    current_batch_matches = [m for m in all_matches if m in current_upload_filenames]

    image_urls = []
    for key in current_batch_matches:
        try:
            url = s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": BUCKET_NAME, "Key": key},
                ExpiresIn=3600
            )
        except Exception:
            url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
        image_urls.append(url)

    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))

    return JsonResponse({"matches": image_urls, "keys": current_batch_matches})


@csrf_exempt
def download_zip(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)

    try:
        data = json.loads(request.body)
        keys = data.get("keys", [])
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not keys:
        return JsonResponse({"error": "No keys provided"}, status=400)

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in keys:
            try:
                obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                content = obj["Body"].read()
                zf.writestr(key, content)
            except Exception:
                continue
    mem_zip.seek(0)
    response = FileResponse(mem_zip, content_type="application/zip")
    response["Content-Disposition"] = "attachment; filename=matches.zip"
    return response

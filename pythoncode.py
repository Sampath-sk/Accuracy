import face_recognition
import cv2
import numpy as np
import dlib

# Load dlib's face detector and landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download separately
detector = dlib.get_frontal_face_detector()

def align_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    face = faces[0]
    shape = predictor(gray, face)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    # Use eyes for alignment
    left_eye = np.mean(landmarks[36:42], axis=0).astype("int")
    right_eye = np.mean(landmarks[42:48], axis=0).astype("int")

    # Compute angle
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Center between eyes
    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                   (left_eye[1] + right_eye[1]) // 2)

    # Get rotation matrix
    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)

    # Rotate image
    aligned_image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]),
                                   flags=cv2.INTER_CUBIC)
    return aligned_image

def compare_faces_accurate(license_path, passport_path, threshold=0.43):
    # Load images
    license_image = cv2.imread(license_path)
    passport_image = cv2.imread(passport_path)

    # Align faces
    aligned_license = align_face(license_image)
    aligned_passport = align_face(passport_image)

    if aligned_license is None or aligned_passport is None:
        return "❌ Face alignment failed. Make sure faces are clearly visible."

    # Convert to RGB for face_recognition
    rgb_license = cv2.cvtColor(aligned_license, cv2.COLOR_BGR2RGB)
    rgb_passport = cv2.cvtColor(aligned_passport, cv2.COLOR_BGR2RGB)

    # Encode faces
    license_enc = face_recognition.face_encodings(rgb_license)
    passport_enc = face_recognition.face_encodings(rgb_passport)

    if not license_enc or not passport_enc:
        return "❌ Face not found in one of the images."

    # Calculate face distance
    face_distance = face_recognition.face_distance([license_enc[0]], passport_enc[0])[0]
    confidence = (1 - face_distance) * 100
    match = face_distance < threshold

    result_text = "✅ Match" if match else "❌ Mismatch"
    return f"{result_text}\nConfidence: {confidence:.2f}%\nDistance: {face_distance:.4f}"

# Example usage
license_img = "license.jpg"
passport_img = "passport.jpg"
print(compare_faces_accurate(license_img, passport_img))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import json
import time
import uuid
import numpy as np
import librosa
from pydub import AudioSegment
from pathlib import Path
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths and models setup
base_path = Path(__file__).resolve().parent
ffmpeg_path = base_path / "ffmpeg.exe"
model_path = base_path / "detect_deepVoice_model.keras"

AudioSegment.converter = str(ffmpeg_path)
deepvoice_model = load_model(str(model_path), compile=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
phishing_model_path = "best_kobert_model.pth"
tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
phishing_model = AutoModelForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=2)
phishing_model.load_state_dict(torch.load(phishing_model_path, map_location=device))
phishing_model.to(device)
phishing_model.eval()

TEMP_DIR = "temp"
SPLIT_DIR = "split_data"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)
load_dotenv()

suspicious_keywords = [
    # 금융 관련 용어
    "계좌", "이체", "입금", "출금", "잔액", "통장",
    "대출", "한도", "이자",
    "은행", "금융감독원", "검찰청", "경찰서",
    "공인인증서", "OTP", "비밀번호", "비밀 번호", "보안 코드",
    "신용 등급", "신용 조사",
    "자동 이체", "연체", "연체료",
    "미납 금액", "초과 금액", "추가 부담금",
    "계정 정지", "계좌 동결",
    "예금 보험", "보험료 납부",

    # 위기감을 조성하는 표현
    "긴급", "위급", "사기",
    "범죄", "피해", "검거",
    "즉시", "지금", "바로",
    "보안", "차단", "동결",
    "확인", "조회", "조사",
    "조속히", "즉시 이행", "지체 없이",
    "경고", "중대 사안",
    "벌금", "압류", "징수",
    "진술서 작성", "기록 남기기",
    "적발", "발각",
    "지급 정지", "잔액 부족",

    # 신뢰를 얻기 위한 표현
    "정부기관", "담당자", "담당부서",
    "공문서", "인증", "확인해 드리겠습니다", "안내드립니다",
    "고지 의무", "통보",
    "의무 사항", "법적 절차",
    "본인 동의", "확약서",

    # 개인정보 및 인증 요구
    "본인 확인", "신분증", "주민등록증",
    "주민번호", "사업자 등록번호",
    "계좌번호", "전화번호 확인",
    "기기 인증", "디지털 서명",
    "본인 인증 요청", "주민등록증 확인",
    "대표 번호", "상담원 연결"
]


# STT
async def transcribe_audio(file: UploadFile):
    CLIENT_ID = os.getenv("CLIENT_ID")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET")

    # JWT 토큰 요청
    auth_resp = requests.post(
        'https://openapi.vito.ai/v1/authenticate',
        data={'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET}
    )
    auth_resp.raise_for_status()
    JWT_TOKEN = auth_resp.json()['access_token']

    url = 'https://openapi.vito.ai/v1/transcribe'
    headers = {'Authorization': f'bearer {JWT_TOKEN}'}
    config = {"use_diarization": True, "use_itn": False, "use_disfluency_filter": False, "use_profanity_filter": False}

    # 오디오 파일을 바이너리 형식으로 읽기
    audio_data = await file.read()
    # STT 요청
    response = requests.post(
        url,
        headers=headers,
        data={'config': json.dumps(config)},
        files={'file': (file.filename, audio_data, file.content_type)}
    )
    response.raise_for_status()
    transcription_id = response.json()['id']

    # 상태 확인 및 결과 가져오기
    status_url = f'https://openapi.vito.ai/v1/transcribe/{transcription_id}'
    while True:
        status_response = requests.get(status_url, headers=headers)
        status_response.raise_for_status()
        status_data = status_response.json()

        if status_data['status'] == 'completed':
            results = status_data['results']['utterances']
            transcribed_text = " ".join([utterance['msg'] for utterance in results])
            return transcribed_text
        elif status_data['status'] == 'failed':
            raise HTTPException(status_code=500, detail=f"Transcription failed for {file.filename}")
        else:
            time.sleep(5)


# 보이스피싱 위험도
def predict_phishing_risk(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    #모델 예측
    with torch.no_grad():
        outputs = phishing_model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs.logits, dim=1)
        phishing_prob = probabilities[0][1].item() * 100
    return phishing_prob


# 딥보이스피싱
# 5초 이상 파일을 5초 이하로 분할하는 함수
def split_audio(file_path: str, split_duration: int = 5000):
    audio = AudioSegment.from_file(file_path, format="wav")
    split_files = []

    if len(audio) <= split_duration:
        split_files.append(file_path)
    else:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        for i, start in enumerate(range(0, len(audio), split_duration)):
            split_audio = audio[start:start + split_duration]
            split_file_name = f"{file_name}_split_{i + 1}_{uuid.uuid4().hex}.wav"
            split_file_path = os.path.join(SPLIT_DIR, split_file_name)
            split_audio.export(split_file_path, format="wav")
            split_files.append(split_file_path)

    return split_files

# 음성 파일을 분석
def detect_fake(file_path: str) -> bool:
    sound_signal, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
    mfccs_features_scaled = np.mean(mfcc_features.T, axis=0)
    mfccs_features_scaled = mfccs_features_scaled.reshape(1, -1)
    result_array = deepvoice_model.predict(mfccs_features_scaled)
    result = np.argmax(result_array[0])
    return result == 0  # "FAKE" if result == 0

# Combined API endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Step 1: Transcribe audio
        transcription = await transcribe_audio(file)

        # Step 2: Phishing risk prediction
        phishing_risk = predict_phishing_risk(transcription)
        risk_keywords = [word for word in suspicious_keywords if word in transcription]

        # Step 3: Deep voice detection
        # 고유한 파일 이름 생성
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(TEMP_DIR, unique_filename)

        # 파일 저장
        with open(file_path, "wb") as f:
            f.write(await file.read())

        split_files = split_audio(file_path) # 음성을 5초 이하로 분할
        deepvoice = any(detect_fake(split_file) for split_file in split_files) # 각 파일에 대해 딥보이스 여부 감지

        return {
            "calls_stt": transcription,
            "phishing_risk": phishing_risk,
            "risk_keywords": risk_keywords,
            # "deepvoice": deepvoice
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        for split_file in split_files:
            if os.path.exists(split_file):
                os.remove(split_file)

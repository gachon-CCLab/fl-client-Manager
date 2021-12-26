from typing import Optional

import boto3
from pydantic.main import BaseModel

import requests
import uvicorn
from fastapi import FastAPI
import asyncio

app = FastAPI()

MODEL_PATH: str = ''
class manager_status(BaseModel):
    INFER_SE: str = '0.0.0.0:8001'
    FL_client: str = '0.0.0.0:8002'
    FL_server_ST: str = '10.152.183.186:8000'
    FL_server: str = 'fl-server:8080'  # '0.0.0.1:8080'
    S3_filename: str = '' # 다운로드된 모델이 저장될 위치
    S3_bucket: str = 'ccl-fl-demo-model'
    S3_key: str = ''# 모델 가중치 파일 이름
    infer_ready: bool =False # 모델이 준비되어있음
    s3_ready: bool =False # s3주소를 확보함
    have_server_ip: bool =True # server 주소가 확보되어있음
    FL_ready: bool =False # FL준비됨

manager=manager_status()
@app.get("/")
def read_root():
    return {"Hello": "World"}

# 모델 pull(requests를 이용하여 s3에 접근해 모델을 받아온다. 모델은 공유 폴더에 저장한다.)
def pull_model():
    global manager
    boto3.client('s3').download_file(manager.S3_bucket, manager.S3_key, manager.S3_filename)
    #예외 처리 추가 필요
    return

# request 상태 검사(10초에 한 번 requests를 이용하여 연합학습 준비가 되어있는지 확인한다)
def health_check():
    global manager
    res = requests.get(manager.FL_server_ST + '/info')
    manager.FL_ready = res.json()['FLSeReady']
    pass
# request 학습 시작(학습 서버에 신호를 보낸다. )
def client_start():
    global manager
    res=requests.get(manager.FL_client+'/start')
    #예외 처리 추가 필요
    return
# request inference 서버 시작
def infer_start():
    global manager
    res=requests.get(manager.INFER_SE+'/start')
    #예외 처리 추가 필요
    return
# request inference 서버 갱신
def infer_update():
    global manager
    res=requests.get(manager.INFER_SE+'/update')
    #예외 처리 추가 필요
    return
# request 서버 정보 수신(서버 상태를 받아온다:(S3관련,FL_server )
def get_server_info():
    global manager
    res = requests.get(manager.FL_server_ST + '/info')
    manager.S3_key = res.json()['S3_key']
    manager.S3_bucket = res.json()['S3_bucket']
    return


# post 학습 완료 정보 수신

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)

from typing import Optional

import boto3
from pydantic.main import BaseModel
import logging
import requests
import uvicorn
from fastapi import FastAPI
import asyncio
import wget

app = FastAPI()


class manager_status(BaseModel):
    INFER_SE: str = '127.0.0.1:8001'
    FL_client: str = '127.0.0.1:8002'
    FL_server_ST: str = '10.152.183.186:8000'
    FL_server: str = '10.152.183.181:8080'  # '0.0.0.1:8080'
    S3_filename: str = '/model/model.h5'  # 다운로드된 모델이 저장될 위치#######################
    S3_bucket: str = 'ccl-fl-demo-model'
    S3_key: str = ''  # 모델 가중치 파일 이름
    infer_ready: bool = False  # 모델이 준비되어있음 infer update 필요
    s3_ready: bool = False  # s3주소를 확보함
    have_server_ip: bool = True  # server 주소가 확보되어있음
    FL_ready: bool = False  # FL준비됨
    infer_running: bool = False  # inference server 작동중
    FL_learning: bool = False  # flower client 학습중


manager = manager_status()


# 모델 pull(requests를 이용하여 s3에 접근해 모델을 받아온다. 모델은 공유 폴더에 저장한다.)
def pull_model():
    global manager
    # s3=boto3.client('s3')
    # s3.download_file(manager.S3_bucket, manager.S3_key, manager.S3_filename)
    # 예외 처리 추가 필요
    url = "https://"+manager.S3_bucket+".s3.ap-northeast-2.amazonaws.com/"+manager.S3_key
    wget.download(url, out=manager.S3_filename)
    manager.infer_ready= True
    return


# request 상태 검사(10초에 한 번 requests를 이용하여 연합학습 준비가 되어있는지 확인한다)
async def health_check():
    global manager
    while True:
        print(manager.FL_learning)

        if manager.FL_learning == False:
            try:
                res = requests.get('http://' + manager.FL_server_ST + '/FLSe/info')
                print('health_check')
                if (res.status_code == 200) and (res.json()['Server_Status']['FLSeReady']):
                    manager.FL_ready = res.json()['Server_Status']['FLSeReady']
                    print(manager.FL_learning)
                    manager.FL_learning = True
                    break
                else:
                    await asyncio.sleep(5)
            except Exception as e:
                print('[E] health_check',e)
                await asyncio.sleep(5)
        else:
            await asyncio.sleep(5)


# request 학습 시작(학습 서버에 신호를 보낸다. )
async def client_start():
    global manager
    print('client_start')

    while True:
        try:
            print('client_start')
            res = requests.get('http://' + manager.FL_client + '/start')
            print('client_start')
            if (res.status_code == 200) and (res.json()['FLCLstart']):
                print('client_start')
                break
            else:
                print(res)
                await asyncio.sleep(2)
        except Exception as e:
            print('[E] client_start',e)
            await asyncio.sleep(5)
    # 예외 처리 추가 필요
    return


# request inference 서버 시작
async def infer_start():
    global manager
    while True:
        try:
            if (manager.infer_running == False) and (manager.infer_ready == True):  # 항상 inferserver가 켜져있도록 한다.
                print('infer_start')
                res = requests.get('http://' + manager.INFER_SE + '/start')
                print('infer_start')
                if (res.status_code == 200) and (res.json()['running']):
                    manager.infer_running = res.json()['running']
                    manager.infer_ready = False
                    print('infer_start')
                    break
                else:
                    print('infer_start_NO')
                    # pass
                    await asyncio.sleep(12)
        except Exception as erra:
            print("infer_start : ", erra)
            await asyncio.sleep(12)

    # 예외 처리 추가 필요
    return


# request inference 서버 갱신
async def infer_update():
    global manager
    while True:
        if manager.infer_ready == True:
            print('infer_update')
            while True:
                try:
                    res = requests.get('http://' + manager.INFER_SE + '/update')
                    print('infer_update')
                    if (res.status_code == 200) and (res.json()['updating']):
                        print('infer_update')
                        manager.infer_ready = False
                        break
                    else:
                        print('infer_update_NO')
                        await asyncio.sleep(13)
                except Exception as e:
                    print('[E] infer_update', e)
                    await asyncio.sleep(13)
        else:
            await asyncio.sleep(13)
    # 예외 처리 추가 필요


# request 서버 정보 수신(서버 상태를 받아온다:(S3관련,FL_server_ip )
def get_server_info():
    global manager
    res = requests.get('http://' + manager.FL_server_ST + '/FLSe/info')
    # print(res.json())
    manager.S3_key = res.json()['Server_Status']['S3_key']
    manager.S3_bucket = res.json()['Server_Status']['S3_bucket']
    manager.s3_ready = True
    return manager


# post 학습 완료 정보 수신

async def training():
    while True:
        await health_check()
        await client_start()


@app.on_event("startup")
def startup():
    ##### S0 #####
    get_server_info()
    pull_model()
    ##### S1 #####
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    loop.create_task(training())
    loop.create_task(infer_start())
    loop.create_task(infer_update())


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/trainFin")
def fin_train():
    global manager
    print('fin')
    manager.infer_ready = True
    manager.FL_learning = False
    return manager

@app.get("/trainFail")
def fin_train():
    global manager
    print('Fail')
    manager.infer_ready = False
    manager.FL_learning = False
    return manager

@app.get('/info')
def get_manager_info():
    return manager


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # asyncio.run(training())
    uvicorn.run("app:app", host='0.0.0.0', port=8080, reload=True)

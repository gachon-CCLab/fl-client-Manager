from pydantic.main import BaseModel
import logging
import requests
import uvicorn
from fastapi import FastAPI
import asyncio
import wget
from functools import partial
import datetime

from functools import wraps
import time

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
app = FastAPI()


class manager_status(BaseModel):
    INFER_SE: str = '127.0.0.1:8001'
    FL_client: str = '127.0.0.1:8002'
    FL_server_ST: str = '10.152.183.186:8000'
    FL_server: str = '10.152.183.181:8080'  # '0.0.0.1:8080'
    S3_filename: str = '/model/model.h5'  # 다운로드된 모델이 저장될 위치#######################
    S3_bucket: str = 'ccl-fl-demo-model'
    S3_key: str = ''  # 모델 가중치 파일 이름
    s3_ready: bool = False  # s3주소를 확보함
    Model_V: int = 0  # 모델버전
    FL_ready: bool = False  # FL server준비됨
    have_server_ip: bool = True  # server 주소가 확보되어있음

    FL_client_online: bool = False  # flower client online?
    FL_learning: bool = False  # flower client 학습중

    infer_online: bool = False  # infer online?
    infer_running: bool = False  # inference server 작동중
    infer_updating:bool = False #i inference server 업데이트중
    infer_ready: bool = False  # 모델이 준비되어있음 infer update 필요


manager = manager_status()


@app.on_event("startup")
def startup():
    ##### S0 #####

    get_server_info()
    asyncio.create_task(pull_model())
    ##### S1 #####
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    # 전역변수값을 보고 상태를 유지하려고 합니다.
    # 이런식으로 짠 이유는 개발과정에서 각 구성요소의 상태가 불안정할수 있기 때문으로
    # manager가 일정주기로 상태를 확인하고 또는 명령에 대한 반환값을 가지고 정보를 갱신합니다
    loop.create_task(health_check())
    loop.create_task(check_infer_online())
    loop.create_task(check_flclient_online())
    loop.create_task(infer_update())
    loop.create_task(start_training())


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/trainFin")
def fin_train():
    global manager
    print('fin')
    manager.infer_ready = True
    manager.FL_learning = False
    manager.FL_ready = False
    manager.Model_V += 1
    return manager


@app.get("/trainFail")
def fail_train():
    global manager
    print('Fail')
    #manager.infer_ready = False
    manager.FL_learning = False
    manager.FL_ready = False
    return manager


@app.get('/info')
def get_manager_info():
    return manager


def async_dec(awaitable_func):
    async def keeping_state():
        while True:
            try:
                logging.debug(str(awaitable_func.__name__) + '함수 시작')
                # print(awaitable_func.__name__, '함수 시작')
                await awaitable_func()
                logging.debug(str(awaitable_func.__name__) + '_함수 종료')
            except Exception as e:
                # logging.info('[E]' , awaitable_func.__name__, e)
                logging.error('[E]' + str(awaitable_func.__name__) + str(e))
            await asyncio.sleep(1)

    return keeping_state


@async_dec
async def health_check():
    global manager
    if (manager.FL_learning == False) and (manager.FL_client_online == True):
        loop = asyncio.get_event_loop()
        # raise
        res = await loop.run_in_executor(None, requests.get, ('http://' + manager.FL_server_ST + '/FLSe/info'))
        if (res.status_code == 200) and (res.json()['Server_Status']['FLSeReady']):
            if res.json()['Server_Status']['Model_V'] != manager.Model_V:
                await pull_model()
            manager.FL_ready = res.json()['Server_Status']['FLSeReady']
            # logging.info('flclient learning')
            # manager.FL_learning = True
        elif (res.status_code != 200):
            # manager.FL_client_online = False
            logging.error('FL_server_ST offline')
            # exit(0)
        else:
            pass
    else:
        pass


@async_dec
async def check_infer_online():
    global manager
    if (manager.infer_running == False) and (manager.infer_ready == True):
        logging.info('infer offline')
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, requests.get, ('http://' + manager.INFER_SE + '/online'))
        if (res.status_code == 200) and (res.json()['running']):
            manager.infer_online = res.json()['infer_online']
            manager.infer_running = res.json()['running']
            manager.infer_ready = False  # infer update 필요없음
            logging.info('infer online')
        else:
            pass
    else:
        await asyncio.sleep(12)


@async_dec
async def check_flclient_online():
    global manager
    if (manager.FL_client_online == False):
        logging.info('FL_client offline')
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, requests.get, ('http://' + manager.FL_client + '/online'))
        if (res.status_code == 200) and (res.json()['FL_client_online']):
            manager.FL_client_online = res.json()['FL_client_online']
            manager.FL_learning = res.json()['FLCLstart']
            logging.info('FL_client online')
        else:
            logging.info('FL_client offline')
            pass
    else:
        await asyncio.sleep(12)


@async_dec
async def infer_update():
    global manager
    if (manager.infer_ready == True) and (manager.infer_running == True) and (manager.infer_online == True):
        logging.info('Start infer_update')
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, requests.get, ('http://' + manager.INFER_SE + '/update'))
        if (res.status_code == 200) and (res.json()['updating']):
            logging.info('infer_updated')
            manager.infer_ready = False
        elif (res.status_code != 200):
            manager.infer_online = False
            manager.infer_running = False
            logging.info('infer_offline')
        else:
            pass
    else:
        await asyncio.sleep(13)


@async_dec
async def start_training():
    global manager
    if (manager.FL_client_online == True) and (manager.FL_learning == False) and (manager.FL_ready == True):
        logging.info('start training')
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, requests.get, ('http://' + manager.FL_client + '/start'))
        if (res.status_code == 200) and (res.json()['FLCLstart']):
            manager.FL_learning = True
            logging.info('start_train')
        elif (res.status_code != 200):
            manager.FL_client_online = False
            logging.info('flclient offline')
        else:
            pass
    else:
        await asyncio.sleep(13)


def get_server_info():
    global manager
    try:
        res = requests.get('http://' + manager.FL_server_ST + '/FLSe/info')
        # print(res.json())
        manager.S3_key = res.json()['Server_Status']['S3_key']
        manager.S3_bucket = res.json()['Server_Status']['S3_bucket']
        manager.s3_ready = True
        manager.Model_V = res.json()['Server_Status']['Model_V']
    except Exception as e:
        raise e
    return manager


async def pull_model():
    global manager
    # s3=boto3.client('s3')
    # s3.download_file(manager.S3_bucket, manager.S3_key, manager.S3_filename)
    # 예외 처리 추가 필요
    loop = asyncio.get_event_loop()
    url = "https://" + manager.S3_bucket + ".s3.ap-northeast-2.amazonaws.com/" + manager.S3_key
    request = partial(wget.download, url, out=manager.S3_filename)
    res = await loop.run_in_executor(None, request)
    manager.infer_ready = True
    logging.debug(res)
    return res


if __name__ == "__main__":
    # asyncio.run(training())
    uvicorn.run("app:app", host='0.0.0.0', port=8080, reload=True)

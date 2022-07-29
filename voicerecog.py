import requests

SECRET_KEY = '' #API 키 입력

headers = {
    'Host': 'kakaoi-newtone-openapi.kakao.com',
    'Content-Type': 'application/octet-stream',
    'X-DSS-Service': 'DICTATION',
    'Authorization': f'KakaoAK {SECRET_KEY}',
}

data = open("", "rb").read() #인식할 파일 추가
response = requests.post('https://kakaoi-newtone-openapi.kakao.com/v1/recognize', headers=headers, data=data)

print(response.text)

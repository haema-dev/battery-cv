# battery-cv

2차전지 Battery 양품/불량 판별 Anomalib 모델 학습 파이프라인

## 폴더구조

```bash
Repository/
├─ .github/workflows/
│    └─ train.yml   ❌ 수정 금지.
├─ command/         ❌ 수정 금지. 명령어 백업용
├─ env/             ❌ 수정 금지. 환경설정 백업용
├─ src/
│    ├─ ???.py      ✅ index 외 기타 파일 필요하면 src 하위에 추가 가능
│    └─ index.py    ✅ 최초 시작 파일
├─ train-job.yml  ⚠️ 환경 변경 시에만 수정 가능
└─ README.md
```

## github push 가이드

### Job Push

#### 파일명 고정

- 추가 파일 있을 시에 경로는 `src` 내부에서 작업해야 함.

```bash
train.yml
train-job.yml
index.py
```

#### 저장할 때 디렉토리 설정 `./outputs` 로 고정

- index. py

```python
.
.
.

parser.add_argument('--output_dir', type=str, default='./outputs', help='결과 저장 경로')

.
.
.
```

#### config 파라미터 필요하면 추가

- index.py

```python
# Azure ML 경로 설정
parser.add_argument('--data_path', type=str, required=True, help='dataset 폴더 경로')
parser.add_argument('--output_dir', type=str, default='./outputs', help='결과 저장 경로')
```

- train-job.yml

```yaml
.
.
.

command: >-
  python input.py
    --data_path ${{inputs.data}}
    --output_dir ${{outputs.result}}

.
.
.
```

# ebackbone 버전·폴더 안내

2026-09-26 로컬 코드와 Git 상태를 기준으로 정리했습니다.
현재 연구의 중심은 **`ebackbone_V3`**입니다. V1/V2에는 앞선 SSMER++ 및
Event2Vec 연구가 남아 있습니다. 폴더 번호가 올라갔다고 해서 동일한 모델의
가중치나 실행 명령이 그대로 호환되는 것은 아닙니다.

## 어느 폴더를 보면 되는가

기준 경로: `/mnt/ssd1/PycharmProjects/`

| 폴더 | 역할 | 주로 확인할 때 |
| --- | --- | --- |
| `ebackbone/` | 초기 SSMER++ 구현. 이 안내에서는 편의상 프로젝트 V1이라 부름 | shared GeneralEventViT, visual fusion, Event2Vec teacher 정렬, 초기 ASL-DVS/N-ImageNet 실험 |
| `ebackbone_v2/` | V1의 실행 경로를 `src/` 구조로 이관한 뒤 확장한 별도 저장소 | ExACT 호환 encoder, 선택적 semantic teacher, 독립 Event2Vec 사전학습·평가·지도학습 |
| `ebackbone_V3/` | 현재 지도학습 event backbone 연구 저장소 | B0/V1 비교 모델, hierarchy 계열, dual fusion, SeACT 전이학습 |
| `ebackbone_asl_clean_v1/` | V1 저장소에 연결된 별도 Git worktree | 당시 ASL-DVS 코드 스냅샷과 비교할 때 |
| `ebackbone_v3_*_artifacts/` | V3 각 실험의 결과와 실행 근거 | 체크포인트, 학습 이력, 보고서, 고정 소스, 큐 상태를 찾을 때 |

V1/V2 코드는 로컬에 보존합니다. 이 안내만 기존
[`Jwooo2002/ebackbone`](https://github.com/Jwooo2002/ebackbone) 저장소의 V3
문서로 게시합니다. V1/V2에는 GitHub 원격을 새로 연결하지 않았습니다.

## 프로젝트 V1: `ebackbone`

현재 작업 트리의 생산용 경로는 하나의 event sample에서 만든 frame, voxel,
time-surface를 각각 가벼운 stem에 통과시킨 뒤 **같은 GeneralEventViT**를
호출합니다. Visual token들을 합쳐 fusion하고, 별도 frozen Event2Vec teacher의
출력은 정렬 loss의 목표로 사용합니다. Teacher token은 이 visual fusion에
포함하지 않습니다. 초기 generic backbone/adapter도 별도 호출 경로와 테스트가
있어 보존했습니다.

| 위치 | 내용 |
| --- | --- |
| `main.py` | ASL-DVS/N-ImageNet pretrain 명령 분기 |
| `ssmerpp/shared_visual.py`, `architecture_profiles.py` | shared visual encoder와 smoke/production 설정 |
| `ssmerpp/fused_event2vec.py`, `fusion.py`, `alignment.py` | visual fusion 및 teacher 정렬 |
| `ssmerpp/event2vec_wrapper.py`, `event2vec_checkpoint.py` | 외부 teacher 로딩과 체크포인트 검증 |
| `ssmerpp/*pretrain*.py` | 데이터별 SSL 설정·실행·재개 |
| `ssmerpp/asl_dvs_ssl_export.py` | 전이학습용 출력 인터페이스와 export |
| `project/`, `docs/ssmerpp/` | 설계 변경 기록, 코드 감사, 실험 handoff |

`project/`에는 현재 shared encoder 이전의 설계를 설명하는 문서도 있습니다.
현재 코드를 이해할 때는 새 루트 README와 `docs/README.md`를 먼저 읽으세요.
`production-v1`이라는 설정 이름과 `512-D` 전이 인터페이스를 프로젝트 버전이나
모든 내부 tensor의 차원으로 해석하면 안 됩니다.

## 프로젝트 V2: `ebackbone_v2`

V1의 활동 중인 코드를 `src/ssmerpp/` 아래로 재구성한 데서 시작했습니다.
이후 다음 경로가 추가되어, 초기 migration 문서보다 구현 범위가 넓습니다.

| 경로 | 의미 |
| --- | --- |
| `src/ssmerpp/encoders/shared_visual.py` | V1에서 이어진 shared GeneralEventViT 경로 |
| `src/ssmerpp/encoders/exact_*` | ExACT 호환 EventEncoder 및 사전학습 조립 코드 |
| `src/ssmerpp/training/` | 데이터별 SSL 실행, continuous runner, semantic-disabled 비교 경로 |
| `src/ssmerpp/event2vec_pretraining/` | 독립적인 label-free masked Event2Vec 사전학습과 frozen 평가 |
| `src/ssmerpp/event2vec_supervised.py` | 별도의 지도학습 Event2Vec 비교 경로 |
| `src/ssmerpp/cli/` | 각 경로의 명령 진입점 |
| `third_party/event2vec_v2/` | 실행에 필요한 외부 모델 코드와 검증된 ASL teacher 체크포인트 |
| `tools/continue_nimagenet_event2vec_epochs.py` | 기존 사전학습 실행의 epoch 연장 도구 |

**V2 전체를 하나의 SSL 모델로 묶어 설명하면 안 됩니다.** Shared visual SSL,
ExACT 호환 경로, 독립 Event2Vec masked pretraining, Event2Vec supervised training은
각각 목적과 설정이 다릅니다. Label-free 경로의 라벨 격리 규칙은 지도학습 경로와
구분해서 확인해야 합니다. 새 README와 `docs/architecture.md`, `docs/testing.md`가
각 경로와 확인 명령을 안내합니다.

`third_party/event2vec_v2/asl_dvs.ckpt`는 해당 teacher 경로의 필수 입력입니다.
확장자가 체크포인트라는 이유로 삭제하지 않습니다. V2의 `exact_clip.py`도
ExACT 호환 구현의 일부로 남아 있으며, V3에서 중단·삭제한 hierarchy→CLIP
실험과는 별개입니다.

## 프로젝트 V3: `ebackbone_V3`

현재 연구는 event backbone을 지도학습으로 비교하고, 학습된 backbone을
SeACT에 전이하는 흐름입니다. V1/V2의 SSMER++ SSL 코드를 그대로 이어서 실행하는
패키지가 아닙니다.

| 코드 | 역할 |
| --- | --- |
| `ebackbone_v3/b0_*.py` | frame-only ResNet-18과 제한된 진단 코드 |
| `ebackbone_v3/v1*.py` | heterogeneous representation 비교와 통제 baseline |
| `ebackbone_v3/hierarchy*.py` | full-event point → voxel → frame hierarchy 및 TS/polarity/구조 ablation |
| `ebackbone_v3/dual_fusion*.py` | hierarchy-only / latent-only / 두 embedding의 학습 가능한 가중합 |
| `ebackbone_v3/seact*.py` | SeACT fine-tuning 및 같은 구조의 scratch 비교 |

여기서 `v1.py`, `HierarchyV1`, `hierarchy_ts_residual_v2`의 숫자는 **V3 내부
모델·실험 버전**입니다. 옆 폴더 `ebackbone`/`ebackbone_v2`를 가리키지 않습니다.
V3의 B1 foundation에는 representation contract와 synthetic smoke가 있고,
실제 tri-representation 비교 모델은 별도 V1 구현에 있습니다.

자세한 구조는 [문서 색인](README.md), [dual fusion](DUAL_FUSION_STUDY.md),
[SeACT](SEACT_STUDY.md)를 참고하세요. 서로 다른 데이터·split·목표를 사용한
V1/V2 결과를 V3 정확도와 단순히 같은 표에서 순위화하면 안 됩니다.

## 결과 폴더는 코드 복사본과 구분하기

| 폴더 패턴 | 보관한 내용 |
| --- | --- |
| `ebackbone_v3_v1_artifacts/` | V3 내부 heterogeneous V1 비교 |
| `ebackbone_v3_hierarchy_v1_artifacts/` | 원래 hierarchy-only 실험 |
| `ebackbone_v3_hierarchy_ts_*_artifacts/` | TS V1, residual V2, confidence V3 비교 |
| `ebackbone_v3_hierarchy_ablation_artifacts/`, `*_polarity_artifacts/` | 구조와 polarity 비교 |
| `ebackbone_v3_hierarchy_ddp_b64_e50_artifacts/` | 기존 batch-64 / 50-epoch DDP 연구 및 보류 큐 |
| `ebackbone_v3_dual_fusion_b64_e50_artifacts/` | 새 hierarchy-only / latent-only / dual 비교 |
| `ebackbone_v3_seact_artifacts/` | SeACT downstream 결과, supervisor, 고정 소스 |
| `ebackbone_v3_hierarchy_clip*_artifacts/` | 중단된 CLIP 연구의 과거 결과·재현 근거 |
| `ebackbone_v3_repo_cleanup_20260926/`, `ebackbone_family_cleanup_20260926/` | 정리 전 백업, 해시, 검증 기록 |

이 폴더들은 삭제하거나 합치지 않았습니다. 일부 run은 source/config 해시를
재개 조건으로 사용합니다. 재개할 때는 해당 run의 원래 snapshot을 사용해야
합니다. 실행 상태는 각 폴더의 `history.json`, `report.json`, queue/study status로
확인하며 이 문서의 설명만으로 완료 여부를 판단하지 않습니다.

## Python 환경과 Git 구분

| 폴더 | Python import | Git 상태 기준점 |
| --- | --- | --- |
| V1 `ebackbone` | `ssmerpp` | `transfer/export-512-interface`, 기존 미커밋 변경 보존 |
| V2 `ebackbone_v2` | `ssmerpp` (`src/` layout) | 별도 저장소 `main`, 기존 미커밋 변경 보존 |
| `ebackbone_asl_clean_v1` | `ssmerpp` | V1의 detached worktree `a50bd5c` |
| V3 `ebackbone_V3` | `ebackbone_v3` | GitHub `Jwooo2002/ebackbone`의 `main` |

V1/V2/ASL worktree는 모두 배포 이름도 `ssmerpp`입니다. 같은 환경에 여러
checkout을 editable 설치하면 어느 소스를 import하는지 혼동될 수 있습니다.
서로 다른 가상환경을 쓰거나, 실행 전에 해당 checkout에서 import 경로를
확인하세요. 버전별 실제 검증 명령은 각 README와 테스트 안내에 기록했습니다.

GitHub 저장소 이름은 `ebackbone`이지만 현재 `main`의 내용은 로컬
`ebackbone_V3`입니다. 로컬 `ebackbone/` 폴더와 연결되어 있다고 가정하지 마세요.

## 이번 정리에서 바뀐 것

- V1: 루트 README와 문서 색인 신설, 현재 shared encoder/teacher 구조와 과거
  설계 기록 구분, 미사용 import 3개 제거, 산출물 제외 규칙 보완.
- V2: README·architecture·runtime·migration 설명 갱신, 합성 CPU 테스트 안내
  추가, 테스트의 특정 머신 절대경로를 checkout 기준 경로로 수정.
- V1/V2의 생성 캐시 21개 디렉터리, 259개 파일 정리. 기존 미커밋 연구 코드,
  필수 teacher checkpoint, 외부 코드, 연결 worktree와 실험 결과는 보존.
- 데이터 독립적인 선별 CPU 검증: V1 **79 passed / 3 deselected**, V2
  **107 passed / 0 skipped**. 실제 데이터 학습이나 후속 실험은 실행하지 않음.

원본 백업과 변경분·해시는 작업공간의 `ebackbone_family_cleanup_20260926/`에
있습니다. V1/V2의 Git HEAD·브랜치·기존 미커밋 상태를 유지했으며, V3에서는
이 안내와 진입 링크만 커밋합니다.

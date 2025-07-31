---
created: 2025-07-24T12:13
updated: 2025-07-30T20:23
---
# parallel cluster를 사용하지 않는 이유
1. 단일 사용자라 스케줄러의 필요성을 느끼지 못함
2. deepspeed launcher에서 이미 pdsh, slurm 등을 backend로 제공하므로 추가적인 stack을 얹을 필요가 없었음
3. 컴퓨팅 노드에서 커널 컴파일 등을 수행해야 해서 어차피 헤드 노드 접근 만으로는 학습이 불가능함
# Deepspeed로 분산학습을 수행하는 이유
1. 현재 연구주제가 극도로 많은 activation memory를 소모하여 단일 sample forward 만으로 OOM이 발생하는 경우가 있음
2. 사용해본 MP/TP 분산학습 기술 중 메모리 효율성 측면에서 가장 좋음
3. VRAM이 충분할 경우 Zero-2(optimizer+gradient 분할), 모자랄 경우 Zero-3(optimizer+grad+param 분할)
4. 실제로 deepspeed 라이브러리의 함수나 기능을 user 레벨에서 사용하지는 않음 (accelerate integration)
* 분산학습에 사용가능한 라이브러리
	* Pytorch DDP: 파라미터, 옵티마이저 상태 분할 불가. 대형 모델 학습에 적합하지 않음
	* Pytorch FSDP: 기술적으로는 Deepspeed Zero 계열과 유사. 초창기에 오픈소스 usecase가 많지 않아 채택하지 않음
	* Megatron-LM: 상대적으로 적은 사용자 pool
	* Ray: 사용해 보지않음
***
# 현재 개발/학습 환경 setup
## Development
1. Python environment managing with uv
	1. 빠른 패키지 설치
	   > conda나 여타 venv 매니저보다 패키지 설치/빌드 속도가 월등히 빠름
	2. 개발 환경 일관성 유지
	   > pyproject.toml과 uv.lock으로 단순 requirements.txt 기반 패키지관리 보다 일관성 있는 환경을 서로 다른 기기 간에 유지 가능
2. Prototyping with jupyter
3. Major editing & debugging with Jetbrains Pycharm IDE
   * 최상위 project를 editable 패키지로 설치해서 수정하며 사용
## Deep Learning Research
1. Backbone 모델은 huggingface transformers 기반 모델을 사용
> - 오픈소스 가중치가 가장 많으며, (나름) 표준화된 LLM, MLLM 인터페이스를 제공함
> - 모델 구조, 인터페이스가 자주 변경되는 것이 단점. 마이너 업데이트 한번 하면 함수가 사라져있음...
2. 내부 모듈 수정을 위해 기본적으로 torch 이용
3. 하드웨어 수준 최적화 코드가 필요한 경우, 간단한 경우는 triton, 저수준 메모리 조작이 필요하면 CUDA(c++)로 gpu 커널 작성하여 컴파일 후, python 바인딩으로 호출
4. 로깅, 하이퍼파라미터 서치는 W&B 활용
***
# 불편했던 점
* 홈 디렉터리의 NFS마운트로 인한 캐시 기반 프로그램 성능 저하
  1. triton 라이브러리 사용시 컴파일된 kernel cache가 $HOME/.triton에 저장됨 ->따로 변경해주어야 함
  2. uv 사용시 모든 파이썬 환경의 import 시간 극도로 증가(30초~1분)
***
# 해결방안 및 현재 설정
1. uv와 프로젝트 내 venv는 가능하면 동일한 물리적 스토리지에 배치 필요
2. 패키지 구성 파일들의 파편화로 인해 고속 random access 성능을 가진 스토리지가 필요함
	* NFS (4k Random Read)
	  ![[300-Resources/301-Files/Pasted image 20250724131425.png]]
	* NFS (4k Random Write)
	  ![[300-Resources/301-Files/Pasted image 20250724130910.png]]
	* Fsx (4k Random Read)
	  ![[300-Resources/301-Files/Pasted image 20250724125330.png]]
	* Fsx (4k Random Write)
	  ![[300-Resources/301-Files/Pasted image 20250724125412.png]]

	* nvme (4k Random Read)
	  ![[300-Resources/301-Files/Pasted image 20250724125850.png]] 
	* nvme (4k Random Write)
	  ![[300-Resources/301-Files/Pasted image 20250724125718.png]]
3. NFS의 성능이 극히 낮아 공유 홈디렉터리 내의 캐시 디렉터리를 fsx 스토리지로 설정해 보았으나, 네트워크 기반 스토리지라 그런지 여전히 엄청나게 느림.
4. 현재는 프로젝트 내 .venv와 uv cache를 전부 각 인스턴스 별 인스턴스 스토어에 심볼릭 링크로 걸어두고 각각 수동으로 동기화해주는 중

* .venv와 .cache가 파일시스템이 다른경우 링크모드를 변경해주어야함
    > `export UV_LINK_MODE=symlink`
    > 파일 시스템이 다른 경우 link mode 환경변수를 심볼릭링크 방식으로 변경해주어야 함(기본은 하드링크)

# 접속 및 기타 개발 환경 공유
* 로컬 접속환경: Mac
* 터미널 클라이언트: iTerm2
* shell: tmux(head node) + zsh(shared)
* IDE/editor: Pycharm, Jupyter Lab, vim
* Assistant: Claude Code
***

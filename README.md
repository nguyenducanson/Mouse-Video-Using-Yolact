# 🐭 Mouse Video Analysis

## 📷 Contents

- [Install](#install)
- [Weight](#weight)
- [Usage](#usage)
- [Contributing to project](#contributing-to-project)
- [Contributors](#contributors)

## 🛠 Install

### ⛑ Prerequisite:
* Python: 3.7+
* Virtualenv

### ⛑ Requirements:
```bash
python3 -m venv venv
pip3 installl --upgrade pip
pip3 install -r requirements.txt
pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## 💚 Weight:
Contact [me](ansonnguyen123456789@gmail.com).

## 🔑 Usage
```
usage: python detect_arm_entry.py --trained_model --score_threshold --videos --count_time

--trained_model: Path of checkpoint file.
--score_threshold: Thresh to choose confident score. [default: 0.05]
--videos: Directory path includes videos to inference.
--count_time: Number of seconds to count arm entry. [default: 300]
```
You can use `--video` to inference on one video and add `output file` to get inferenced video
```bash
python detect_arm_entry.py --trained_model ... --video="test.mp4:result.mp4"
```

### 🔓 Examples

```bash
python detect_arm_entry.py --trained_model=outputs/yolact_plus_resnet50_mouse.pth --score_threshold=0.5 --video="video" --count_time=120
```

## 😘 Contributing to project

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/<YOUR-FEATURE>`)
3. Commit your Changes (`git commit -m 'Add some <YOUR-FEATURE>'`)
4. Push to the Branch (`git push origin feature/<YOUR-FEATURE>`)
5. Open a Pull Request

## 😜 Contributors
@nguyenducanson: `maintainer`
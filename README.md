[README.md](https://github.com/user-attachments/files/27389506/README.md)
# DIFF_FECG# DIFF_Fetal_ECG

Recommended Python version: **3.12**

## 1. Create virtual environment
```powershell
cd D:\project_fetal_ecg\DIFF_Fetal_ECG
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. Install packages
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Dataset layout
Expected folder layout:
```text
D:\project_fetal_ecg\
├── ADFECGDB\
│   ├── r01.edf
│   ├── r01.edf.qrs
│   ├── r04.edf
│   ├── r04.edf.qrs
│   ├── r07.edf
│   ├── r07.edf.qrs
│   ├── r08.edf
│   ├── r08.edf.qrs
│   ├── r10.edf
│   └── r10.edf.qrs
├── BDDB\
│   ├── data.json
│   └── ...
└── DIFF_Fetal_ECG\
```

## 4. Run loader test
```powershell
python train.py
```

This first step only tests:
- ADFECGDB EDF loading
- preprocessing
- segmentation
- BDDB folder inspection

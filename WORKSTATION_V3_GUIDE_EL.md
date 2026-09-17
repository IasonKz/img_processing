# Visual Inspection Pipeline v3 — Οδηγός Windows workstation

Εγκατάσταση από το μηδέν, GPU, Optuna, dashboard, αποτελέσματα και classification.

Οδηγός για το συγκεκριμένο `visual_inspection_pipeline_v3`. Έλεγχος εντολών: 17 Σεπτεμβρίου 2026.

## Πώς χρησιμοποιείς αυτόν τον οδηγό

- Για πρώτη εγκατάσταση ακολουθείς τις ενότητες **1–8 με τη σειρά**. Οι υπόλοιπες είναι αναφορά για αργότερα.
- Όλες οι εντολές είναι για **PowerShell**, είτε μέσα στο PyCharm είτε ως ανεξάρτητο terminal. Όχι Python Console, CMD ή Git Bash.
- Αντιγράφεις ένα code block κάθε φορά. **Δεν εκτελείς ολόκληρο το αρχείο μονομιάς**: περιλαμβάνει εναλλακτικές εκτελέσεις, νέες εκπαιδεύσεις και προαιρετικά exports.
- Όταν μια εντολή αποτύχει, σταματάς στο συγκεκριμένο βήμα και διαβάζεις το error. Δεν προχωράς σε πολύωρο training.
- Δεν διαγράφουμε ούτε αντιγράφουμε παλιά venv. Φτιάχνουμε ένα ξεχωριστό, μόνο για το v3 GPU.
- Δεν χρειάζεται ενεργοποίηση venv. Καλούμε πάντα τον σωστό interpreter με `& $V3Python`.
- Η εγκατάσταση πακέτων και η λήψη δημόσιων weights χρειάζονται επιτρεπόμενη πρόσβαση στο internet. Training και classification χρησιμοποιούν τοπικά δεδομένα.
- Σε εταιρικό υπολογιστή, αν μπλοκάρεται εγκατάσταση, driver ή download, ζητάς από το IT. Δεν παρακάμπτεις εταιρικές πολιτικές, certificates ή antivirus.

**Σημαντικό:** αυτό είναι guide, όχι patch του κώδικα. Τα προηγούμενα `FAILED` trials δεν έχουν διαγνωστεί χωρίς το `error` τους. Περισσότερη GPU ή νέα εγκατάσταση δεν εγγυώνται ότι διορθώνεται η αιτία.

## 1. Τι μεταφέρεις και πού

Μεταφέρεις στον workstation:

1. Τον πλήρη κώδικα v3, μαζί με `inspection.py`, `project.yaml`, `nilt`, `scripts`, requirements και τα υπόλοιπα αρχεία του ZIP.
2. Το dataset VIG: έναν φάκελο με υποφακέλους `good` και `bad`, με τις πραγματικές ετικέτες.
3. Προαιρετικά τον φάκελο των ήδη κατεβασμένων pretrained weights.
4. Προαιρετικά παλιά αποτελέσματα για αναφορά. Για **συνέχιση** ενός run σε άλλον υπολογιστή ακολουθείς την ενότητα 17, όχι απλή αλλαγή των paths μέσα στο run.

Δεν χρειάζονται για μεταφορά τα `.venv`, `.venv_vig`, `.venv_v3` ή `__pycache__`. Δεν χρειάζεται να τα διαγράψεις από το laptop. Τα venv γενικά δεν είναι φορητά· δημιουργούνται ξανά στο νέο μηχάνημα. [Τεκμηρίωση Python venv](https://docs.python.org/3.12/library/venv.html).

Για να αποφύγουμε το προηγούμενο πρόβλημα μεγάλων Windows paths, ο οδηγός χρησιμοποιεί:

| Τι | Διαδρομή στο νέο PC |
|---|---|
| Project | `%USERPROFILE%\vip3` |
| Νέο GPU venv | `%USERPROFILE%\venvs\vip3gpu` |
| Dataset παράδειγμα | `D:\inspection_data\vig` |
| Αποτελέσματα | `results` μέσα στο project, εκτός αν αλλάξεις το YAML |
| Weights | `weights` μέσα στο project, εκτός αν αλλάξεις το YAML |

Το `%USERPROFILE%` είναι ο προσωπικός φάκελος του χρήστη στον **νέο** υπολογιστή. Δεν αντιγράφεις το παλιό `C:\Users\laka` αν ο νέος χρήστης λέγεται αλλιώς.

Αποσυμπίεσε το ZIP και τοποθέτησε τα περιεχόμενα ώστε το αρχείο να βρίσκεται στο `%USERPROFILE%\vip3\inspection.py`. Απόφυγε κατά λάθος διπλό `visual_inspection_pipeline_v3\visual_inspection_pipeline_v3`.

Αν προτιμάς άλλο project directory, άλλαξε το `$V3Project` στις ενότητες 3 και 9. Αν προτιμάς άλλο venv, άλλαξε αντίστοιχα το `$V3Env` και τις αναφορές στον interpreter. Κράτησε σύντομες διαδρομές σε τοπικό δίσκο με δικαίωμα εγγραφής.

## 2. Python και NVIDIA GPU — έλεγχος πριν την εγκατάσταση

### 2.1 Δες τι υπάρχει ήδη

```powershell
py -0p
nvidia-smi
```

- Το πρώτο εμφανίζει εγκατεστημένες Python. Αν δεν αναγνωρίζεται το `py`, λείπει ο launcher ή δεν έχει ενημερωθεί το terminal.
- Το δεύτερο πρέπει να δείχνει NVIDIA GPU και driver. Αν λείπει, δεν λειτουργεί ή έχεις AMD/Intel GPU, **σταματάς τη διαδρομή CUDA αυτού του οδηγού** και ζητάς τα στοιχεία hardware/driver από IT.
- Η ένδειξη `CUDA Version` στο `nvidia-smi` αφορά την υποστήριξη του driver, όχι την έκδοση του εγκατεστημένου PyTorch.

Η συνταγή παρακάτω χρησιμοποιεί Python **3.12 x64**, `torch==2.7.1`, `torchvision==0.22.1` και CUDA wheels `cu128`. Είναι το ζευγάρι που ορίζει το v3, όχι ισχυρισμός ότι είναι οι νεότερες εκδόσεις. Η επίσημη PyTorch τεκμηρίωση δίνει αυτόν τον συνδυασμό για Windows/Linux. [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).

Δεν αλλάζουμε αυθαίρετα σε Python 3.14 ή σε ανεξάρτητες εκδόσεις torch/torchvision. Αυτό δεν σημαίνει ότι κανένα νεότερο PyTorch δεν υποστηρίζει Python 3.14· σημαίνει ότι κρατάμε τη συγκεκριμένη εγκατάσταση συμβατή με τα pins του project.

### 2.2 Εγκατάστησε Python 3.12 αν λείπει

```powershell
winget install --id Python.Python.3.12 --exact --source winget --architecture x64 --scope user
```

Αν ζητήσει όρους, τους διαβάζεις και τους αποδέχεσαι μόνο αν συμφωνείς και επιτρέπεται εταιρικά. Το `--source winget` αποφεύγει αναζήτηση στο Microsoft Store. [Microsoft WinGet install](https://learn.microsoft.com/en-us/windows/package-manager/winget/install).

Αν το winget δεν υπάρχει ή μπλοκάρεται, ζήτησε εγκεκριμένη Python 3.12 x64 με launcher από IT. Μην απεγκαταστήσεις την υπάρχουσα Python που χρειάζεσαι για άλλα projects.

Μετά την εγκατάσταση άνοιξε νέο PowerShell. Αν είναι μέσα στο PyCharm και δεν βλέπει ακόμα το `py`, κλείσε και ξανάνοιξε το PyCharm. Έλεγξε:

```powershell
py -3.12 -c "import sys, struct, platform; print(sys.version); print('Bits:', struct.calcsize('P')*8); print('Arch:', platform.machine())"
```

Θέλεις Python `3.12.x`, `Bits: 64` και συμβατή x64 αρχιτεκτονική. Το `platform.machine()` χρειάζεται παρενθέσεις.

### 2.3 Για τον driver

Χρειάζεται driver συμβατός με τη συγκεκριμένη GPU και τα CUDA 12.8 wheels. Ζήτησε από το IT κατάλληλο ενημερωμένο NVIDIA driver αν αποτύχει το GPU test της ενότητας 4. Μην θεωρήσεις ότι τα «16 GB VRAM» από μόνα τους αρκούν για συμβατότητα.

Δεν απαιτείται να εγκαταστήσεις ξεχωριστά ολόκληρο CUDA Toolkit για τις παρακάτω binary-wheel εντολές. Οι περιορισμοί driver/runtime εξηγούνται από την [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## 3. Νέο, ξεχωριστό περιβάλλον και requirements

Τρέξε αυτόν τον αρχικό ορισμό μεταβλητών. Η πρώτη γραμμή είναι αυτή που αλλάζεις αν το project βρίσκεται αλλού:

```powershell
$V3Project = Join-Path $env:USERPROFILE 'vip3'
$V3Env = Join-Path $env:USERPROFILE 'venvs\vip3gpu'
$V3Python = Join-Path $V3Env 'Scripts\python.exe'
$V3Config = Join-Path $V3Project 'project.yaml'
$V3Profile = 'workstation'

Set-Location -LiteralPath $V3Project
Get-Item -LiteralPath '.\inspection.py', '.\project.yaml', '.\requirements.txt'
```

Αν κάποιο από αυτά τα αρχεία δεν βρίσκεται, διόρθωσε τον φάκελο πριν συνεχίσεις.

Το παρακάτω block δημιουργεί νέο venv και εγκαθιστά τις εξαρτήσεις. **Είναι μόνο για την πρώτη εγκατάσταση**, χωρίς ενεργό training. Αν υπάρχει ήδη `$V3Env`, σταματά για να μην πειράξει άλλο περιβάλλον. Αν έχεις ήδη ολοκληρωμένη εγκατάσταση, πήγαινε στην ενότητα 9. Αν έχει μείνει μισοτελειωμένο setup, κράτησέ το και όρισε νέο όνομα περιβάλλοντος, π.χ. `vip3gpu_new`, και το αντίστοιχο `$V3Python`.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    if (-not (Test-Path -LiteralPath $V3Config -PathType Leaf)) { throw 'Δεν βρέθηκε project.yaml.' }
    if (Test-Path -LiteralPath $V3Env) { throw 'Το venv directory υπάρχει ήδη. Μην το αντικαταστήσεις αν χρησιμοποιείται.' }

    py -3.12 -m venv $V3Env
    if ($LASTEXITCODE -ne 0) { throw 'Απέτυχε η δημιουργία Python 3.12 venv.' }

    & $V3Python -c "import sys, struct; print(sys.executable); assert sys.version_info[:2] == (3,12); assert struct.calcsize('P')*8 == 64"
    if ($LASTEXITCODE -ne 0) { throw 'Λάθος Python ή αρχιτεκτονική.' }

    & $V3Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Απέτυχε το pip setup.' }

    & $V3Python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'Απέτυχε η εγκατάσταση PyTorch CUDA. Σταμάτα εδώ.' }

    & $V3Python -m pip install -r .\requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Απέτυχε η εγκατάσταση requirements.' }

    & $V3Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Βρέθηκε σύγκρουση εξαρτήσεων.' }

    & $V3Python -c "import torch, torchvision, numpy, PIL, yaml, scipy, sklearn, matplotlib, optuna, psutil; print('IMPORTS OK'); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA runtime:', torch.version.cuda)"
    if ($LASTEXITCODE -ne 0) { throw 'Απέτυχε κάποιο import. Σταμάτα εδώ.' }
}
```

Δεν χρειάζεται `torchaudio`. Δεν χρειάζεται να τρέξεις επιπλέον `requirements-training.txt`: οι παραπάνω εντολές εγκαθιστούν ήδη τη βάση και το σωστό torch/torchvision pair, επιλέγοντας ρητά CUDA wheels.

Το ότι στο prompt γράφει κάποιο άλλο `(.venv)` δεν μας επηρεάζει: το `& $V3Python` χρησιμοποιεί ρητά το νέο περιβάλλον. Δεν χρειάζεται `activate`, `deactivate` ή αλλαγή execution policy. [Python: activation is optional](https://docs.python.org/3.12/library/venv.html).

## 4. Επιβεβαίωσε ότι η GPU κάνει πραγματικό υπολογισμό

Το `torch.cuda.is_available()` μόνο του δεν είναι πλήρες τεστ. Το παρακάτω δοκιμάζει convolution, backward και optimizer update στη GPU με συνθετικά δεδομένα, χωρίς εταιρικές εικόνες ή downloads:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    nvidia-smi
    if ($LASTEXITCODE -ne 0) { throw 'Δεν λειτουργεί το nvidia-smi.' }

    & $V3Python -c "import torch, torchvision; print('torch:',torch.__version__); print('torchvision:',torchvision.__version__); print('CUDA runtime:',torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:',torch.cuda.get_device_name(0)); print('VRAM GiB:',round(torch.cuda.get_device_properties(0).total_memory/2**30,2)); print('Capability:',torch.cuda.get_device_capability(0)); m=torch.nn.Conv2d(3,8,3).cuda(); x=torch.randn(2,3,64,64,device='cuda'); opt=torch.optim.AdamW(m.parameters()); loss=m(x).square().mean(); loss.backward(); opt.step(); torch.cuda.synchronize(); print('GPU FORWARD/BACKWARD OK:',float(loss.detach().cpu()))"
    if ($LASTEXITCODE -ne 0) { throw 'Απέτυχε το GPU test. Μην ξεκινήσεις workstation search.' }
}
```

Θέλεις `GPU FORWARD/BACKWARD OK`. Αν δεις `+cpu`, `CUDA runtime: None`, `CUDA unavailable`, `no kernel image` ή driver error, πήγαινε στην ενότητα 19. Δεν συνεχίζεις υποθέτοντας ότι το πρόγραμμα εκπαιδεύει στη GPU.

## 5. Ρύθμισε μόνο τα paths στο project.yaml

Άνοιξέ το με το PyCharm ή:

```powershell
notepad.exe $V3Config
```

Μέσα στο υπάρχον αρχείο κράτησε την υπόλοιπη δομή και διόρθωσε τα αντίστοιχα πεδία. **Το παρακάτω είναι απόσπασμα YAML, όχι ολόκληρο project.yaml και όχι PowerShell command.** Μην δημιουργήσεις δεύτερα `paths:` ή `tasks:`.

```yaml
paths:
  output_root: ./results
  weights_root: ./weights
  incoming_root: ./incoming
  classification_output_root: ./classified

tasks:
  vig:
    enabled: true
    dataset_root: 'D:/inspection_data/vig'
    augmentation:
      rotation90: true
  top:
    enabled: false
    dataset_root: ./data/top
  bottom:
    enabled: false
    dataset_root: ./data/bottom
```

Αν το PC δεν έχει `D:`, βάλε τον πραγματικό φάκελό σου. Το `dataset_root` είναι ο γονικός φάκελος που περιέχει **και** `good` **και** `bad`, όχι ο ένας από τους δύο. Μη βάλεις τα `results`, `classified`, `incoming`, weights ή smoke outputs μέσα στο dataset.

Κράτησε `resize: pad`, `preserve_native_resolution: true`, `image_size: auto` και VIG περιστροφές μόνο κατά 90°. Μην αλλάξεις την ανάλυση για να κρύψεις σφάλμα μνήμης. Τα 16 GB VRAM δεν εγγυώνται ότι κάθε μοντέλο χωρά σε κάθε native ανάλυση.

Στο v3 το profile επιλέγεται με `--profile workstation` ή μέσα στο dashboard. Δεν προσθέτεις αυθαίρετο νέο top-level `profile:` στο YAML.

Διάβασε τα πραγματικά resolved paths και αποθήκευσέ τα ως PowerShell μεταβλητές:

```powershell
& $V3Python -c "from nilt.common import load_project; import sys,json; c=load_project(sys.argv[1],sys.argv[2],'vig')[0]; print(json.dumps({k:c[k] for k in ('profile','device','models','data_root','output_root','weights_dir','incoming_root','classification_output_root')},indent=2))" $V3Config $V3Profile

$V3Data = & $V3Python -c "from nilt.common import load_project; import sys; print(load_project(sys.argv[1],sys.argv[2],'vig')[0]['data_root'])" $V3Config $V3Profile
if ($LASTEXITCODE -ne 0) { throw 'Config error.' }
$V3Results = & $V3Python -c "from nilt.common import load_project; import sys; print(load_project(sys.argv[1],sys.argv[2],'vig')[0]['output_root'])" $V3Config $V3Profile
if ($LASTEXITCODE -ne 0) { throw 'Config error.' }
$V3Weights = & $V3Python -c "from nilt.common import load_project; import sys; print(load_project(sys.argv[1],sys.argv[2],'vig')[0]['weights_dir'])" $V3Config $V3Profile
if ($LASTEXITCODE -ne 0) { throw 'Config error.' }

Get-Item -LiteralPath (Join-Path $V3Data 'good'), (Join-Path $V3Data 'bad')
```

Οι σχετικές διαδρομές του YAML υπολογίζονται δίπλα στο YAML. Οι σχετικές διαδρομές CLI όπως `--output-root .\decisions` υπολογίζονται από το τρέχον terminal directory· γι' αυτό δουλεύουμε από το project root.

## 6. Pretrained weights και doctor

### 6.1 Κατέβασε τα weights του workstation profile

Αν έχεις ήδη μεταφέρει τα σωστά αρχεία, βάλε τον φάκελό τους στο `paths.weights_root`. Το download command ελέγχει υπάρχοντα αρχεία και δεν ξανακατεβάζει κανονικά όσα υπάρχουν με σωστό hash.

```powershell
& $V3Python inspection.py download-weights --config $V3Config --profile workstation --task vig
if ($LASTEXITCODE -ne 0) { throw 'Weights setup failed. Μην προχωρήσεις στο training.' }
```

Κατεβάζει δημόσια pretrained weights. Δεν στέλνει το dataset. Για το προεπιλεγμένο workstation roster καλύπτει ResNet18, EfficientNet-B0, ResNet50, ConvNeXt-Tiny και Swin-T. Το PatchCore χρησιμοποιεί τα weights του backbone του, προεπιλεγμένα ResNet18.

Αν χρειάζεσαι ρητά μόνο ορισμένα μοντέλα:

```powershell
& $V3Python download_weights.py --output $V3Weights --models resnet18 efficientnet_b0
```

Ή όλα τα supervised backbones που υποστηρίζει το project:

```powershell
& $V3Python download_weights.py --output $V3Weights --models resnet18 efficientnet_b0 resnet50 convnext_tiny swin_t
```

Δεν δίνεις `patchcore` στο `download_weights.py`: δίνεις το backbone του. Η ρύθμιση στο YAML λέγεται `patchcore.backbone`.

### 6.2 Έλεγχος του project

```powershell
& $V3Python -m pip check
& $V3Python inspection.py doctor --config $V3Config --profile workstation --task vig
```

Θέλεις `ready: true`, `resources_ready: true` και `resolved_device: cuda`. Αν η έξοδος δείχνει πρόβλημα, μην ξεκινήσεις πολύωρο training.

Τι αποδεικνύει κάθε έλεγχος:

| Έλεγχος | Τι ελέγχει — και τι όχι |
|---|---|
| `pip check` | Δηλωμένες εξαρτήσεις/συγκρούσεις εκδόσεων. Όχι πλήρη λειτουργία kernels ή εφαρμογής. |
| Import test ενότητας 3 | Ότι τα βασικά modules φορτώνονται. Όχι όλα τα πιθανά code paths. |
| `doctor` | Configuration, παρουσία πακέτων, dataset directory, weights files, επιλογή device. Δεν εκπαιδεύει όλα τα μοντέλα ούτε αποδεικνύει ότι όλα τα weights/inputs είναι έγκυρα. |
| GPU test ενότητας 4 | Μικρό πραγματικό forward/backward στη GPU. Όχι ότι κάθε native-resolution μοντέλο χωρά στη VRAM. |
| Smoke test ενότητας 7 | Μικρό ολοκληρωμένο workflow σε συνθετικά δεδομένα. Όχι απόδοση στα δικά σου defects. |

Με `pretrained: true`, απόντα weights προκαλούν `Missing LOCAL pretrained weights`. Το training δεν τα κατεβάζει μόνο του και δεν τα αντικαθιστά σιωπηλά με random weights. Κατεστραμμένο αρχείο μπορεί να αποτύχει αργότερα κατά το loading.

Το ότι ένα μοντέλο έφτασε σε epochs αποδεικνύει ότι η αρχική φόρτωση/εκτέλεση προχώρησε, **όχι** ότι όλα τα requirements και όλα τα επόμενα στάδια είναι εγγυημένα σωστά.

## 7. Μικρό end-to-end test πριν το μεγάλο search

Τρέξε μία φορά, πριν αρχίσεις πολύωρο training:

```powershell
$V3Smoke = Join-Path $V3Project ('smoke_checks\check_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& $V3Python scripts/smoke_v3.py --output $V3Smoke --weights $V3Weights
if ($LASTEXITCODE -ne 0) { throw 'Smoke test failed. Κράτησε το error και μη ξεκινήσεις πολύωρο search.' }
```

Θέλεις `SMOKE PASSED`. Το υπάρχον `smoke_v3.py` χρησιμοποιεί μικρές συνθετικές εικόνες και **CPU/laptop** configuration. Ελέγχει το workflow, όχι το workstation GPU profile· το GPU test είναι ξεχωριστό στην ενότητα 4. Δεν αγγίζει το πραγματικό dataset και τα συνθετικά metrics δεν έχουν βιομηχανική σημασία.

### Προαιρετικά: ONNX/export dependencies

Δεν απαιτούνται για training. Αν θέλεις export, εγκατέστησέ τα όταν δεν τρέχει job:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    & $V3Python -m pip install -r .\requirements-export.txt
    if ($LASTEXITCODE -ne 0) { throw 'Export dependency installation failed.' }
    & $V3Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency conflict.' }
    & $V3Python inspection.py doctor --config $V3Config --profile workstation --task vig --deployment
    if ($LASTEXITCODE -ne 0) { throw 'Deployment doctor failed.' }
}
```

Μετά, νέο synthetic check που περιλαμβάνει ONNX:

```powershell
$V3Smoke = Join-Path $V3Project ('smoke_checks\onnx_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& $V3Python scripts/smoke_v3.py --output $V3Smoke --weights $V3Weights --with-onnx
```

## 8. Ξεκίνα από το dashboard — η απλή καθημερινή χρήση

```powershell
& $V3Python inspection.py dashboard --config $V3Config
```

Ανοίγει στο [τοπικό dashboard](http://127.0.0.1:8765). Κράτησε το terminal ανοιχτό. Αν δεν ανοίξει browser, βάλε εσύ αυτή τη διεύθυνση.

1. Στο **Dataset & setup**, έλεγξε VIG και τον σωστό φάκελο.
2. Στο **Train & search**, επίλεξε `workstation`, έως 6 ώρες, και τα μοντέλα που θέλεις.
3. Πάτησε **Start Optuna search μία φορά**. Μην ξεκινήσεις παράλληλα CLI search για το ίδιο πείραμα.
4. Δες `Recorded trials` και `Operation log`. Για `FAILED`, διάβασε το πραγματικό error από την ενότητα 13 αντί να συμπεράνεις ότι το μοντέλο είναι κακό.
5. Μετά την ολοκλήρωση, πήγαινε **Compare models**, **Decision explorer**, **Models & reports**.

| Profile | Device | Συνολικό όριο | Προεπιλεγμένα μοντέλα στο παραδοθέν YAML |
|---|---|---:|---|
| `laptop` | CPU | 2 ώρες | ResNet18, EfficientNet-B0 |
| `workstation` | NVIDIA CUDA | 6 ώρες | ResNet18, EfficientNet-B0, ResNet50, ConvNeXt-Tiny, Swin-T, PatchCore |
| `workstation_large` | NVIDIA CUDA | 12 ώρες | Τα ίδια, με μεγαλύτερα όρια trials/epochs και περισσότερα seeds |

Ο χρόνος είναι **συνολικός**, όχι ανά μοντέλο. Περισσότερα μοντέλα σημαίνουν λιγότερο χρόνο αναζήτησης ανά μοντέλο. Ο αριθμός trials είναι ανώτατο όριο, όχι υπόσχεση ότι θα ολοκληρωθούν όλα.

Αν η αποθήκευση ρυθμίσεων από το dashboard δημιουργήσει `project.dashboard.<timestamp>.yaml`, αυτό είναι νέο config snapshot. Για επανεκκίνηση με εκείνες τις ρυθμίσεις, χρησιμοποίησε το συγκεκριμένο αρχείο ως `$V3Config`. Μην θεωρήσεις ότι άλλαξε υποχρεωτικά το αρχικό `project.yaml`.

Δεν χρειάζεται το `inspection_observed.py` του παλιού wrapper για αυτή τη ροή v3.

## 9. Αύριο: πώς το ξανανοίγεις χωρίς εγκατάσταση

Νέο PowerShell σημαίνει ότι οι μεταβλητές δεν διατηρούνται. Βάλε αυτό το block στην αρχή κάθε νέου terminal:

```powershell
$V3Project = Join-Path $env:USERPROFILE 'vip3'
$V3Env = Join-Path $env:USERPROFILE 'venvs\vip3gpu'
$V3Python = Join-Path $V3Env 'Scripts\python.exe'
$V3Config = Join-Path $V3Project 'project.yaml'
$V3Profile = 'workstation'
Set-Location -LiteralPath $V3Project

if (-not (Test-Path -LiteralPath $V3Python -PathType Leaf)) { throw 'Δεν βρέθηκε το GPU venv.' }
if (-not (Test-Path -LiteralPath $V3Config -PathType Leaf)) { throw 'Δεν βρέθηκε το configuration.' }

$V3Data = & $V3Python -c "from nilt.common import load_project; import sys; print(load_project(sys.argv[1],sys.argv[2],'vig')[0]['data_root'])" $V3Config $V3Profile
if ($LASTEXITCODE -ne 0) { throw 'Config error.' }
$V3Results = & $V3Python -c "from nilt.common import load_project; import sys; print(load_project(sys.argv[1],sys.argv[2],'vig')[0]['output_root'])" $V3Config $V3Profile
if ($LASTEXITCODE -ne 0) { throw 'Config error.' }
$V3Weights = & $V3Python -c "from nilt.common import load_project; import sys; print(load_project(sys.argv[1],sys.argv[2],'vig')[0]['weights_dir'])" $V3Config $V3Profile
if ($LASTEXITCODE -ne 0) { throw 'Config error.' }
```

Μετά, για dashboard:

```powershell
& $V3Python inspection.py dashboard --config $V3Config
```

Δεν ξανατρέχεις `pip install` σε κάθε εκκίνηση. Δεν αλλάζεις packages ενώ εκπαιδεύεται μοντέλο.

Στο PyCharm μπορείς να επιλέξεις ως υπάρχον interpreter το `%USERPROFILE%\venvs\vip3gpu\Scripts\python.exe`. Τα ακριβή ονόματα μενού διαφέρουν ανά έκδοση· οι εντολές παραπάνω λειτουργούν ανεξάρτητα από τον επιλεγμένο interpreter του IDE.

## 10. Training από PowerShell αντί για το κουμπί

Οι παρακάτω εντολές είναι **εναλλακτικές**. Διαλέγεις μία. Κάθε `search --config` δημιουργεί νέο run και κάνει δικό του audit. Δεν χρειάζεται να εκτελέσεις προηγουμένως ξεχωριστό `audit`.

### Όλα τα μοντέλα του workstation profile, έως 6 ώρες

```powershell
& $V3Python inspection.py search --config $V3Config --profile workstation --task vig --hours 6
```

### Όλα τα μοντέλα του μεγάλου workstation profile, έως 12 ώρες

```powershell
& $V3Python inspection.py search --config $V3Config --profile workstation_large --task vig --hours 12
```

### Μόνο ResNet18 και EfficientNet-B0, GPU, έως 6 ώρες

```powershell
& $V3Python inspection.py search --config $V3Config --profile workstation --task vig --models resnet18 efficientnet_b0 --hours 6
```

### Μόνο ένα μοντέλο, GPU, έως 2 ώρες

```powershell
& $V3Python inspection.py search --config $V3Config --profile workstation --task vig --models efficientnet_b0 --hours 2
```

### Ρητή λίστα όλων των μοντέλων

```powershell
& $V3Python inspection.py search --config $V3Config --profile workstation --task vig --models resnet18 efficientnet_b0 resnet50 convnext_tiny swin_t patchcore --hours 6
```

Υποστηριζόμενα model IDs: `resnet18`, `efficientnet_b0`, `resnet50`, `convnext_tiny`, `swin_t`, `patchcore`.

Για να προσθέσεις μοντέλα σε σχέση με προηγούμενο experiment: κατεβάζεις τα πρόσθετα weights και ξεκινάς **νέο search** με την επιθυμητή λίστα. Δεν μεταβάλλεις το roster ενός παλιού run. Το `download-weights --profile ...` και το `doctor` διαβάζουν τη λίστα του config/profile, όχι κάποια ξεχωριστή επιλογή μοντέλων στο dashboard ή στο προηγούμενο CLI `--models`.

Προαιρετικό audit χωρίς training, που δημιουργεί ξεχωριστό προετοιμασμένο run:

```powershell
& $V3Python inspection.py audit --config $V3Config --profile workstation --task vig
```

Η υπάρχουσα `run` εντολή επίσης υπάρχει στο v3, αλλά για Optuna χρησιμοποίησε τις ρητές `search` εντολές ώστε να είναι σαφές τι ξεκινάς.

## 11. Pause, κλείσιμο, resume και χρονικά όρια

### Πριν φύγεις ή κλείσεις τον υπολογιστή

- Κλείσιμο μόνο της καρτέλας browser: δεν σταματά το search.
- Το dashboard εκκινεί ανεξάρτητο job, άρα το κλείσιμο του dashboard **δεν είναι αξιόπιστος τρόπος διακοπής training**.
- Για ελεγχόμενη παύση, πάτησε **Pause search** και περίμενε να εμφανιστεί η παύση. Μετά κλείσε dashboard/PC.
- Αν το search εκτελείται απευθείας σε terminal, χρησιμοποίησε `Ctrl+C` και περίμενε να επιστρέψει το prompt. Κρατιέται η τελευταία επιτυχώς αποθηκευμένη κατάσταση· δεν είναι εγγυημένο ότι σώζεται το τρέχον μισό epoch.
- Μην κλείνεις απότομα διεργασίες και μην κάνεις restart ως συνηθισμένο τρόπο pause.
- Όσο εκπαιδεύει, κράτησε τροφοδοσία και αποφυγή sleep/hibernate σύμφωνα με τις εταιρικές ρυθμίσεις. Δεν αλλάζουμε εδώ power policies ή Windows registry.

### Βρες το run και όρισέ το μία φορά

```powershell
Get-ChildItem -LiteralPath (Join-Path $V3Results 'vig') -Directory | Sort-Object Name -Descending | Select-Object Name, FullName
$V3Run = Read-Host 'Επικόλλησε το πλήρες path του συγκεκριμένου run, χωρίς εισαγωγικά'
if (-not (Test-Path -LiteralPath (Join-Path $V3Run 'config.json'))) { throw 'Αυτό δεν φαίνεται να είναι v3 run.' }
```

Επιλέγεις το σωστό run από τη λίστα ή από το dashboard, όχι αυτόματα το «τελευταίο». Μπορεί να έχεις δημιουργήσει και audit ή smoke run.

### Συνέχισε υπάρχον search

Μόνο όταν δεν υπάρχει ήδη ενεργή διεργασία που γράφει σε αυτό:

```powershell
& $V3Python inspection.py search --run $V3Run --device cuda
```

Εναλλακτικά, από το dashboard επίλεξε εκείνο το experiment και **Resume search**.

Το resume κρατά το ίδιο dataset/splits, μοντέλα, hyperparameters και checkpoints. Δεν προσθέτεις `--models` και δεν αλλάζεις το αποθηκευμένο config.

**Δεν δίνει νέο εξάωρο.** Χρησιμοποιεί μόνο το υπόλοιπο του αρχικού συνολικού budget. Αν αυτό τελείωσε, το resume δεν προσθέτει χρόνο. Η αλλαγή `project.yaml` δεν επεκτείνει το budget παλιού run. Για νέο training budget δημιουργείς νέο search. Μετά από απότομο shutdown μπορεί επίσης να χρεωθεί χρόνος της διακοπείσας συνεδρίας, σύμφωνα με το αποθηκευμένο ledger· προτίμησε Pause.

### Αν έχει μείνει εγκαταλελειμμένο lock

Έλεγξε πρώτα τις σχετικές διεργασίες, χωρίς να τις τερματίσεις μαζικά:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'inspection\.py' } | Select-Object ProcessId, ParentProcessId, CommandLine | Format-List
```

Μόνο αφού βεβαιωθείς ότι το συγκεκριμένο job έχει σταματήσει:

```powershell
& $V3Python inspection.py search --run $V3Run --device cuda --recover-lock
```

Μη διαγράφεις χειροκίνητα lock files και μη σκοτώνεις όλα τα `python.exe`: μπορεί να αφορούν άλλα projects.

### Τελείωσε το training budget αλλά έμεινε σύγκριση;

Αν υπάρχουν ολοκληρωμένοι υποψήφιοι και εκκρεμεί evaluation:

```powershell
& $V3Python inspection.py finalize-search --run $V3Run --device cuda --hours 0.5
```

Αυτό είναι **προαιρετικό πρόσθετο έως 30 λεπτών**, πέρα από το αρχικό search allowance. Δεν ξεκινά νέα training trials και δεν μετατρέπει ένα αποτυχημένο/ανολοκλήρωτο trial σε ολοκληρωμένο. Αν δεν θέλεις να ξεπεραστεί ο συνολικός χρόνος σου, μην το τρέξεις.

## 12. Αποτελέσματα, reports και σωστή ανάγνωση

Με `$V3Run` ορισμένο από την ενότητα 11, όταν δεν τρέχει άλλη μεταβαλλόμενη λειτουργία πάνω στο run:

```powershell
& $V3Python inspection.py status --run $V3Run
& $V3Python inspection.py report --run $V3Run
```

Για αναφορά των runs που βρίσκει το config:

```powershell
& $V3Python inspection.py report --config $V3Config --task vig
```

Άνοιξε το HTML path που τυπώνεται. Για τον πίνακα του συγκεκριμένου ολοκληρωμένου run:

```powershell
Start-Process -FilePath (Join-Path $V3Run 'comparison.html')
```

Αν λείπει το `comparison.html`, έλεγξε την κατάσταση/τα errors αντί να συμπεράνεις ότι ολοκληρώθηκε η σύγκριση. Το report δεν κάνει νέα εκπαίδευση. Για live παρακολούθηση προτίμησε το dashboard.

| Πεδίο | Πώς το διαβάζεις |
|---|---|
| Validation AP | Πόσο καλά κατατάσσει τις bad ψηλότερα. Δεν είναι accuracy ή ποσοστό σωστών αποφάσεων. |
| Bad escape rate | Ποιο ποσοστό πραγματικών bad περνά ως good. |
| Good reject rate | Ποιο ποσοστό πραγματικών good απορρίπτεται ως bad. |
| Balanced accuracy | Μέσος όρος σωστής αναγνώρισης των δύο κλάσεων. |
| `TARGET_NOT_MET` | Δεν πέτυχε τους καταγεγραμμένους στόχους· δεν σημαίνει τεχνικό crash. |
| `COMPLETE` | Ολοκληρωμένο trial, όχι εγγυημένα καλό ή production-approved μοντέλο. |
| `FAIL` / `FAILED` | Trial απέτυχε τεχνικά. Διαβάζεις `error` και `failure.json`. |
| `PRUNED` | Σταμάτησε από τον μηχανισμό pruning, όχι το ίδιο με crash. |
| `TIMEOUT_RESUMABLE` | Έληξε η χρονική μερίδα εκτέλεσης. Μπορεί να συνεχιστεί αν υπάρχει αποθηκευμένη κατάσταση και υπόλοιπο budget. |
| `RUNNING` state + `TIMEOUT_RESUMABLE` outcome | Ανολοκλήρωτο Optuna trial, όχι απαραίτητα το μοντέλο που καταναλώνει GPU αυτή τη στιγμή. |

Μη συγκρίνεις ένα AP validation με παλιό AP selection σαν να είναι το ίδιο benchmark. Συγκρίσεις θέλουν ίδιο σύνολο/ίδια labels και ξεκάθαρο decision rule.

## 13. Αν κάποιο trial γράφει FAILED — πάρε το ακριβές error

Σε δεύτερο PowerShell terminal, αφού ορίσεις τις μεταβλητές της ενότητας 9 και το `$V3Run` της ενότητας 11, αυτά είναι μόνο ανάγνωση αρχείων:

```powershell
Import-Csv -LiteralPath (Join-Path $V3Run 'trials.csv') | Format-List model, trial, state, outcome, value, seconds, error

Get-ChildItem -LiteralPath (Join-Path $V3Run 'search') -Recurse -File -Filter 'failure.json' | ForEach-Object {
    Write-Host $_.FullName
    Get-Content -LiteralPath $_.FullName -Raw
}

Get-Content -LiteralPath (Join-Path $V3Run 'model_state.json') -Raw
```

Τα trial errors βρίσκονται συνήθως στο `search\MODEL\trial_00000\failure.json`. Αν η αποτυχία έγινε πριν ξεκινήσει trial, μπορεί να υπάρχει μόνο στο operation log/terminal.

Logs jobs του dashboard:

```powershell
Get-ChildItem -LiteralPath (Join-Path (Split-Path -Parent $V3Config) 'dashboard_logs') -Filter '*.log' | Sort-Object LastWriteTime -Descending | Select-Object Name, FullName
$V3Log = Read-Host 'Path του συγκεκριμένου log, χωρίς εισαγωγικά'
Get-Content -LiteralPath $V3Log -Tail 100
```

Μπορείς να αντιγράψεις το error ως κείμενο, αφαιρώντας ευαίσθητα paths. Δεν χρειάζεται να ανεβάσεις εικόνες/δεδομένα. Τα αποτυχημένα trials που είδαμε στο laptop πρέπει να ελεγχθούν έτσι· δεν τα αποδίδουμε σε weights ή μνήμη χωρίς στοιχεία.

## 14. Threshold μετά το training — χωρίς νέα εκπαίδευση

Προϋπόθεση: να υπάρχει ολοκληρωμένος καταγεγραμμένος candidate. Η επιλογή mode δεν αλλάζει τι έμαθε το μοντέλο, αλλά τον κανόνα απόφασης.

| Mode | Τι κάνει |
|---|---|
| `policy` | Threshold από calibration, λαμβάνοντας υπόψη και τα δύο error targets. Αν είναι ανέφικτα, το fallback δεν αποτελεί επιτυχία στόχων. |
| `manual` | Ορίζεις αριθμητικά threshold. `score >= threshold` σημαίνει bad. |
| `argmax` | Για binary softmax classifiers αντιστοιχεί σε όριο 0.5, με ισοπαλία προς bad. Δεν ισχύει για PatchCore distances. |
| `scores_only` | Μόνο scores, χωρίς good/bad απόφαση. |

Διάλεξε **μία** εντολή ανά decision που θέλεις να δημιουργήσεις:

```powershell
& $V3Python inspection.py decision --run $V3Run --mode policy --output-root .\decisions
```

```powershell
& $V3Python inspection.py decision --run $V3Run --mode manual --threshold 0.65 --output-root .\decisions
```

```powershell
& $V3Python inspection.py decision --run $V3Run --mode argmax --output-root .\decisions
```

```powershell
& $V3Python inspection.py decision --run $V3Run --mode scores_only --output-root .\decisions
```

Το `0.65` είναι **παράδειγμα**, όχι προτεινόμενο βέλτιστο όριο για το μοντέλο σου. Υψηλότερο threshold δέχεται περισσότερα good αλλά μπορεί να αφήσει περισσότερα bad να περάσουν. Αλλαγή threshold δεν βελτιώνει τη διάκριση/ranking του μοντέλου.

Κάθε εντολή δημιουργεί νέο timestamped decision directory και τυπώνει path. Δεν πειράζεις το `selected.json`. Το slider στο explorer μόνο του δεν αλλάζει την αποθηκευμένη απόφαση για επόμενα classification jobs.

Για άλλον διαθέσιμο candidate αντί του selected:

```powershell
$V3Candidates = Get-Content -LiteralPath (Join-Path $V3Run 'candidates.json') -Raw | ConvertFrom-Json
$V3Candidates | Select-Object name
$V3Candidate = Read-Host 'Ακριβές candidate name από τη λίστα'
& $V3Python inspection.py decision --run $V3Run --candidate $V3Candidate --mode policy --output-root .\decisions
```

Για αποθηκευμένη σύγκριση υποθετικών thresholds μόνο στο calibration split:

```powershell
& $V3Python inspection.py threshold-review --run $V3Run --thresholds 0.30 0.50 0.65 --output-root .\threshold_review
```

## 15. Classification νέων εικόνων

Όρισε το πραγματικό `decision.json` που τυπώθηκε στην προηγούμενη ενότητα και τον φάκελο νέων εικόνων:

```powershell
$V3Decision = Read-Host 'Πλήρες path στο decision.json, χωρίς εισαγωγικά'
$V3Input = Read-Host 'Πλήρες path στον φάκελο νέων εικόνων, χωρίς εισαγωγικά'
if (-not (Test-Path -LiteralPath $V3Decision -PathType Leaf)) { throw 'Δεν βρέθηκε decision.json.' }
if (-not (Test-Path -LiteralPath $V3Input -PathType Container)) { throw 'Δεν βρέθηκε input folder.' }
```

### Χωρίς πραγματικά labels

```powershell
& $V3Python inspection.py classify --decision $V3Decision --input $V3Input --output-root .\classified --device cuda --copy-images
```

### Με πραγματικά good/bad labels

Ο φάκελος input περιέχει `good` και `bad` που έχουν επιβεβαιωθεί ανεξάρτητα, όχι φακέλους από τις προηγούμενες προβλέψεις ενός μοντέλου:

```powershell
& $V3Python inspection.py classify --decision $V3Decision --input $V3Input --output-root .\classified --device cuda --copy-images --labeled
```

Εκτελείς την κατάλληλη από τις δύο εντολές. Κάθε εκτέλεση γράφει νέο output, χωρίς να αλλάζει τα αρχικά images. Το `--copy-images` ζητά αντίγραφα· χωρίς αυτό χρησιμοποιείς κυρίως πίνακες προβλέψεων. Σε `scores_only` δεν υπάρχει ταξινόμηση σε good/bad folders. Χωρίς labels δεν μπορεί να υπολογιστεί accuracy ή ποσοστό λανθασμένων προβλέψεων.

## 16. Τελικό test, export και προαιρετικό activation

### 16.1 Τελική αξιολόγηση του αρχικού selected συστήματος

Μόνο αφού κλειδώσεις την επιλογή, όχι κάθε φορά για να ρυθμίζεις hyperparameters:

```powershell
& $V3Python inspection.py evaluate --run $V3Run --split final-test
```

Αυτό αξιολογεί το selected μοντέλο/threshold του run. **Δεν** αξιολογεί αυτομάτως κάποιο διαφορετικό manual `decision.json`. Για νέα decision χρησιμοποίησε ξεχωριστό φρέσκο labeled holdout με `classify --labeled` και κράτησε τα αποτελέσματα χωριστά.

Προαιρετικό νέο external holdout για το αρχικό selected σύστημα:

```powershell
$V3Holdout = Read-Host 'Path φρέσκου ανεξάρτητου labeled good/bad dataset'
& $V3Python inspection.py evaluate --run $V3Run --split external --input $V3Holdout
```

### 16.2 ONNX package

Πρώτα έχεις εγκαταστήσει τα export requirements της ενότητας 7. Για να εξάγεις την **explicit decision** που επέλεξες:

```powershell
$V3Package = Join-Path $V3Project ('packages\vig_onnx_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& $V3Python inspection.py package --run $V3Run --kind inference --format onnx --decision $V3Decision --output $V3Package
```

Αν θέλεις το αρχικό selected σύστημα, χρησιμοποίησε την ίδια εντολή **χωρίς** `--decision $V3Decision`, με νέο output directory. Η εξαγωγή ελέγχει συμφωνία scores/αποφάσεων εντός ανοχών. Αν αποτύχει, δεν αντιμετωπίζεις το μερικό output ως έτοιμο package.

Πρόβλεψη από το package σε CPU:

```powershell
$V3BatchOutput = Join-Path $V3Project ('classified\onnx_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& $V3Python inspection.py predict --model $V3Package --input $V3Input --output $V3BatchOutput --device cpu
```

Μεταφέρεις **ολόκληρο** το package directory, όχι μόνο το `.onnx`. Η παρούσα εγκατάσταση export χρησιμοποιεί `onnxruntime` CPU. GPU ONNX execution είναι διαφορετική προαιρετική εγκατάσταση, δεν προκύπτει μόνο επειδή έχεις PyTorch CUDA.

### 16.3 Activation — μόνο όταν υπάρχει επαρκής τεκμηρίωση

Δεν χρειάζεται activation για development classification μέσω explicit decision. Μόνο αν θέλεις να ορίσεις το αρχικό selected σύστημα ως ενεργό release και πληροί τους ελέγχους:

```powershell
& $V3Python inspection.py activate --run $V3Run --config $V3Config
```

Μπορεί σωστά να αρνηθεί λόγω στόχων, ελλιπούς σύγκρισης, pilot run ή ανεπαρκούς στατιστικής τεκμηρίωσης. Μην αλλάζεις targets σε `1.0` και μην απενεργοποιείς evidence checks απλώς για να περάσει. Αυτό δεν ενεργοποιεί αυτόματα ένα διαφορετικό manual decision.

## 17. Μεταφορά παλιού run από laptop και συνέχιση

Για **νέα** εκπαίδευση στον workstation αρκούν code, dataset, weights και νέο environment. Για να συνεχίσεις το **ίδιο** παλιό run:

1. Στο laptop κάνε Pause και περίμενε να σταματήσει.
2. Φτιάξε training package με τον interpreter του laptop, όχι του workstation.
3. Μετέφερε το πλήρες package directory και το ίδιο αρχικό dataset με τα ίδια περιεχόμενα, μέσω επιτρεπόμενου εταιρικού τρόπου.
4. Κάνε restore στον workstation.

Στο laptop, από τον δικό του φάκελο v3:

```powershell
$LaptopPython = Join-Path $env:USERPROFILE 'venvs\vip3\Scripts\python.exe'
$LaptopRun = Read-Host 'Πλήρες path στο paused laptop run'
$LaptopPackage = Join-Path (Get-Location) ('packages\training_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& $LaptopPython inspection.py package --run $LaptopRun --kind training --output $LaptopPackage
```

Στον workstation, αφού εκτελέσεις ξανά το session block της ενότητας 9:

```powershell
$V3TrainingPackage = Read-Host 'Path στο μεταφερμένο training package directory'
$V3Restored = Join-Path (Join-Path $V3Results 'vig') ('restored_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& $V3Python inspection.py restore --package $V3TrainingPackage --data-root $V3Data --output $V3Restored
```

Αν το restore ολοκληρωθεί και υπάρχει υπόλοιπο budget:

```powershell
& $V3Python inspection.py search --run $V3Restored --device cuda
```

Η αλλαγή device σε CUDA **δεν μετατρέπει** το laptop run σε workstation experiment 6 ωρών. Διατηρεί την αρχική διαμόρφωση και το αρχικό υπόλοιπο χρόνου. Για περισσότερα μοντέλα/6 ώρες δημιουργείς νέο workstation search.

Το training package του v3 περιλαμβάνει τα απαιτούμενα pretrained weights και την αποθηκευμένη κατάσταση του run, **όχι τις εικόνες του dataset ούτε το Python environment**. Δεν αντικαθιστά backup δεδομένων ή εγκατάσταση dependencies. Για καινούρια experiments εξακολουθείς να χρειάζεσαι weights στον φάκελο που ορίζει το νέο project configuration. Μην αντιγράψεις μεμονωμένο `best.pt` ή μόνο την SQLite βάση ενώ το job γράφει. Μην επεξεργάζεσαι χειροκίνητα `manifest.csv`, hashes και `selected.json`.

## 18. Προαιρετικά: νέα δεδομένα, σύγκριση runs και offline setup

### 18.1 Fine-tuning με νέα labeled δεδομένα

Η `update` δεν είναι το ίδιο με νέο Optuna search. Δημιουργεί νέα γενιά με lineage/holdout προστασίες. Το dataset πρέπει να διατηρεί τα απαιτούμενα παλιά records και να περιλαμβάνει τα νέα, όχι μόνο τα καινούρια images.

Φτιάξε ξεχωριστό config για την ενημέρωση. Αυτό αφήνει ανέγγιχτο το αρχικό:

```powershell
$V3UpdateConfig = Join-Path $V3Project ('project.update.' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.yaml')
Copy-Item -LiteralPath $V3Config -Destination $V3UpdateConfig
notepad.exe $V3UpdateConfig
```

Διόρθωσε το dataset path στο νέο config, αποθήκευσέ το και μόνο τότε:

```powershell
& $V3Python inspection.py update --config $V3UpdateConfig --profile workstation --task vig --from-run $V3Run
```

### 18.2 Σύγκριση δύο runs

```powershell
$V3OldRun = Read-Host 'Path παλιού run'
$V3NewRun = Read-Host 'Path νέου run'
$V3CompareOutput = Join-Path $V3Project ('comparisons\compare_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& $V3Python inspection.py compare --old-run $V3OldRun --new-run $V3NewRun --output $V3CompareOutput
```

Χρειάζονται κοινά κατάλληλα selection records και συνεπείς ετικέτες. Αν δεν υπάρχουν, η άμεση σύγκριση δεν είναι έγκυρη και η εντολή μπορεί να αρνηθεί. Μη χρησιμοποιείς το test για συνεχή tuning.

### 18.3 Offline setup — μόνο αν δεν επιτρέπεται internet στον workstation

Ετοίμασε πακέτα σε άλλο **Windows x64 / Python 3.12** μηχάνημα, με τη σχετική εταιρική άδεια. Δεν χρειάζεται dataset για αυτό. Χρησιμοποίησε καθαρό αποκλειστικό `wheelhouse_cuda`, όχι μικτό CPU/GPU wheelhouse.

Στο online μηχάνημα, με `$V3Python` που δείχνει στη δική του Python 3.12:

```powershell
$V3Wheelhouse = Join-Path $V3Project ('wheelhouse_cuda_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& $V3Python -m pip download --only-binary=:all: --dest $V3Wheelhouse pip
& $V3Python -m pip download --only-binary=:all: --dest $V3Wheelhouse -r .\requirements.txt -r .\requirements-export.txt
& $V3Python -m pip download --only-binary=:all: --dest $V3Wheelhouse torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

Κάθε download πρέπει να πετύχει. Μετέφερε το wheelhouse, τον κώδικα και τα pretrained weights. Python και NVIDIA driver πρέπει ήδη να είναι εγκατεστημένα/εγκεκριμένα στον offline workstation.

Στον offline workstation ορίζεις τις μεταβλητές της ενότητας 3 και, αντί για το online install block, τρέχεις:

```powershell
$V3Wheelhouse = Read-Host 'Path στο μεταφερμένο CUDA wheelhouse'
& {
    $ErrorActionPreference = 'Stop'
    if (Test-Path -LiteralPath $V3Env) { throw 'Διάλεξε νέο αποκλειστικό venv directory.' }
    py -3.12 -m venv $V3Env
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed.' }
    & $V3Python -m pip install --no-index --find-links $V3Wheelhouse --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Offline pip setup failed.' }
    & $V3Python -m pip install --no-index --find-links $V3Wheelhouse torch==2.7.1 torchvision==0.22.1 -r .\requirements.txt -r .\requirements-export.txt
    if ($LASTEXITCODE -ne 0) { throw 'Λείπουν ή δεν ταιριάζουν offline wheels.' }
    & $V3Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency conflict.' }
}
```

Μετά κάνεις GPU test, paths, doctor, smoke και dashboard όπως πριν. Το `--no-index` αποτρέπει fallback σε online package index. Η offline εγκατάσταση δεν κατεβάζει missing weights.

## 19. Συχνά προβλήματα και ασφαλής αντιμετώπιση

| Πρόβλημα | Τι κάνεις |
|---|---|
| `python` δείχνει 3.14 ή άλλο venv | Μην χρησιμοποιείς σκέτο `python`. Έλεγξε `& $V3Python --version` και `& $V3Python -c "import sys; print(sys.executable)"`. |
| `No matching distribution` για torch | Έλεγξε Python 3.12 x64 και τον επίσημο CUDA index. Μην αναμειγνύεις τυχαίες εκδόσεις torch/torchvision. |
| Μεγάλο path / missing sklearn `.lib` στην εγκατάσταση | Κράτησε σύντομο ξεχωριστό venv όπως στον οδηγό. Ένα τέτοιο μήνυμα δεν αποδεικνύει μόνο του την αιτία· κράτησε το πλήρες error αν επιμένει. Δεν χρειάζεται διαγραφή άλλων environments. |
| `CUDA unavailable`, `+cpu`, `torch.version.cuda is None` | Ελέγχεις ότι χρησιμοποιείς το νέο GPU interpreter και το `cu128` wheel. Δεν εγκαθιστάς CPU wheel πάνω στο GPU environment για να «ξεκινήσει». |
| `no kernel image`, unsupported architecture ή driver error | Στέλνεις GPU model, driver, torch version/capability στο IT ή για διάγνωση. Δεν αρκεί η χωρητικότητα VRAM. |
| `DLL load failed`, `torchvision::nms` error | Πιθανή ασυμβατότητα binaries/runtime. Έλεγξε ακριβές ζευγάρι, imports, `pip check` και το GPU test. Μην κάνεις τυφλό upgrade όλων των πακέτων. |
| Certificate / proxy / permission error | Ζητάς εγκεκριμένη ρύθμιση από IT ή offline wheelhouse. Όχι `--trusted-host`, απενεργοποίηση TLS ή antivirus. |
| `Missing LOCAL pretrained weights` | Έλεγξε resolved `weights_root` και κατέβασε/μετέφερε τα weights για όλα τα επιλεγμένα μοντέλα. |
| Checksum mismatch σε weight | Μην το χρησιμοποιήσεις. Κράτησε το error και ζήτησε καθαρό σωστό αρχείο· μην παρακάμπτεις τον έλεγχο. |
| `CUDA out of memory` | Σταμάτα το συγκεκριμένο search ελεγχόμενα. Για **νέο** experiment μείωσε το microbatch/eval batch, ή διάλεξε λιγότερο απαιτητικό μοντέλο. Μην πειράξεις native εικόνες ή stored run config. |
| `FAILED` | Διάβασε την ενότητα 13. Δεν σημαίνει «χαμηλό accuracy» και δεν θεωρούμε δεδομένο ότι θα διορθωθεί μόνο του. |
| `TIMEOUT_RESUMABLE` | Δεν είναι crash. Χρειάζεται υπόλοιπο budget και αποθηκευμένο checkpoint. Μην ξεκινήσεις παράλληλο resume όταν ήδη τρέχει search. |
| `TARGET_NOT_MET` | Τεχνικά μπορεί να ολοκληρώθηκε. Η απόδοση δεν πέτυχε τους στόχους· χρειάζεται αξιολόγηση δεδομένων/μοντέλου, όχι reinstall. |
| Audit: ίδια εικόνα σε good και bad | Διόρθωσε τις αληθινές ετικέτες, με backup των δεδομένων. Μη διαγράψεις απλώς το audit check. |
| Audit: unsupported bit depth/size ή grouping | Διάβασε το συγκεκριμένο μήνυμα. Μην παρακάμπτεις με τυχαίο crop, normalization ή αλλαγή filenames. |
| `Run is busy` / lock | Έλεγξε αν πράγματι υπάρχει ενεργό job. Χρησιμοποίησε `--recover-lock` μόνο όταν έχει σταματήσει. |
| `ready: true` αλλά αργότερα failure | Το doctor δεν είναι εξαντλητική δοκιμή. Κράτησε το traceback του trial και δοκίμασε το synthetic smoke πριν νέο μεγάλο run. |

Για νέο memory-reduced workstation experiment, επεξεργάζεσαι **τα υπάρχοντα** πεδία `profiles.workstation.training` στο configuration, όχι αποθηκευμένο run:

```yaml
batch_size: 1
eval_batch_size: 1
accumulation_steps: 16
```

Αυτό διατηρεί το baseline effective batch στο 16, αλλά δεν μειώνει τη μνήμη που απαιτεί μία μόνο εικόνα. Το Optuna εξακολουθεί να ψάχνει effective batch σύμφωνα με το search space. Αν ούτε μία εικόνα χωρά, χρειάζεται διαφορετικό μοντέλο/μηχάνημα ή ρητή νέα μεθοδολογική απόφαση — όχι σιωπηλό downsampling.

### Αν η θύρα dashboard χρησιμοποιείται ήδη

Πρώτα δοκίμασε αν υπάρχει ήδη dashboard στο `http://127.0.0.1:8765`. Μην ξεκινήσεις δεύτερο training job. Μόνο αν χρειάζεσαι διαφορετική θύρα για το dashboard:

```powershell
& $V3Python inspection.py dashboard --config $V3Config --port 8766
```

Άνοιξε `http://127.0.0.1:8766`. Η διαφορετική θύρα δεν δημιουργεί ανεξάρτητα ασφαλή experiments πάνω στο ίδιο run.

### Προαιρετική ζωντανή παρακολούθηση GPU

Σε δεύτερο terminal:

```powershell
nvidia-smi -l 2
```

`Ctrl+C` σε αυτό το terminal σταματά μόνο την παρακολούθηση. Δεν αλλάζεις ή τερματίζεις το training από εκεί.

## 20. Καταγραφή περιβάλλοντος και help όλων των εντολών

Μετά από επιτυχημένο setup κράτησε snapshot για αναπαραγωγή/διάγνωση. Τα requirements του project περιέχουν ranges για αρκετά πακέτα· δεν αποτελούν πλήρες environment lock.

```powershell
$V3SetupRecord = Join-Path $V3Project ('setup_records\setup_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $V3SetupRecord | Out-Null
& $V3Python -m pip freeze | Out-File -LiteralPath (Join-Path $V3SetupRecord 'requirements_snapshot.txt') -Encoding utf8
& $V3Python -c "import sys,torch,torchvision; print(sys.version); print(sys.executable); print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda)" | Out-File -LiteralPath (Join-Path $V3SetupRecord 'python_torch.txt') -Encoding utf8
nvidia-smi | Out-File -LiteralPath (Join-Path $V3SetupRecord 'nvidia-smi.txt') -Encoding utf8
```

Πριν κοινοποιήσεις logs έλεγξε για προσωπικά/εταιρικά paths. Μην ανεβάζεις dataset ή credentials. Το snapshot δεν εγγυάται ακριβώς ίδιους floating-point υπολογισμούς σε άλλο hardware.

Για τα flags της δικής σου εγκατεστημένης έκδοσης:

```powershell
& $V3Python inspection.py --version
& $V3Python inspection.py --help
& $V3Python inspection.py doctor --help
& $V3Python inspection.py search --help
& $V3Python inspection.py dashboard --help
& $V3Python inspection.py decision --help
& $V3Python inspection.py classify --help
& $V3Python inspection.py evaluate --help
& $V3Python inspection.py package --help
& $V3Python inspection.py restore --help
& $V3Python inspection.py update --help
```

## 21. Η καθημερινή ελάχιστη εντολή

Αφού ολοκληρωθεί η πρώτη εγκατάσταση, για άνοιγμα του dashboard αρκεί:

```powershell
Set-Location -LiteralPath (Join-Path $env:USERPROFILE 'vip3')
& (Join-Path $env:USERPROFILE 'venvs\vip3gpu\Scripts\python.exe') inspection.py dashboard --config .\project.yaml
```

Μέσα στο dashboard: **VIG → workstation → μοντέλα → έως 6 ώρες → Start Optuna search**. Αν έχεις paused run με χρόνο που απομένει, διαλέγεις **Resume search** αντί για νέο search.

---

### Βάση και όρια του οδηγού

Οι project-specific εντολές διασταυρώθηκαν με τα `inspection.py`, `nilt/control.py`, `nilt/search.py`, `nilt/dashboard.py`, `nilt/registry.py`, `download_weights.py`, `scripts/smoke_v3.py`, requirements και τον οδηγό v3. Έγινε στατικός έλεγχος 47 παραδειγμάτων CLI με τον πραγματικό argument parser, καθώς και έλεγχος σύνταξης των ενσωματωμένων Python/YAML παραδειγμάτων. Δεν εκτελέστηκε PowerShell από εδώ. Οι επίσημες πηγές εγκατάστασης παρατίθενται δίπλα στα σχετικά σημεία.

Δεν εκτελέστηκε εγκατάσταση στον δικό σου Windows workstation από εδώ και δεν έχει επιβεβαιωθεί το ακριβές μοντέλο της GPU σου. Γι' αυτό τα GPU/doctor/smoke checks είναι ουσιαστικά βήματα, όχι προαιρετικές διακοσμητικές εντολές. Επιτυχής εγκατάσταση δεν εγγυάται καλή ταξινόμηση: αυτή πρέπει να τεκμηριωθεί σε αντιπροσωπευτικά, σωστά labeled και ανεξάρτητα δεδομένα.

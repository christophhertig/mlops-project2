# mlops-project2
HS23 MLOps Project 2  
Willkomen zu meinem GitHub Repository von MLOps Project 2. Im Repository finden Sie die Dateien, welche für das Projekt erarbeitet wurden, Strukturiert nach den einzelnen Tasks des Projektes. Das Jupyter Notebook "MLOPS_DistilBERT_MRPC_ChristophHerzog.ipynb" wurde im Rahmen von Projekt 1 erstellt und diente als Grundlage für die folgenden Arbeiten.  
Bei Fragen oder Unklarheiten stehe ich gerne zur Verfügung.
## Task 1: Jupyter Notebook zu Python Skript
Das Python Skript src/main.py ist die angepasste Version des Jupyter Notebooks "MLOPS_DistilBERT_MRPC_ChristophHerzog.ipynb". Beim Ausführen von main.py werden die Ergebnisse in Weights and Biases erfasst. Um das Skript auszuführen, müssen die Packages aus requirements_without_jupyter.txt installiert werden.  
Beim Ausführen von main.py können diverse Parameter definiert werden. In der Tabelle "Tabelle für Parameterübergabe" sind die Werte inkl. benötigtem Befehl aufgelistet. Wenn ein Parameter beim Ausführen von main.py nicht definiert, wird der default-Wert gesetzt. Beispiel für das Ausführen von main.py:  

*python main.py --api_key [personal API-Key] --wandb_projectname MLOPS_Project2 --save_path notebook/checkpoint --learning_rate 2.84468e-5 --warmup_steps 209.9549193 --optimizer_choice adam*

## Task 2: Docker Image
Das Dockerfile mlops_final wurde auf der Grundlage von main.py erstellt. Die Konfiguration für das Dockerfile ist in der Datei Dockerfile ersichtlich. Um das Docker Image lokal zu Verwenden, gibt es zwei Möglichkeiten:
### Image laden per Docker Hub
Das Docker Image kann über Docker Hub bezogen werden. Dazu muss folgender Befehl ausgeführt werden: *docker pull christoph01/mlops_final:latest*
### Image laden per mlops_final.tar File
Das Docker Image kann auch über die Datei mlops_final.tar geladen werden. Das Image hat eine Grösse von 1.44GByte und kann über diesen Link heruntergeladen werden (Zugriffsrechte nur mit HSLU-Konto): https://hsluzern-my.sharepoint.com/:u:/g/personal/christoph_herzog_01_stud_hslu_ch/ESJyD0-cFYdCtvwYQR-ss_MB6RmOT3cVUtTe-VRqrL61tg?e=pzc7bV  

Um das Image zu Verwenden muss folgender Befehl ausgeführt werden: *docker load -i mlops_final.tar*

### Image ausführen
Wenn das Image korrekt geladen wurde, kann es mit auf dem lokalen Gerät ausgeführt werden. Für das setzen der Parameter kann wieder die unte stehende Tabelle zu Hilfe genommen werden. Beispiel für das Ausführen des Images:

*docker run -e API_KEY=[personal API-Key] -e WAND_PROJECTNAME=MLOPS_Project2 -e SAVE_PATH=notebook/checkpoint -e LEARNING_RATE=2.84468e-5 -e WARMUP_STEPS=209.9549193 -e OPTIMIZER_CHOICE=adam christoph01/mlops_final:latest*

## Tabelle für Parameterübergabe

| Befehl main.py | Befehl Docker run                | Beschreibung                                      | Default-Wert       |
|------------------------|---|--------------------------------------------------|--------------------|
| `--api_key`            | `API_KEY` | Persönlicher API-Key von Weights & Biases         | -                  |
| `--wandb_projectname`  | `WAND_PROJECTNAME` | Name des Projektes in Weights & Biases            | MLOPS_Project2     |
| `--save_path`          | `SAVE_PATH` | Pfad, wo die Checkpoints gespeichert werden sollen | ./checkpoint      |
| `--learning_rate`      | `LEARNING_RATE` | Learning Rate für das Training                    | 2.84468e-5         |
| `--warmup_steps`       | `WARMUP_STEPS` | Anzahl der Warmup Steps für das Training          | 209.9549193        |
| `--optimizer_choice`   | `OPTIMIZER_CHOICE` | Wahl des Optimizers (Optionen: adam, sgd)               | 'adam'             |

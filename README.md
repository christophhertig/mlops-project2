# mlops-project2
HS23 MLOps Project 2  
Willkomen zu meinem GitHub Repository von MLOps Project 2. Im Repository finden Sie die Dateien, welche für das Projekt erarbeitet wurden, Strukturiert nach den einzelnen Tasks des Projektes.
## Task 1: Jupyter Notebook zu Python Skript
Das Jupyter Notebook "BLABLA" wurde im Rahmen von Projekt 1 erstellt. Dieses war die Grundlage von den nachfolgenden Dateien. Das Python Skript main.py ist die angepasste Version des Jupyter Notebooks. Beim Ausführen von main.py werden die Ergebnisse in Weights and Biases erfasst. Um das Skript auszuführen, müssen die Packages aus requirements_without_jupyter.txt installiert werden.  

Beim Ausführen von main.py können folgende Parameter übergeben werden:
| Befehl                | Beschreibung                                      | Default-Wert       |
|------------------------|--------------------------------------------------|--------------------|
| `--api_key`            | Persönlicher API-Key von Weights & Biases         | -                  |
| `--wandb_projectname`  | Name des Projektes in Weights & Biases            | MLOPS_Project2                  |
| `--save_path`          | Pfad, wo die Checkpoints gespeichert werden sollen | ./checkpoint                  |
| `--learning_rate`      | Learning Rate für das Training                    | 2.84468e-5                  |
| `--warmup_steps`       | Anzahl der Warmup Steps für das Training          | 209.9549193        |
| `--optimizer_choice`   | Wahl des Optimizers (Optionen: adam, sgd)               | 'adam'             |
  
Wenn ein Parameter beim Ausführen von main.py nicht definiert, wird der default-Wert gesetzt. Beispiel für das Ausführen von main.py:  

*python main.py --api_key [personal API-Key] --wandb_projectname MLOPS_Project2 --save_path notebook/checkpoint --learning_rate 2.84468e-5 --warmup_steps 209.9549193 --optimizer_choice adam*

## Task 2: Docker Image
Das Dockerfile mlops_final wurde auf der Grundlage von main.py erstellt. Die Konfiguration für das Dockerfile ist in der Datei Dockerfile ersichtlich. Um das Docker Image lokal zu Verwenden, gibt es zwei Möglichkeiten:
### Docker Hub
Das Docker Image kann über Docker Hub bezogen werden. Dazu muss folgender Befehl ausgeführt werden: *docker pull mlopsproject2/mlops_final:latest*
### Image mlops_final.tar
Das Docker Image kann auch über die Datei mlops_final.tar geladen werden. Dazu muss folgender Befehl ausgeführt werden: *docker load -i mlops_final.tar*

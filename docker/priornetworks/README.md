## Installation

`python3` und `pip` wird benötigt.

Installation der Dependencies (am besten in einem Virtual Environment):

`$ pip install -r requirements.txt`

## Vorbereiten der Datensätze

`CIFAR10` und `SVHN` werden von Pytorch automatisch heruntergeladen.
`Random` wird vom Programm selbst generiert. 

Lediglich `TIM` muss manuell heruntergeladen werden:

```
$ mkdir -p resources/data/TIM
$ cd resources/data/TIM
$ wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
$ unzip tiny-imagenet-200.zip
$ rm tiny-imagenet-200.zip
```

## Nutzung

### Trainieren eines neuen Netzwerks:

`$ python3 src/main_train_new_network.py --name NAME [optional arguments]`

Auflistung der optionalen Argumente:

`$ python3 src/main_train_new_network.py -h`

Nach der Ausführung sind folgende Dateien erstellt:
* `resources/model/{name}_initial_model_{timestamp}.tar` - Model vor dem Training
* `resources/model/{name}_trained_model_{timestamp}.tar` - Model nach dem Training
* `resources/json/{name}.json` - Ergebnisse im Epochenverlauf

### Testen eines fertig trainierten Netzwerks:

`$ python3 src/main_test_old_network.py --name NAME --model_name MODEL_NAME [optional arguments]`

Auflistung der optionalen Argumente:

`$ python3 src/main_test_old_network.py -h`

### Anzeigen von Alpha-Werten:

`src/main_show_simplex` ist fast identisch zu `src/main_test_old_network.py`, nur dass im Output zusätzlich einige Simplex-Werte ausgegeben werden.

### Erstellen von Graphen im Epochenverlauf

`$ python3 src/main_create_graphics.py`

`src/main_create_graphics.py` nutzt die Dateien aus `resources/json`. Damit werden die Graphen in `resources/graphics` erstellt.
 
 ### Auswertung unterschiedlicher OOD-Datensätze beim Trainieren und Testen
  
`$ python3 src/main_calc_results.py`
  
`resources/stdout` enthält Auszüge des `stdout` verschiedener Ausführungen von `src/main_test_old_network.py`. Format der Dateien: `{OOD-Trainingsdatensatz}-{OOD-Testdatensatz}.txt`.

Wesentliche Ergebnisse wurden in `src/main_calc_results.py` als Dictionary übertragen.

Daraus werden Durchschnitte und Standardabweichungen der Ergebnisse von Experimenten mit denselben genutzten Datensätzen berechnet und ausgegeben.

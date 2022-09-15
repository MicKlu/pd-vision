Kod źródłowy algorytmów wizyjnego zliczania obiektów.

Uruchomienie graficznego interfejsu użytkownika odbywa się poprzez wydanie polecenia:

```sh
$ python main.py
```

Do prawidłowego działania wymagane są biblioteki wymienione w pliku `requirements.txt` oraz OpenCV.

```
matplotlib==3.5.1
numpy==1.23.3
PyQt5==5.15.7
```

Właściwy kod źródłowy algorytmów znajduje się w katalogu `alg`:
- `__init__.py` – wspólny kod dla obu wersji algorytmu,
- `custom.py` – kod modyfikacji algorytmu,
- `ref.py` – kod algorytmu bazowego.

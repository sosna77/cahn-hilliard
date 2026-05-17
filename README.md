# Symulacja Równania Cahna-Hilliarda 2D

Numeryczny solver równania Cahna-Hilliarda modelujący zjawisko rozkładu spinodalnego i koalescencji faz. Wykorzystuje pół-niejawną metodę spektralną (Semi-Implicit Spectral Method) oraz kompilator NJIT (Numba) do maksymalizacji wydajności obliczeniowej. Projekt zawiera interfejs graficzny pozwalający na obserwację ewolucji domen na żywo oraz monitorowanie spadku całkowitej energii swobodnej.

## Wymagania

Do uruchomienia projektu wymagany jest **Python 3.8+**.
Zaleca się utworzenie wirtualnego środowiska (`.venv`).

Instalacja wymaganych bibliotek:

```bash
pip install numpy numba PySide6 pyqtgraph

```

## Uruchomienie

Symulację uruchamia się bezpośrednio z głównego pliku w katalogu root projektu:

```bash
python main.py

```

## Konfiguracja modelu

Wszystkie parametry fizyczne i algorytmiczne są odseparowane i znajdują się w pliku `config.py`. Modyfikacja tego pliku pozwala na badanie różnych reżimów termodynamicznych.

**Kluczowe parametry:**

* `N` – rozdzielczość siatki 2D (np. 256). Dla optymalizacji FFT zaleca się potęgi liczby 2.
* `L` – fizyczny rozmiar domeny symulacyjnej.
* `DT` – krok czasowy całkowania algorytmu (zależny od sztywności równania).
* `m0` – średnia magnetyzacja (od `-1.0` do `1.0`). Wartość `0.0` generuje struktury labiryntowe, np. `0.3` faworyzuje układ kropelkowy.
* `alpha` – kontroluje szerokość granicy faz ($\epsilon = \alpha \cdot \Delta x$). Dla stabilności numerycznej i braku artefaktów aliasowania, zachowaj $\alpha \ge 1.5$.
* `noise_amp` – amplituda szumu początkowego (fluktuacji termicznych).

## Struktura plików

* `main.py` – punkt wejściowy inicjalizujący aplikację Qt.
* `src/config.py` – definicja klasy `SimConfig` z parametrami.
* `src/gui.py` – interfejs graficzny, rysowanie siatki (`pyqtgraph.ImageItem`) i wykresu energii.
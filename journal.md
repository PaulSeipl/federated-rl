# Wokring Journal

## 23. Nov

### Stand

Der erste Ansatz mit federated Reinforcement learning ist implementiert. 5 Worker trainieren auf unterschiedlichen rooms Domänen, die mit den weights des _mainAgents_ initsialisiert werden. Nach X Episoden werden deren weights (state dicts) aggregiert und der **Durchschnitt** genommen. Mit diesem Durchschnitt wird der _mainAgent_ gespeist. Dieser durchlauf passiert X mal. Zum Schluss wird der **main Agent** anhand einer _TestDomäne_ im interference mode getestet.
Die Returns der worker werden momentan nicht weiter verarbeitet.

### Rooms

#### Worker

```txt
v0                  v1                  v2                  v3                  v4
# # # # # # # # #   # # # # # # # # #   # # # # # # # # #   # # # # # # # # #   # # # # # # # # #
# S . . # . . . #   # . . . # . . . #   # . . . # . . S #   # . . . # . . S #   # G . . # . . . #
# . . . # . . . #   # . . . # . . . #   # . . . # . . . #   # . . . # . . . #   # . . . # . . . #
# . . . . . . . #   # . . . . . . . #   # . . . . . . . #   # . . . . . . . #   # . . . . . . . #
# . . . # # . # #   # . . S # # . # #   # . . . # # . # #   # . . . # # . # #   # . . . # # . # #
# . # # # . . . #   # . # # # G . . #   # . # # # . . . #   # . # # # . . . #   # . # # # . . . #
# . . . # . . . #   # . . . # . . . #   # . . . # . . . #   # . . . # . . . #   # . . . # . . . #
# . . . . . . G #   # . . . . . . . #   # . . . . . . G #   # G . . . . . . #   # . . . . . . S #
# # # # # # # # #   # # # # # # # # #   # # # # # # # # #   # # # # # # # # #   # # # # # # # # #
```

#### Testing

```txt
t0
# # # # # # # # #
# . . . # . . G #
# . . . # . . . #
# . . . . . . . #
# . . . # # . # #
# . # # # . . . #
# . . S # . . . #
# . . . . . . . #
# # # # # # # # #
```

### Durchläufe

| Episoden pro Worker | Updates nach Episoden |
| ------------------- | --------------------- |
| 1000                | 100                   |
| 2500                | 500                   |
| 3000                | 1000                  |

### Beobachtung

- Worker tuen sich leicht bei v0, v2, v3
- Worker tut schwer bei v1
- Worker tut sich sehr schwer (erreicht fast nie das Ziel) bei v4
- Nach jedem **update** des **mainAgents** laufen v0, v2, v3 besser, jedoch v1 und v4 schlechter
- MainAgent erreicht kein mal das Ziel beim Test von t0

### Diskussion/Vermutung

- Ungleichgewicht in den Daten/Räumen (bei der Mehrheit der Räumen beginnt der Agent oben und das Ziel ist unten -> es lohnt sich fast immer nach unten zu laufen)
- Einfach den Durchschnitt nehmen ist nicht ausreichend

### TODO

- Returns der Worker als Plot speichern
- Im Plott anzeigen, wann das Netz geupdated wurde (eine vertikale Linie)
- State dicts der Worker unterschiedlich gewichten (z.B. Worker, die seltener das Ziel erreichen, höher gewichten)
- Main Agent Anhand von Räumen(~10 Räume) mit random S und G Punkten im interference modus testen
- Main Agent nach jedem **Update** testen und einen Plot erstellen
- Sachen aufschreiben wenn ich was Teste. Alle Daten sind relevant!!

## 9. Feb

### Stand

Es arbeiten immer so viele Worker, wie rooms im gegebenen Folder vorhanden sind. Plots werden jetzt für die Worker
erstelllt. Ich habe eine gewisse Balance für die Räume und neue Räume hinzugefügt (Laufrichtungen).
Mit den neuen Räumen schafft der **main Agent** zum Schluss auch ab und zu den Testraum.

### Rooms

Neuer Room für Folder 9_9_4:

```txt
v5
# # # # # # # # #
# . . . # . . G #
# . . . # . . . #
# . . . . . . . #
# . . . # # . # #
# . # # # . . . #
# . . . # . . . #
# . . . . . . S #
# # # # # # # # #
```

Neuer Rooms Folder 9_9_4_test

```txt
sDg                 sLg                 sDRg                sRg                 sUg
# # # # # # # # #   # # # # # # # # #   # # # # # # # # #   # # # # # # # # #   # # # # # # # # #
# . . . # . . S #   # G . . # . . S #   # S . . # . . . #   # S . . # . . G #   # . . . # . . G #
# . . . # . . . #   # . . . # . . . #   # . . . # . . . #   # . . . # . . . #   # . . . # . . . #
# . . . . . . . #   # . . . . . . . #   # . . . . . . . #   # . . . . . . . #   # . . . . . . . #
# . . . # # . # #   # . . . # # . # #   # . . . # # . # #   # . . . # # . # #   # . . . # # . # #
# . # # # . . . #   # . # # # . . . #   # . # # # . . . #   # . # # # . . . #   # . # # # . . . #
# . . . # . . . #   # . . . # . . . #   # . . . # . . . #   # . . . # . . . #   # . . . # . . . #
# . . . . . . G #   # . . . . . . . #   # . . . . . . G #   # . . . . . . . #   # . . . . . . S #
# # # # # # # # #   # # # # # # # # #   # # # # # # # # #   # # # # # # # # #   # # # # # # # # #

sULg                sULg(reverseDefault)  sURg
# # # # # # # # #   # # # # # # # # #     # # # # # # # # #
# . . . # . . . #   # G . . # . . . #     # . . . # . . . #
# . . . # . . . #   # . . . # . . . #     # . . . # . . . #
# . . . . . . . #   # . . . . . . . #     # . . . . . . . #
# . . . # # S # #   # . . . # # . # #     # . . . # # G # #
# G # # # . . . #   # . # # # . . . #     # S # # # . . . #
# . . . # . . . #   # . . . # . . . #     # . . . # . . . #
# . . . . . . . #   # . . . . . . S #     # . . . . . . . #
# # # # # # # # #   # # # # # # # # #     # # # # # # # # #

```

### Testing

Unverändert

### Durchläufe

| ID | Episoden pro Worker | Updates nach Episoden | Folder      |
| -- | ------------------- | --------------------- | ----------- |
| 1  | 5000                | 100                   | 9_9_4       |
| 2  | 10000               | 500                   | -           |
| 3  | 5000                | 1000                  | -           |
| 4  | 10000               | 500                   | 9_9_4_test  |

### Beobachtung

#### Lauf 1 (ohne v5)

- Bei Lauf 1 werden v0,v2,v3 besser oder bleiben gleich gut; v4 bekommt nach dem dritten **Update** gar nichts mehr geschissen; v1 wird schlechter und erreicht zum Schluss, wie der **main Agent** beim Testen nicht das Ziel

#### Lauf 2 (ohne v5)

- Ähnlich wie Lauf 1 nur alles intensiver. Bei den Räumen, wo man nach unten gehen muss, um das Ziel zu erreichen (v0,v2,v3) werden immer schneller bewältigt, während in den beiden anderen Räumen (v1,v4) nur selten oder gar nicht (vor allem ab dem 11. **Update**) das Ziel erreicht wird.
- Auch zu beobachten ist, dass nach einem Update meistens ein paar Episoden benötigt werden, bis das Ziel wieder erreicht wird.
- **main Agent** erreicht kein mal das Ziel

#### Lauf 3

- v2 und v3 werden nach jedem **Update** besser; v0 und **v1** werden ein bisschen besser
- v4 wird bis zum dritten **Update** ein bisschen besser, aber nur in den späteren Episoden. Direkt nach den **Updates** braucht "er" immer länger um sich zu "fangen". Nach dem viertem **Update** wird ca. 500 Episoden nicht das Ziel erreicht und danach manchmal
- v5 braucht nach jedem **Update** immer länger um sich zu "fangen", bis er nach dem 4.(letztem) **Update** kein mal mehr das Ziel erreciht

#### Lauf 4

- **main Agent** hat ab und zu das Ziel erreicht (im eval modus)
- sLg und sDg werden ingesamt immer besser mit kleineren Hängern zwischen durch
- das weiter zu aufzuschreiben macht wahrscheinlich nur für plots Sinn, die ich klar analysieren, wo ich was rauslesen kann

### Diskussion/Vermutung

- die Agenten, die vor dem ersten **Update** zufällig oder auch wegen einem einfacherem Raum schnell das Ziel finden und danach lernen das Ziel immer schnellerer zufinden, werden vorraussichtlich über die Updates bis zum Schluss immer besser/sicherer
- es hat einen starken Einfluss, nach wie vielen Episoden man die Updates macht
- eine Balance der Räume bringt zum Teil was aber zum Teil auch nicht
- die Räume die von Anfang gut abschließen bleiben auch meisten gut, wahrscheinlich weil sie jeden "run" ihre Parameter festigen können anstatt wieder zufällig das Ziel zu finden.
- ??Zum Teil scheints so als ob der Agent abhängig von seiner Startposition entscheidet wie er gehen wird??
- Einfach den Durchschnitt nehmen ist nicht ausreichend

### TODO

- ~~Returns der Worker als Plot speichern~~
- ~~Im Plott anzeigen, wann das Netz geupdated wurde (eine vertikale Linie)~~
- ~~THOMY SCHREIBEN!!!!!!!!~~
- ~~Main Agent nach jedem **Update** testen und einen Plot erstellen~~
- State dicts der Worker unterschiedlich gewichten (z.B. Worker, die seltener das Ziel erreichen, höher gewichten)
- Main Agent Anhand von Räumen(~10 Räume) mit random S und G Punkten im interference modus testen
- Sachen aufschreiben wenn ich was Teste. Alle Daten sind relevant!!
- <https://arxiv.org/pdf/2108.11887.pdf> lesen

## 16. Februar

### Stand

Neuen Weg hinzugefügt, wie die Modelle zusammengefügt werden. Je öfter der Agent das Ziel nicht erreicht hat, desto mehr werden seine weights wahrgenommen/miteinberechnet.

## Other Day

### TODO

- ~~a2c: speicher episoden im batch. netz nach ungeäfhr 10. done flag mit batch updaten~~
- ~~update aus ganzem batch~~
- ~~Permutation vom raum~~
- ~~gewichted nach gewinn~~

### Quetion to Thomy

```js
// When?
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

- updaten nach X erfolgreichen dones? testen mit 1,5,10,20
- raum wann drehen? nach x erfolgreichen dones? nach jedem gesamt update? -> nach jedem reset: DONE
- andere aggregation? erstmal nicht
- change architecture? weiter investigieren

# Wokring Journal

## 23. Nov

### Stand

Der erste Ansatz mit federated Reinforcement learning ist implementiert. 5 Worker trainieren auf unterschiedlichen rooms Domänen, die mit den weights des _mainAgents_ initsialisiert werden. Nach X Episoden werden deren weights (state dicts) aggregiert und der **Durchschnitt** genommen. Mit diesem Durchschnitt wird der _mainAgent_ gespeist. Dieser durchlauf passiert X mal. Zum Schluss wird der **main Agent** anhand einer _TestDomäne_ im interference mode getestet.
Die Returns der worker werden momentan nicht weiter verarbeitet.

#### Rooms

##### Worker

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

##### Testing

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

#### Durchläufe

| Episoden pro Worker | Updates nach Episoden |
| ------------------- | --------------------- |
| 1000                | 100                   |
| 2500                | 500                   |
| 3000                | 1000                  |

#### Beobachtung

- Worker tuen sich leicht bei v0, v2, v3
- Worker tut schwer bei v1
- Worker tut sich sehr schwer (erreicht fast nie das Ziel) bei v4
- Nach jedem **update** des **mainAgents** laufen v0, v2, v3 besser, jedoch v1 und v4 schlechter
- MainAgent erreicht kein mal das Ziel beim Test von t0

#### Diskussion/Vermutung

- Ungleichgewicht in den Daten/Räumen (bei der Mehrheit der Räumen beginnt der Agent oben und das Ziel ist unten -> es lohnt sich fast immer nach unten zu laufen)
- Einfach den Durchschnitt nehmen ist nicht ausreichend

### TODO

- Returns der Worker als Plot speichern
- Im Plott anzeigen, wann das Netz geupdated wurde (eine vertikale Linie)
- State dicts der Worker unterschiedlich gewichten (z.B. Worker, die seltener das Ziel erreichen, höher gewichten)
- Main Agent Anhand von Räumen(~10 Räume) mit random S und G Punkten im interference modus testen
- Main Agent nach jedem **Update** testen und einen Plot erstellen
- Sachen aufschreiben wenn ich was Teste. Alle Daten sind relevant!!

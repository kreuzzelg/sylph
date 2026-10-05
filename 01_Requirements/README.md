# 01_Requirements (in German)

In Colibri Projekt habe ich beim Testing bemerkt, dass die GGUF Format effizienter als GS64:
https://github.com/JustVugg/colibri/issues/1370

Da der Owner nicht GGUF Format unterstützen möchte, erstelle ich diese Fork mit dem Name sylph (im Sinne einer speziellen Kolibri), um zu prüfen, ob es einen gängbaren Weg gibt. Wenn ja, wie effizient ist es. 

Die Programmierung sollte das Stil von Colibri behalten, in C und ohne Referenz auf anderen Bibliothek. 

Ollama Code kann man als Referenz nehmen. Erstell bitte auch automatische Tests, um sicherzustellen, dass sie äqualent sind. Wenn doch Differenz gäbe, dann muss bewusste Verbesserung sein. 

Bevorzugt ist Testing mit Qwen 3.6 35B Modell, weil dort habe ich am meisten Daten und kann auch mit Ollama bzw. Colibri auf RTX 3070 (CUDA) vergleichen. 

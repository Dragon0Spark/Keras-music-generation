""" Rassemble les musiques """
import midi
import os
instrument_type = ["KeyboardInstrument", "Piano", "Harpsichord", "Clavichord", "Celesta", "Electric Bass", "Electric Guitar", "StringInstrument", ]
pattern = [[], [], [], [], [], [], [], []]
array_notes = len(instrument_type)
total = midi.Pattern()
for i in range(array_notes):   
    if os.path.isfile('music/' + str(instrument_type[i]) + '.mid'):
        pattern = midi.read_midifile('music/' + str(instrument_type[i]) + '.mid')
        for track in pattern:
            total.append(track)

midi.write_midifile('music/TotalSound.mid', total)
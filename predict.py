""" Génère les musiques """
import pickle
import numpy as np
import os
from pathlib import Path
from midi2audio import FluidSynth
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation

def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    instrument_type = ["KeyboardInstrument", "Piano", "Harpsichord", "Clavichord", "Celesta", "Electric Bass", "Electric Guitar", "StringInstrument", ]
    pre_vocab = [[], [], [], [], [], [], [], []]
    notes = [[], [], [], [], [], [], [], []]
    s_length = 30
    array_instru = len(instrument_type)
    for i in range(array_instru):
        print(str(instrument_type[i]))
        if os.path.isfile('data/' + str(instrument_type[i])):
            with open("data/"+ str(instrument_type[i]), 'rb') as filepath:
                notes[i] = pickle.load(filepath)

        array_notes = len(notes)
    for i in range(array_notes):    
        pre_vocab[i].append(len(set(notes[i])))
    
    t_vocab = np.array(pre_vocab)
    vocab = t_vocab.ravel()
    #print(vocab)   

    x, normal_x, pitchnames = prepare_sequences(notes, vocab, s_length)
    #print(len(x[1]))

    for i in range(array_instru):
        print('updates/' + str(instrument_type[i]) + '.hdf5')
        if os.path.isfile('updates/' + str(instrument_type[i]) + '.hdf5'):
            model = create_network(normal_x, vocab, instrument_type, i)
            prediction = generate_notes(model, x[i], pitchnames, vocab, i)
            create_midi(prediction, instrument_type, i)
    
def prepare_sequences(notes, vocab, sequence_length):
    #Prepare les séquences
    x = [[], [], [], [], [], [], [], []]
    normal_x = [[], [], [], [], [], [], [], []]

    array_notes = len(notes)
    result = []
    n_patterns = [[], [], [], [], [], [], [], []]
    for i in range(array_notes):
        result += notes[i]
    
    pitchnames = sorted(set(item for item in result))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    for nb_notes in range(array_notes):
        for i in range(0, len(notes[nb_notes]) - sequence_length, 1):
            sequence_in = notes[nb_notes][i:i + sequence_length]
            x[nb_notes].append([note_to_int[char] for char in sequence_in])
        n_patterns[nb_notes] = len(x[nb_notes])
        normal_x[nb_notes] = np.reshape(x[nb_notes], (n_patterns[nb_notes], sequence_length, 1))
        normal_x[nb_notes] = normal_x[nb_notes] / float(vocab[nb_notes])

    return (x, normal_x, pitchnames)  

def create_network(x, vocab, instrument_type, i):
    #Crée la structure du réseau de neurones
    model = Sequential()
    model.add(LSTM(512, input_shape=(x[i].shape[1], x[i].shape[2]), return_sequences=True ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(228))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.load_weights('updates/' + str(instrument_type[i]) + '.hdf5')

    return model

def generate_notes(model, x, pitchnames, vocab, i):
    start = np.random.randint(0, len(x)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = x[start]
    output = []

    # génère 50 notes
    for note_index in range(50):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(vocab[i])
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return output

def create_midi(prediction, instrument_type, i):
    # crée les fichiers .midi
    offset = 0
    output_notes = []

    for pattern in prediction:
        if pattern != 'Rest':
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    if str(instrument_type[i]) == "KeyboardInstrument":
                        new_note.storedInstrument = instrument.KeyboardInstrument()
                    if str(instrument_type[i]) == "Piano":
                        new_note.storedInstrument = instrument.Piano()
                    if str(instrument_type[i]) == "Harpsichord":
                        new_note.storedInstrument = instrument.Harpsichord()
                    if str(instrument_type[i]) == "Clavichord":
                        new_note.storedInstrument = instrument.Clavichord()
                    if str(instrument_type[i]) == "Celesta":
                        new_note.storedInstrument = instrument.Celesta()
                    if str(instrument_type[i]) == "ElectricBass":
                        new_note.storedInstrument = instrument.ElectricBass()
                    if str(instrument_type[i]) == "ElectricGuitar":
                        new_note.storedInstrument = instrument.ElectricGuitar()
                    if str(instrument_type[i]) == "StringInstrument":
                        new_note.storedInstrument = instrument.StringInstrument()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                if str(instrument_type[i]) == "KeyboardInstrument":
                    new_note.storedInstrument = instrument.KeyboardInstrument()
                if str(instrument_type[i]) == "Piano":
                    new_note.storedInstrument = instrument.Piano()
                if str(instrument_type[i]) == "Harpsichord":
                    new_note.storedInstrument = instrument.Harpsichord()
                if str(instrument_type[i]) == "Clavichord":
                    new_note.storedInstrument = instrument.Clavichord()
                if str(instrument_type[i]) == "Celesta":
                    new_note.storedInstrument = instrument.Celesta()
                if str(instrument_type[i]) == "ElectricBass":
                    new_note.storedInstrument = instrument.ElectricBass()
                if str(instrument_type[i]) == "ElectricGuitar":
                    new_note.storedInstrument = instrument.ElectricGuitar()
                if str(instrument_type[i]) == "StringInstrument":
                    new_note.storedInstrument = instrument.StringInstrument()
                output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp= 'music/' + str(instrument_type[i])+'.mid')

if __name__ == '__main__':
    generate()

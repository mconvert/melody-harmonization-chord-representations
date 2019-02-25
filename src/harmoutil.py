import pickle
import re
import numpy as np
import sklearn.metrics as skl_mtr
from collections import Counter, deque

# General Utility Functions ---------------------------------------------------
def remove_silence_melodies(pairs):
    """ I: list of chord-melody pairs
        O: list of chord-melody pairs with all empty melody pairs removed
    """
    no_silence = []
    for pair in pairs:
        chord_notes = []
        for beat in pair[1]:
            for note in beat:
                chord_notes.append(note)
        if not set(list(map(int, chord_notes))) == {-1}:
            no_silence.append(pair)
    return no_silence


def remove_alt_chords(pairs):
    no_alt = []
    for pair in pairs:
        if not ('alt' in pair[0]):
            no_alt.append(pair)
    return no_alt



def flatten_semitones(data):
    sharps = {'C#':'Db',
              'D#':'Eb',
              'F#':'Gb',
              'G#':'Ab',
              'A#':'Bb',
              'Cb':'B',
              'Fb':'E'}

    pairs = []
    for pair in data:
        chord = pair[0]
        melody = pair[1]

        root, suffix = split_root_suffix(chord)
        if any([sharp in root for sharp in sharps.keys()]):
            chord = sharps[root[:2]]+chord[2:]
        pairs.append((chord, melody))
    return pairs




def load_pickled_data(file):
	""" I: pickle file
		O: pickled object
	"""
	pkl_file = open(file, 'rb')
	data = pickle.load(pkl_file)
	pkl_file.close()
	return data


def split_root_suffix(label):
	""" I: chord label of any length
		O: root string 
           suffix string
	"""
	pattern = "((?:NC)?(?:[A-G][#b]?)?)(.*)"
	m = re.search(pattern, label)
	root = m.group(1)
	suffix = m.group(2)
	return root, suffix


def chord_melody_pairs(data):
	""" I: list of sections, each containing chord sequences with their respective melody sequence
		O: list of chord-melody pairs
	"""
	pairs = []
	for form in data:
		for pair in form:
			pairs.append(pair)
	return pairs



def melody_to_octave_range(data):
    """ I: list of chord-melody pairs
        O: list of chord-melody pairs where melody is now integers from 0 to 12
    """
    pairs = []
    for pair in data:
        chord = pair[0]
        melody = pair[1]
        beats = []
        for beat in melody:
            notes = []
            for note in beat:
                if note == -1:
                    notes.append(-1)
                else:
                    notes.append(int(note)%12)
            beats.append(notes)
        pairs.append((chord, beats))
    return pairs


def notes_and_chords(pairs):
    """ I: list of chord-melody pairs
        O: set of distinct chord labels
           set of distinct melody notes
    """
    midi_notes = []
    chord_labels = []
    for pair in pairs:
        chord_labels.append(pair[0])
        for beat in pair[1]:
            for note in beat:
                midi_notes.append(note)
    return midi_notes, chord_labels


def roots_and_suffixes(chords):
    """ I: list chord labels
        O: set of distinct roots
           set of distinct suffixes
    """
    roots = []
    suffixes = []
    for label in chords:
        root, suffix = split_root_suffix(label)
        roots.append(root)
        suffixes.append(suffix)
    return roots, suffixes


def create_class_mapping(classes_set):
    """ I: set of possible discrete classes
        O: dictionary mapping each class to an integer
    """
    classes = sorted(list(classes_set))
    classes_to_int = dict((c, i) for i, c in enumerate(classes))
    return classes_to_int


# Suffix Truncation -----------------------------------------------------------

def truncate_to_triad(melcho_data):
	""" I: (chord, melody) pair
		O: (chord, melody) pair where chord has been truncated to a triad label
	"""
	chord = melcho_data[0]
	melody = melcho_data[1]
	if chord == 'NC':
		return (chord, melody)
	else:
		triad_pat = '(^[N]?[A-G][#b]?[+osus-]*)'
		m = re.match(triad_pat, chord)
		triad = m.group(1)
		return (triad, melody)


def to_triads(data):
    pairs = []
    for pair in data:
        truncated_pair = truncate_to_triad(pair)
        pairs.append(truncated_pair)
    return pairs

def to_sevenths(data):
    pairs = []
    for pair in data:
        truncated_pair = truncate_to_seventh(pair)
        pairs.append(truncated_pair)
    return pairs

def truncate_to_seventh(melcho_data):
	""" I: (chord, melody) pair
		O: (chord, melody) pair where chord has been truncated to a seventh label
	"""
	chord = melcho_data[0]
	melody = melcho_data[1]
	if chord == 'NC':
		return (chord, melody)
	else:
		seventh_pat = '(^[N]?[A-G][#b]?[+osus-]*[j67mb5]*)'
		m = re.match(seventh_pat, chord)
		seventh = m.group(1)
		return (seventh, melody)


def truncate_to_extended(melcho_data):
	""" I: (chord, melody) pair
		O: (chord, melody) pair where chord has been truncated to an extended label
	"""
	chord = melcho_data[0]
	melody = melcho_data[1]
	if chord == 'NC':
		return (chord, melody)
	else:
		extended_pat = '(^[N]?[A-G][#b]?[+osus-]*[j67mb5]*[alt913#b]*)'
		m = re.match(extended_pat, chord)
		extended = m.group(1)
		return (extended, melody)


def truncate_to_bassnote(melcho_data):
	""" I: (chord, melody) pair
		O: (chord, melody) pair where chord has been truncated to a label including bassnotes
	"""
	chord = melcho_data[0]
	melody = melcho_data[1]
	if chord == 'NC':
		return (chord, melody)
	else:
		bassnote_pat = '(^[N]?[A-G][#b]?[+osus-]*[j67mb5]*[alt913#b]*[/A-Gb#]*)'
		m = re.match(bassnote_pat, chord)
		bassnote = m.group(1)
		return (bassnote, melody)



# Tensor creation functions ---------------------------------------------------

def notes_and_duration(melody):
    """ I: list of list of MIDI notes grouped in beats
        O: list of MIDI notes
           list of duration sequence of forresponding notes
    """
    notes = []
    duration = []
    for beat in melody:                       # for each beat in the chord melody
        for i in range(1, len(beat)+1):       # i is the number of events in each beat
            notes.append(beat[i-1])
            duration.append(1/i)
    return notes, duration



# Counts and statistics -------------------------------------------------------

def max_melody_notes(pairs):
    """ I: list of chord-melody pairs
        O: maximum count of melody notes by chord
    """
    event_count = []
    for pair in pairs:
        notes = []
        for beat in pair[1]:
            for note in beat:
                notes.append(note)
        event_count.append(len(notes))
    return max(event_count)



# Functions for suffix label - sequence conversion ----------------------------

def suffix_label_to_int_sequence(suffix, seqlen):
    """ I: string suffix label (triad, seventh, extended or bassnote)
           expected length of the sequence (triad=4, seventh=5, extended=8, bassnote=8?)
        O: list of integers corresponding to the suffix
    """
    # should end with -1

    def handle_chordal(ch):

        chordal_seqs = {"": [0, 4, 7],      # major
                        "+": [0, 4, 8],     # augmented
                        "-": [0, 3, 7],     # minor
                        "o": [0, 3, 6],     # diminished
                        "sus": [0, 5, 7],   # suspended
                        "j7": [0, 4, 7, 11],    # major 7th
                        "7": [0, 4, 7, 10],     # dominant 7th
                        "+j7": [0, 4, 8, 11],   # augmented major 7th
                        "+7": [0, 4, 8, 10],    # augmented 7th
                        "-j7": [0, 3, 7, 11],   # minor major 7th
                        "-7": [0, 3, 7, 10],    # minor 7th
                        "m7b5": [0, 3, 6, 10],  # half-diminished 7th
                        "o7": [0, 3, 6, 9],     # diminished 7th
                        "sus7": [0, 5, 7, 10],  # suspended 7th
                        "6": [0, 4, 7, 9],      # major 6th
                        "-6": [0, 3, 7, 9]}     # minor 6th

        if ch in chordal_seqs:
            return chordal_seqs[ch]
        else:
            return []
        



    def handle_extended(ex):
        ext_pat = "^((?:9[#b]?)?)((?:11[#b]?)?)((?:13[#b]?)?)"
        m = re.match(ext_pat, ex)
        ninth, eleventh, thirteenth = m.group(1), m.group(2), m.group(3)
        
        ext_seq = []
        if ninth:
            if 'b' in ninth:
                ext_seq += [13]
            elif '#' in ninth:
                ext_seq += [15]
            else:
                ext_seq += [14]
        if eleventh:
            if '#' in eleventh:
                ext_seq += [18]
            else:
                ext_seq += [17]
        if thirteenth:
            if 'b' in thirteenth:
                ext_seq += [20]
            elif '#' in thirteenth:
                ext_seq += [22]
            else:
                ext_seq += [21]
        return ext_seq


    # determine triad/sev part, extended part and bass part
    pattern = "^((?:[+osusj76m7b5-]*)?)((?:[alt913#b]*)?)((?:[/A-G#b]*)?)"
    m = re.match(pattern, suffix)
    chordal, extended, bassnote = m.group(1), m.group(2), m.group(3)

    seq = []
    seq += handle_chordal(chordal)
    seq += handle_extended(extended)
   
    return seq + [-1]*(seqlen - len(seq))



def suffix_label_to_bin_sequence(suffix):
    """ I: string suffix label
        O: list of 0 or 1 for the corresponding suffix
    """
    def handle_chordal(ch):

        chordal_seqs = {"":     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], #[0, 4, 7],      # major
                        "+":    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], #[0, 4, 8],     # augmented
                        "-":    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], #[0, 3, 7],     # minor
                        "o":    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], #[0, 3, 6],     # diminished
                        "sus":  [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], #[0, 5, 7],   # suspended
                        "j7":   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], #[0, 4, 7, 11],    # major 7th
                        "7":    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], #[0, 4, 7, 10],     # dominant 7th
                        "+j7":  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], #[0, 4, 8, 11],   # augmented major 7th
                        "+7":   [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], #[0, 4, 8, 10],    # augmented 7th
                        "-j7":  [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], #[0, 3, 7, 11],   # minor major 7th
                        "-7":   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], #[0, 3, 7, 10],    # minor 7th
                        "m7b5": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], #[0, 3, 6, 10],  # half-diminished 7th
                        "o7":   [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], #[0, 3, 6, 9],     # diminished 7th
                        "sus7": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0], #[0, 5, 7, 10],  # suspended 7th
                        "6":    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0], #[0, 4, 7, 9],      # major 6th
                        "-6":   [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]} #[0, 3, 7, 9]}     # minor 6th

        if ch in chordal_seqs:
            return chordal_seqs[ch]
        else:
            return []


    def handle_extended(ex):
        ext_pat = "^((?:9[#b]?)?)((?:11[#b]?)?)((?:13[#b]?)?)"
        m = re.match(ext_pat, ex)
        ninth, eleventh, thirteenth = m.group(1), m.group(2), m.group(3)

        ext_seq = []
        if ninth:
            if 'b' in ninth:
                ext_seq += [0, 1, 0, 0]# [13]
            elif '#' in ninth:
                ext_seq += [0, 0, 0, 1]# [15]
            else:
                ext_seq += [0, 0, 1, 0]# [14]
        else:
            ext_seq += [0, 0, 0, 0]
        if eleventh:
            if '#' in eleventh:
                ext_seq += [0, 0, 1, 0]#[18]
            else:
                ext_seq += [0, 1, 0, 0]#[17]
        else:
            ext_seq += [0, 0, 0, 0]
        if thirteenth:
            if 'b' in thirteenth:
                ext_seq += [1, 0, 0, 0]#[20]
            elif '#' in thirteenth:
                ext_seq += [0, 0, 1, 0]#[22]
            else:
                ext_seq += [0, 1, 0, 0]#[21]
        else:
            ext_seq += [0, 0, 0, 0]


        return ext_seq

    pattern = "^((?:[+osusj76m7b5-]*)?)((?:[alt913#b]*)?)((?:[/A-G#b]*)?)"
    m = re.match(pattern, suffix)
    chordal, extended, bassnote = m.group(1), m.group(2), m.group(3)    

    seq = []
    seq += handle_chordal(chordal)
    # seq += handle_extended(extended)

    return seq + [-1]




def int_sequence_to_suffix_label(seq, all_chords): 
    """ I: integer sequence (with trailing padding -1 removed)
           set of possible chord labels in the original data
        O: chord label
        /!\ Does not handle 'alt' extended chords and bassnote chords
            This function needs rework, it doesn't catch cases where the extended sequence does not exist. For now, this is handled outside, in the kernel.
    """

    chordal_seqs = {"047": '',      # major
                    "048": '+',     # augmented
                    "037": '-',     # minor
                    "036": 'o',     # diminished
                    "057": 'sus',   # suspended
                    "04711": 'j7',  # major 7th
                    "04710": '7',   # dominant 7th
                    "04811": '+j7', # augmented major 7th
                    "04810": '+7',  # augmented 7th
                    "03711": '-j7', # minor major 7th
                    "03710": '-7',  # minor 7th
                    "03610": 'm7b5',# half-diminished 7th
                    "0369": 'o7',   # diminished 7th
                    "05710": 'sus7',# suspended 7th
                    "0479": '6',    # major 6th
                    "0379": '-6'}   # minor 6th


    # extended_notes = {"13": '9b',
    #                   "14": '9',
    #                   "15": '9#',

    #                   "17": '11',
    #                   "18": '11#',
                      
    #                   "20": '13b',
    #                   "21": '13',
    #                   "22": '13#'}


    chordal_suffix = ''
    extended_suffix = ''

    if seq[-1] == 12:
        seq = seq[:-1]

    # reconstruct the chord label from  input sequence
    if len(seq) == 3:
        chordal_seq = seq[0:3]
        chordal_string_seq = ''.join(str(e) for e in list(map(int, chordal_seq)))
        if chordal_string_seq in chordal_seqs:
            chordal_suffix = chordal_seqs[chordal_string_seq]
    elif len(seq) == 4:
        chordal_seq = seq[0:4]
        chordal_string_seq = ''.join(str(e) for e in list(map(int, chordal_seq)))
        if chordal_string_seq in chordal_seqs:
            chordal_suffix = chordal_seqs[chordal_string_seq]
    # elif len(seq) > 4:
    #     chordal_seq = seq[0:4]
    #     extended_seq = seq[4:]
        
    #     extended_note_exists = True
    #     for i in extended_seq:
    #         if str(int(i)) not in extended_notes:
    #             extended_note_exists = False
    #             break

        # if extended_note_exists:
        #     extended_suffix = ''.join(extended_notes[str(int(e))] for e in extended_seq)

        # chordal_string_seq = ''.join(str(e) for e in list(map(int, chordal_seq)))
        # if chordal_string_seq in chordal_seqs:
        #     chordal_suffix = chordal_seqs[chordal_string_seq]
    
    chord_label = chordal_suffix #+ extended_suffix


    # if the reconstructd chord label actually exists in the original dataset return it, otherwise return None
    if chord_label in all_chords:
        return chord_label
    else:
        return None        




def bin_sequence_to_suffix_label(seq, all_suffixes):
    """ I: binary sequence
           original existing suffix labels
        O: suffix label
        /!\ Does not handle bassnote suffixes
    """

    chordal_seqs = {"100010010000": '', #[0, 4, 7],      # major
                    "100010001000": '+', #[0, 4, 8],     # augmented
                    "100100010000": '-', #[0, 3, 7],     # minor
                    "100100100000": 'o', #[0, 3, 6],     # diminished
                    "100001010000": 'sus', #[0, 5, 7],   # suspended
                    "100010010001": 'j7', #[0, 4, 7, 11],    # major 7th
                    "100010010010": '7', #[0, 4, 7, 10],     # dominant 7th
                    "100010001001": '+j7', #[0, 4, 8, 11],   # augmented major 7th
                    "100010001010": '+7', #[0, 4, 8, 10],    # augmented 7th
                    "100100010001": '-j7', #[0, 3, 7, 11],   # minor major 7th
                    "100100010010": '-7', #[0, 3, 7, 10],    # minor 7th
                    "100100100010": 'm7b5', #[0, 3, 6, 10],  # half-diminished 7th
                    "100100100100": 'o7', #[0, 3, 6, 9],     # diminished 7th
                    "100001010010": 'sus7', #[0, 5, 7, 10],  # suspended 7th
                    "100010010100": '6', #[0, 4, 7, 9],      # major 6th
                    "100100010100": '-6'} #[0, 3, 7, 9]}     # minor 6th

    # ninth_note = {"0100": '9b',
    #               "0010": '9',
    #               "0001": '9#'}

    # eleventh_note = {"0100": '11',
    #                  "0010": '11#'}

    # thirteenth_note = {"1000": '13b',
    #                    "0100": '13',
    #                    "0010": '13#'}

    # make sure the sequence has length 12
    if not len(seq) == 12:
        seq += [0]*(12-len(seq))


    seq = ''.join(str(e) for e in list(map(int, seq)))

    chordal_seq = seq[0:12]
    # extended_seq = seq[12:24]
    # ninth_seq = extended_seq[0:4]
    # eleventh_seq = extended_seq[4:8]
    # thirteenth_seq = extended_seq[8:12]


    chordal_suffix = ''
    extended_suffix = ''


    # reconstruct the chord label from the bin sequence
    if chordal_seq in chordal_seqs:
        chordal_suffix += chordal_seqs[chordal_seq]
    # if ninth_seq in ninth_note:
    #     extended_suffix += ninth_note[ninth_seq]
    # if eleventh_seq in eleventh_note:
    #     extended_suffix += eleventh_note[eleventh_seq]
    # if thirteenth_seq in thirteenth_note:
    #     extended_suffix += thirteenth_note[thirteenth_seq]

    chord_label = chordal_suffix #+ extended_suffix

    # if the reconstructed chord label actually exists, return it, otherwise None
    if chord_label in all_suffixes:
        return chord_label
    else:
        return None




def chord_label_to_int_sequence(root, suffix, seqlen):
    """ I: root label
           suffix label
           expected sequence length
        O: integer sequence
    """

    def get_root_offset(rt):
        root_offset = {'A': 0,
                       'A#': 1,
                       'Bb': 1,
                       'B': 2,
                       'Cb': 2,
                       'C': 3,
                       'C#': 4,
                       'Db': 4,
                       'D': 5,
                       'D#': 6,
                       'Eb':6,
                       'E':7,
                       'E#':8,
                       'Fb':7,
                       'F':8,
                       'F#':9,
                       'Gb':9,
                       'G':10,
                       'G#':11}
        
        intervals = {''}



def count_each_category(pairs):
    '(^[N]?[A-G][#b]?[+osus-]*[j67mb5]*[alt913#b]*[/A-Gb#]*)'
    


# Numpy and general machine learning utility functions ------------------------

def softmax_to_argmax(a):
    """ I: softmax array
        O: argmax array (with the largest value index set to 1, the rest at 0)
    """
    return (a == a.max(axis=1)[:,None]).astype(float)



def get_class_weights(class_occurrences):
    """ I: list of ocurring classes (of length num_samples)
        O: dictionary of class weights (integer, weight)
    """
    counter = Counter(class_occurrences)
    majority = max(counter.values())
    class_index = sorted(list(set(class_occurrences)))
    return  {class_index.index(cls_l): float(majority/count) for cls_l, count in counter.items()}


def compute_accuracy_score(y_true, y_pred):
    """ I: array of original target data
           array of predicted values
        O: accuracy score
    """
    correct_preds = [np.argmax(true) == np.argmax(pred) for (true, pred) in zip(y_true, y_pred)]
    return correct_preds.count(True)/len(correct_preds)

def compute_binary_accuracy_score(y_true, y_pred):
    """ I: array of original target data
           array of predicted values
        O: binary accuracy score
    """
    correct_preds = [round(true[0]) == round(pred[0]) for (true, pred) in zip(y_true, y_pred)]
    return correct_preds.count(True)/len(correct_preds)

def compute_multiclass_binary_accuracy_score(y_true, y_pred):
    """ I: array of original target data of binary vector of notes
           array of predicted binary values of notes
        O: accuracy score of correctly predicted notes
    """
    correct_preds = []
    for true, pred in zip(y_true, y_pred):
        for true_note, pred_note in zip(true, pred):
            correct_preds.append(round(true_note) == round(pred_note))
    return correct_preds.count(True)/len(correct_preds)


def compute_recall_score(t_true, y_pred):
    """ I: array of original target data
           array of predicted values
        O: recall score
    """
    y_true_classes = np.zeros(y_true.shape[0])
    y_pred_classes = np.zeros(y_pred.shape[0])
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        y_true_classes[i] = true.argmax()
        y_pred_classes[i] = pred.argmax()    

    cnf_matrix = skl_mtr.confusion_matrix(y_true_classes, y_pred_classes)

    recalls = []
    for i in range(cnf_matrix.shape[0]):
        rec = cnf_matrix[i,i]/sum(cnf_matrix[i,:])
        recalls.append(rec)
    return sum(recalls)/len(recalls)


def compute_precision_score(t_true, y_pred):
    """ I: array of original target data
           array of predicted values
        O: precision score
    """
    y_true_classes = np.zeros(y_true.shape[0])
    y_pred_classes = np.zeros(y_pred.shape[0])
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        y_true_classes[i] = true.argmax()
        y_pred_classes[i] = pred.argmax()    

    cnf_matrix = skl_mtr.confusion_matrix(y_true_classes, y_pred_classes)

    precisions = []
    for i in range(cnf_matrix.shape[0]):
        pre = cnf_matrix[i,i]/sum(cnf_matrix[:,i])
        precisions.append(pre)
    return sum(precisions)/len(precisions)


def compute_kappa_score(y_true, y_pred):
    """ I: array of original target data
           array of predicted class (argmax, not softmax)
        O: kappa score
    """

    y_true_classes = np.zeros(y_true.shape[0])
    y_pred_classes = np.zeros(y_pred.shape[0])
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        y_true_classes[i] = true.argmax()
        y_pred_classes[i] = pred.argmax()

    cnf_matrix = skl_mtr.confusion_matrix(y_true_classes, y_pred_classes)
    
    observed_acc = cnf_matrix.diagonal().sum()/cnf_matrix.sum()
    expected_acc = 0
    for i in range(cnf_matrix.shape[0]):
        expected_acc += (cnf_matrix[:,i].sum() * cnf_matrix[i,:].sum())/cnf_matrix.sum()
    expected_acc = expected_acc/cnf_matrix.sum()
    
    kappa = (observed_acc - expected_acc)/(1 - expected_acc)
    return kappa


def compute_binary_kappa_score(y_true, y_pred):
    """ I: array of original target data
           array of predicted 
        O: binary kappa score
    """
    
    y_true_classes = np.zeros(y_true.shape[0])
    y_pred_classes = np.zeros(y_pred.shape[0])
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        y_true_classes[i] = round(true[0])
        y_pred_classes[i] = round(pred[0])

    cnf_matrix = skl_mtr.confusion_matrix(y_true_classes, y_pred_classes)

    observed_acc = cnf_matrix.diagonal().sum()/cnf_matrix.sum()
    expected_acc = 0
    for i in range(cnf_matrix.shape[0]):
        expected_acc += (cnf_matrix[:,i].sum() * cnf_matrix[i,:].sum())/cnf_matrix.sum()
    expected_acc = expected_acc/cnf_matrix.sum()
    
    kappa = (observed_acc - expected_acc)/(1 - expected_acc)
    return kappa    


def compute_multiclass_binary_kappa_score(y_true, y_pred):
    """ I: 
        Same as normal kappa score, except the 
    """

    y_true_classes_list = []
    y_pred_classes_list = []
    for true, pred in zip(y_true, y_pred):
        for true_note, pred_note in zip(true, pred):
            y_true_classes_list.append(true_note)
            y_pred_classes_list.append(pred_note)

    y_true_classes = np.zeros(len(y_true_classes_list))
    y_pred_classes = np.zeros(len(y_pred_classes_list))
    for i, (true, pred) in enumerate(zip(y_true_classes_list, y_pred_classes_list)):
        y_true_classes[i] = round(true)
        y_pred_classes[i] = round(pred)

    cnf_matrix = skl_mtr.confusion_matrix(y_true_classes, y_pred_classes)
    tn, fp, fn, tp = skl_mtr.confusion_matrix(y_true_classes, y_pred_classes).ravel()
    print('TP:', tp, 'TN:', tn, 'FP:', fp, 'FN:', fn) # used these values in paper

    observed_acc = cnf_matrix.diagonal().sum()/cnf_matrix.sum()
    expected_acc = 0
    for i in range(cnf_matrix.shape[0]):
        # print(cnf_matrix[:,i].sum(), cnf_matrix[i,:].sum())
        expected_acc += (cnf_matrix[:,i].sum() * cnf_matrix[i,:].sum())/cnf_matrix.sum()
    expected_acc = expected_acc/cnf_matrix.sum()
    
    kappa = (observed_acc - expected_acc)/(1 - expected_acc)
    return kappa



def compute_binary_fscore(y_true, y_pred):
    """ I: 
    """

    y_true_classes_list = []
    y_pred_classes_list = []
    for true, pred in zip(y_true, y_pred):
        for true_note, pred_note in zip(true, pred):
            y_true_classes_list.append(true_note)
            y_pred_classes_list.append(pred_note)

    y_true_classes = np.zeros(len(y_true_classes_list))
    y_pred_classes = np.zeros(len(y_pred_classes_list))
    for i, (true, pred) in enumerate(zip(y_true_classes_list, y_pred_classes_list)):
        y_true_classes[i] = round(true)
        y_pred_classes[i] = round(pred)

    F1 = skl_mtr.f1_score(y_true_classes, y_pred_classes)
    return F1



# Data augmentation ------------------------------------------


def transpose_chord(chord, offset):
    """ transposes one chord offset amount
    """
    index_root = {0:'C', 1:'Db', 2:'D', 3:'Eb', 4:'E', 5:'F', 6:'Gb', 7:'G', 8:'Ab', 9:'A', 10:'Bb', 11:'B'}
    root_index = {'C':0, 'Db':1, 'D':2, 'Eb':3, 'E':4, 'F':5, 'Gb':6, 'G':7, 'Ab':8, 'A':9, 'Bb':10, 'B':11}

    if not (chord == 'NC'):
        root, suffix = split_root_suffix(chord)
        transposed_root = index_root[(root_index[root] + offset)%12]
        transposed_chord = transposed_root+suffix
    else:
        transposed_chord = chord
    # print(transposed_chord)
    return transposed_chord



def transpose_melody(melody, offset):
    """ transposes one melody sequence (list of list) offset amount
    """
    transposed_melody = []

    for beat in melody:
        tr_beat = []
        for note in beat:
            if not (note == -1.0):
                tr_beat.append(note+offset)
            else:
                tr_beat.append(note)
        transposed_melody.append(tr_beat)

    return transposed_melody


def transpose_section(section, offset):
    """ transposes entire input section (chord and melody) offset amount (positive or negative integer)
    """

    transposed_section = []

    for chord_info in section:
        chord = chord_info[0]
        melody = chord_info[1]
        transposed_chord = transpose_chord(chord, offset)
        transposed_melody = transpose_melody(melody, offset)
        transposed_section.append((transposed_chord, transposed_melody))

    return transposed_section


def shift_range(offset_range, amount):
    return (offset_range[0]+amount, offset_range[1]+amount)


def get_offset_range(data_min, data_max, section_min, section_max):
    """ compute the 
    """


    offset_range = (-6, 6)

    lower_limit = int(data_min - section_min)
    upper_limit = int(data_max - section_max)+1

    # if (offset_range[0]<lower_limit) or (offset_range[1]>upper_limit):
    #     print("lower:", offset_range[0], lower_limit, "upper:", offset_range[1], upper_limit)


    if (offset_range[0]<lower_limit):
        diff = -(offset_range[0]-lower_limit)
        offset_range = shift_range(offset_range, diff)
    elif (offset_range[1]>upper_limit):
        diff = -(offset_range[1]-upper_limit)
        offset_range = shift_range(offset_range, diff)

    # print("lower:", offset_range[0], lower_limit, "upper:", offset_range[1], upper_limit)


    return offset_range


def get_all_transpositions(section, data_min, data_max):
    """ I: list for one section of chord-melody pairs
        O: list of twelve transpositions (1 original and 11 transpositions) of the input section
        D: computes which way to transpose depending on mel_min and mel_max. 
    """
    transpositions = []

    section_min, section_max = get_min_max_notes_section(section)
    offset_range = get_offset_range(data_min, data_max, section_min, section_max)

    for i in range(*offset_range):
        transpositions.append(transpose_section(section, i))

    return transpositions


def get_min_max_notes_data(data):
    """ find lowest and highest note in dataset.
    """
    melody_notes = []
    for section in data:
        for chord_info in section:
            melody = chord_info[1]
            chord_notes = [note for beat in melody for note in beat if not note == -1.0]
            melody_notes += chord_notes
    mel_min = min(melody_notes)
    mel_max = max(melody_notes)
    return mel_min, mel_max

def get_min_max_notes_section(section):
    """ find lowest and highest note in section
    """
    melody_notes = []
    for chord_info in section:
        melody = chord_info[1]
        chord_notes = [note for beat in melody for note in beat if not note == -1.0]
        melody_notes += chord_notes

    if not melody_notes == []:
        mel_min = min(melody_notes)
        mel_max = max(melody_notes)

    return mel_min, mel_max

def section_is_empty(section):
    is_empty = False
    chords = []
    notes = []
    for chord_info in section:
        melody = chord_info[1]
        chord = chord_info[0]
        chords.append(chord)
        notes += [note for beat in melody for note in beat]

    if list(set(chords)) == ['NC'] or list(set(notes)) == [-1.0]:
        is_empty = True
    return is_empty



def transpose_and_augment_data(data):
    """ I: list of sections
        O: list of sections with each corresponding transpose sections
    """

    mel_min, mel_max = get_min_max_notes_data(data)

    augmented_data = []
    for section in data:
        if not section_is_empty(section):
            section = flatten_semitones(section)
            transpositions = get_all_transpositions(section, mel_min, mel_max)
            augmented_data += transpositions

    return augmented_data



# Model 4 functions ---------------------------------------------------------------


def root_position(suffix):
    """ I: string suffix label
        O: list of notes presence in root position
    """
    chordal_seqs = {"":     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], #[0, 4, 7],      # major
                    "+":    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], #[0, 4, 8],     # augmented
                    "-":    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], #[0, 3, 7],     # minor
                    "o":    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], #[0, 3, 6],     # diminished
                    "sus":  [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], #[0, 5, 7],   # suspended
                    "j7":   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], #[0, 4, 7, 11],    # major 7th
                    "7":    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], #[0, 4, 7, 10],     # dominant 7th
                    "+j7":  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], #[0, 4, 8, 11],   # augmented major 7th
                    "+7":   [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], #[0, 4, 8, 10],    # augmented 7th
                    "-j7":  [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], #[0, 3, 7, 11],   # minor major 7th
                    "-7":   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], #[0, 3, 7, 10],    # minor 7th
                    "m7b5": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], #[0, 3, 6, 10],  # half-diminished 7th
                    "o7":   [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], #[0, 3, 6, 9],     # diminished 7th
                    "sus7": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0], #[0, 5, 7, 10],  # suspended 7th
                    "6":    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0], #[0, 4, 7, 9],      # major 6th
                    "-6":   [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]} #[0, 3, 7, 9]}     # minor 6th

    if suffix in chordal_seqs:
        return chordal_seqs[suffix]
    else:
        return []


def root_index(root):
    """ I: string root label
        O: index of the input note starting counting C
    """
    notes_index = {'C': 0,
             'Db': 1,
             'D': 2,
             'Eb': 3,
             'E': 4,
             'F': 5,
             'Gb': 6,
             'G': 7,
             'Ab': 8,
             'A': 9,
             'Bb': 10,
             'B': 11}    

    if root in notes_index:
        return notes_index[root]
    else:
        return None


def shift(l, n):
    # return l[n:] + l[:n]
    return l[-n:] + l[:-n]



def chord_to_notes(ch):
    """ I: chord label
        O: list of length 12 with the notes in the input chord set to 1
    """
    rt, sfx = split_root_suffix(ch)
    if not rt == 'NC': 
        notes = root_position(sfx)
        shift_amt = root_index(rt)
        notes = shift(notes, shift_amt)
    else:
        notes = [0]*12
    return notes


def chord_to_window(ch):
    """ I: chord label
        O: list of constituting pitches over a two octave window 
    """
    rt, sfx = split_root_suffix(ch)
    if not rt == 'NC':
        notes = root_position(sfx)
        shift_amt = root_index(rt)
        notes = [0]*shift_amt + notes + [0]*(24-len([0]*shift_amt + notes))
    else:
        notes = [0]*24
    return notes


def notes_to_chords(notes):
    """ I: list of length 12 with notes presence
        O: chord label corresponding to input notes
    """
    pass























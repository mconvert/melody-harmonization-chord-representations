import harmoutil as harmo
from midiutil import MIDIFile
from chord_struct import ChordStruct
import pickle
import re
import os

def write_harmony(filename, harmony, signature):
	""" I: List of list of chord labels
		   String time signature ("4/4" or "3/4")
		O: MIDI object with corresponding chord notes
	"""
	track	= 0
	time	 = 0	# In beats
	tempo	= 120   # In BPM

	MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
						  # automatically)
	MyMIDI.addTempo(track, time, tempo)

	harmo_data = get_song_degrees_durations_times(harmony, signature)
	midi_events = get_midi_events(harmo_data)
	
	for e in midi_events:
		MyMIDI.addNote(*e)

	with open(filename, "wb") as output_file:
		MyMIDI.writeFile(output_file)



def get_midi_events(harmony_data):
	"""
	"""
	track = 0
	channel = 0
	volume = 100

	events = []
	for bar_degrees, bar_durations, bar_times in harmony_data:
		for degs, dur, time in zip(bar_degrees, bar_durations, bar_times):
			for deg in degs:
				ev = (track, channel, deg, time, dur, volume)
				events.append(ev)
	return events



def get_song_degrees_durations_times(chords, sig):
	bar_len = 3 if sig == "3/4" else 4

	harmony = []
	for i, bar in enumerate(chords):
		degrees, durations, times = get_bar_degrees_durations_times(bar, bar_len, i*bar_len)
		harmony.append((degrees, durations, times))

	return harmony



def get_bar_degrees_durations_times(bar_chords, bar_len, bar_offset):
	""" I: List of chord labels for one bar
		O: List of degrees for the given bar
	"""
	bar_durations = []
	bar_degrees = []
	bar_times = []
	for i, ch in enumerate(bar_chords):
		bar_degrees.append(get_chord_degrees(ch))
		bar_durations.append(bar_len // len(bar_chords))
		bar_times.append(bar_offset+(bar_len - (len(bar_chords)-i)*(bar_len//len(bar_chords))))
	return bar_degrees, bar_durations, bar_times



def get_chord_degrees(chord):
	""" I: Chord object as defined in current file
	"""
	harmony_range = [i for i in range(48, 72)]

	if (chord.repr == "label") or (chord.repr == "rootsuffix"):
		root_index = harmo.root_index(chord.root)
		intervals = harmo.root_position(chord.suffix)
		relevant_notes = harmony_range[root_index:root_index+len(intervals)]
		degrees = [note for note, pres in zip(relevant_notes, intervals) if pres]
	elif (chord.repr == "rootintervals"):
		root_index = harmo.root_index(chord.root)
		intervals = chord.intervals
		relevant_notes = harmony_range[root_index:root_index+len(intervals)]
		degrees = [note for note, pres in zip(relevant_notes, intervals) if pres]
	elif (chord.repr == "pitch"):
		degrees = [note for note, pres in zip(harmony_range[:12], chord.pitches) if pres]
	else:
		print("Unrecognized chord representation token:", chord.repr)

	return degrees



def read_melody_file_by_bar(infile):
	""" I: String input file name
		O: List of melody by bar
	"""
	with open(infile, 'r') as f:
		content = f.read().split('\n')
		mel_content = '|'.join(content)
		melody = [[note for note in mel.replace(' / ', ' ').strip().split(' ')] for mel in mel_content.split('|')]
	return melody


def read_melody_file_by_chord(infile):
	""" I: String input file name
		O: List of melody notes by chord
	"""
	with open(infile, 'r') as f:
		content = f.read().split('\n')
		mel_content = '|'.join(content)
		melody = [mel.strip().split(' ') for mel in re.findall(r"[A-Gb ]+", mel_content)]
	return melody


def read_chords_file_by_bar(infile):
	""" I: String input file name
		O: List of chords by bars
	"""
	with open(infile, 'r') as f:
		content = f.read().split('\n')
		chord_content = '|'.join(content)
		chords = [[ch for ch in bar.strip().split(' ')] for bar in chord_content.split('|')]
	return chords


def read_chords_file_by_chord(infile):
	with open(infile, 'r') as f:
		content = f.read().split('\n')
		chord_content = '|'.join(content)
		chords = [ch for bar in chord_content.split('|') for ch in bar.strip().split(' ')]
	return chords	


# def read_chords_pick_by_barle(infile):
# 	""" I: String file name
# 		O: 
# 	"""

def read_signature_file(infile):
	""" I: String input file name
		O: String time signature token
	"""
	with open(infile, 'r') as f:
		content = f.read()
		signature = content.strip()
	return signature


def get_melody_degrees_durations_times(melody, sig):
	bar_len = 3 if sig == "3/4" else 4

	melody_data = []
	for i, bar_mel in enumerate(melody):
		for j, note in enumerate(bar_mel):
			degree = get_note_degree(note)
			duration = bar_len*1/len(bar_mel)
			time = i*bar_len + j*duration
			melody_data.append((degree, duration, time))
	return melody_data


def get_note_degree(note):
	melody_range = [i for i in range(72, 84)]
	return melody_range[harmo.root_index(note)]


def get_melody_midi_events(mel_data):
	track = 0
	channel = 0
	volume = 100

	events = []
	
	for deg, dur, time in mel_data:
		ev = (track, channel, deg, time, dur, volume)
		events.append(ev)
	return events


def write_melody(filename, melody, signature):
	""" I: List of melody note strings
	""" 
	track	= 0
	time	 = 0	# In beats
	tempo	= 120   # In BPM

	MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
						  # automatically)
	MyMIDI.addTempo(track, time, tempo)


	melody_data = get_melody_degrees_durations_times(melody, signature)
	midi_events = get_melody_midi_events(melody_data)
	
	for e in midi_events:
		MyMIDI.addNote(*e)

	with open(filename, "wb") as output_file:
		MyMIDI.writeFile(output_file)	


def get_song_name(filename):
	""" I: String file name
		O: Song name
	"""
	p = re.compile("^[a-z_]+")
	m = p.search(filename)
	return m.group()

def to_chord_struct(representation, chord):
	""" I: String representation token
		   Chord variable (label, root-suffix pair, root-intervals pair, pitches)
		O: Corresponding ChordStruct instance
	"""
	if representation == "label":
		return ChordStruct(representation, label=chord)
	elif representation == "rootsuffix":
		return ChordStruct(representation, root=chord[0], suffix=chord[1])
	elif representation == "rootintervals":
		return ChordStruct(representation, root=chord[0], intervals=chord[1])
	elif representation == "pitch":
		return ChordStruct(representation, pitches=chord)
	else:
		print("Unrecognized representation token!")



def check_files_content(signature_dir, melody_dir, harmony_dir):
	""" I: 
		O: 
	"""
	for sig_f in os.listdir(signature_dir):
		if not ".signature" in sig_f:
			continue
		song_name = get_song_name(sig_f)
		melody_f = melody_dir + song_name + ".melody"
		harmony_f = harmony_dir + song_name + ".chords"
		melody = read_melody_file_by_chord(melody_f)
		harmony = read_chords_file_by_chord(harmony_f)
		


if __name__ == "__main__":

	# Input data directories
	pathroot = "../harmony_generation/"

	in_melody_path = pathroot + "input_data/melody_files/"
	in_harmony_path = pathroot + "input_data/truth_harmony_files/"
	in_signature_path = pathroot + "input_data/time_signature_files/"

	in_label_path = pathroot + "label_harmonization/label_harmony_files/"
	in_rootsuffix_path = pathroot + "rootsuffix_harmonization/rootsuffix_harmony_files/"
	in_rootintervals_path = pathroot + "rootintervals_harmonization/rootintervals_harmony_files/"
	in_pitch_path = pathroot + "pitch_harmonization/pitch_harmony_files/"
	in_embeddings_path = pathroot + "embeddings_harmonization/embeddings_harmony_files/"


	# Output midi directories
	outdir = "output_midi/"
	out_melody_path = pathroot + outdir + "melody_midi/"
	out_harmony_path = pathroot + outdir + "harmony_midi/"
	out_label_path = pathroot + outdir + "label_midi/"
	out_rootsuffix_path = pathroot + outdir + "rootsuffix_midi/"
	out_rootintervals_path = pathroot + outdir + "rootintervals_midi/"
	out_pitch_path = pathroot + outdir + "pitch_midi/"
	out_embeddings_path = pathroot + outdir + "embeddings_midi/"

	check_files_content(in_signature_path, in_melody_path, in_harmony_path)

	print("Generating melody and original harmony MIDI files:")
	# Generate melody and original harmony midi
	for song_file in os.listdir(in_signature_path):
		if not ".signature" in song_file:
			continue	
		song_name = get_song_name(song_file)
		harmony_f = in_harmony_path + song_name + ".chords"
		melody_f = in_melody_path + song_name + ".melody"
		signature_f = in_signature_path + song_name + ".signature"

		signature = read_signature_file(signature_f)
		melody = read_melody_file_by_bar(melody_f)
		harmony = [[to_chord_struct('label', ch) for ch in bar] for bar in read_chords_file_by_bar(harmony_f)]


		melody_out_f = out_melody_path + song_name + "_melody.mid"
		harmony_out_f = out_harmony_path + song_name + "_harmony.mid"
		write_melody(melody_out_f, melody, signature)
		write_harmony(harmony_out_f, harmony, signature)
		print(song_name)




	print("-------------------------------------")
	print("Generating models harmony MIDI files:")
 	# Generate harmony from model predictions
	input_pred_dirs = [in_label_path, in_rootsuffix_path, in_rootintervals_path, in_pitch_path, in_embeddings_path]
	output_midi_dirs = [out_label_path, out_rootsuffix_path, out_rootintervals_path, out_pitch_path, out_embeddings_path]
	repr_tokens = ['label', 'rootsuffix', 'rootintervals', 'pitch', 'embeddings']
	
	for in_dir, out_dir, chord_repr in zip(input_pred_dirs, output_midi_dirs, repr_tokens):
		print("\n" + chord_repr.upper() + "\n" + "-"*len(chord_repr))
		if chord_repr == "embeddings":
			chord_repr = "label"
		for song_file in os.listdir(in_signature_path):
			if not ".signature" in song_file:
				continue
			song_name = get_song_name(song_file)
			signature_f = in_signature_path + song_name + ".signature"
			harmony_f = in_dir + song_name + ".pkl"
			signature = read_signature_file(signature_f)
			
			repr_harmony = [[to_chord_struct(chord_repr, ch) for ch in bar] for bar in pickle.load(open(harmony_f, 'rb'))]
			harmony_out_f = out_dir + song_name + ".mid"
			
			# print(pickle.load(open(harmony_f, 'rb')))
			write_harmony(harmony_out_f, repr_harmony, signature)
			print(song_name)








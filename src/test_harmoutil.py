import pytest
import harmoutil

def test_chord_to_notes_example():
	label = 'E6'
	assert harmoutil.chord_to_notes(label) == [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]


def test_chord_to_window():
	label = 'E-7'
	assert harmoutil.chord_to_window(label) == [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
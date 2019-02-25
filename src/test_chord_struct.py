import pytest
from chord_struct import ChordStruct

def test_label_representation_root():
	label = 'E-7'
	c = ChordStruct('label', label=label)
	assert c.root == 'E'

def test_label_representation_suffix():
	label = 'E-7'
	c = ChordStruct('label', label=label)
	assert c.suffix == '-7'

def test_label_representation_intervals():
	label = 'E-7'
	c = ChordStruct('label', label=label)
	assert c.intervals == [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]

def test_label_representation_pitches():
	label = 'E-7'
	c = ChordStruct('label', label=label)
	assert c.pitches == [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1]

def test_rootsuffix_representation_label():
	root = 'G'
	suffix = 'j7'
	c = ChordStruct('rootsuffix', root=root, suffix=suffix)
	assert c.label == 'Gj7'

def test_rootsuffix_representation_intervals():
	root = 'G'
	suffix = 'j7'
	c = ChordStruct('rootsuffix', root=root, suffix=suffix)
	assert c.intervals == [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]

def test_rootsuffix_representation_pitches():
	root = 'G'
	suffix = 'j7'
	c = ChordStruct('rootsuffix', root=root, suffix=suffix)
	assert c.pitches == [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1]

def test_rootintervals_representation_pitches():
	root = 'G'
	intervals = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
	c = ChordStruct('rootintervals', root=root, intervals=intervals)
	assert c.pitches == [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
	
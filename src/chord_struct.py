import harmoutil



class ChordStruct():

	def __init__(self, representation, label=None, root=None, suffix=None, intervals=None, pitches=None):	
		self.repr = representation
	
		if (self.repr == "label"):
			self._label = label
			self._root = harmoutil.split_root_suffix(self._label)[0]
			self._suffix = harmoutil.split_root_suffix(self._label)[1]
			self._intervals = harmoutil.root_position(self._suffix)
			self._pitches = harmoutil.chord_to_notes(self._label)

		if (self.repr == "rootsuffix"):
			self._root = root
			self._suffix = suffix
			self._label = self._root + self._suffix
			self._intervals = harmoutil.root_position(self._suffix)
			self._pitches = harmoutil.chord_to_notes(self._label)

		if (self.repr == "rootintervals"):
			self._root = root
			self._intervals = intervals
			self._pitches = harmoutil.shift(self._intervals, harmoutil.root_index(self._root))

		if (self.repr == "pitch"):
			self._pitches = pitches



	@property
	def label(self):
		return self._label

	@label.setter
	def label(self, lab):
		self._label = lab
		self._root = harmoutil.split_root_suffix(lab)[0]
		self._suffix = harmoutil.split_root_suffix(lab)[1]
		self._intervals = harmoutil.root_position(self._suffix)
		self._pitches = harmoutil.chord_to_notes(self._label)

	@property
	def root(self):
		return self._root

	@root.setter
	def root(self, rt):
		self._root = rt
		self._label = rt + self._suffix
		self._pitches = harmoutil.shift(self._intervals, harmoutil.root_index(self._root))

	@property
	def suffix(self):
		return self._suffix

	@suffix.setter
	def suffix(self, sf):
		self._suffix = sf
		self._label = self._root + sf
		self._intervals = harmoutil.root_position(sf)
		self._pitches = harmoutil.chord_to_notes(self._label)

	@property
	def intervals(self):
		return self._intervals

	@intervals.setter
	def intervals(self, vals):
		self._intervals = vals
		self._pitches = harmoutil.shift(self._intervals, harmoutil.root_index(self._root))

	@property
	def pitches(self):
		return self._pitches
	
	@pitches.setter
	def pitches(self, vals):
		self._pitches = vals








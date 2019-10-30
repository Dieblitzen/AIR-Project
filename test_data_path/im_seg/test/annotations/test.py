for i in range(8):
	s = ''
	s += str((i >> 2) % 2)
	s += str((i >> 1) % 1)
	s += str(i % 2)
	print(s)
	print('-----')


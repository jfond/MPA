

text_file = open('C:\Users\Camera\Desktop\Two Camera Imaging\Data\Trial20\\Timestamps_PS3_Vid3.txt', "r")
ps3_timestamp_array = text_file.readlines()
for n in range(len(ps3_timestamp_array)):
	ps3_timestamp_array[n] = float(ps3_timestamp_array[n])

